from modal import App, Image, Secret
import modal # Import modal itself to access modal.Volume

# Persisted volume to store FAISS indexes
# Updated: Use modal.Volume.from_name instead of Volume.persisted
volume = modal.Volume.from_name("faiss-index-storage", create_if_missing=True)

# Modal container image
image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "faiss-cpu", "torch", "numpy", "ftfy", "regex", "tqdm", "requests",
        "Pillow", "supabase", "python-dotenv"
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
)

app = App("build-faiss-all", image=image, secrets=[Secret.from_name("supabase-creds")])

@app.function(
    volumes={"/data": volume},
    timeout=3600,
    gpu="A10G",
    secrets=[Secret.from_name("supabase-creds")]
)
def build_all_indexes():
    import os, json
    import numpy as np
    import faiss
    import torch
    import clip
    import requests
    from io import BytesIO
    from tqdm import tqdm
    from PIL import Image, UnidentifiedImageError
    from supabase import create_client
    from dotenv import load_dotenv

    load_dotenv()
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("📦 Fetching product images from Supabase...")
    all_images = []
    page = 0
    while True:
        resp = supabase.table("product_images").select("id, image_url") \
            .range(page * 1000, (page + 1) * 1000 - 1).execute()
        batch = resp.data
        if not batch:
            break
        all_images.extend(batch)
        page += 1

    print(f"✅ Retrieved {len(all_images)} images")
    valid_images = [img for img in all_images if img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))]

    # Checkpoint logic
    progress_path = "/data/progress.json"
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            seen_ids = set(json.load(f))
    else:
        seen_ids = set()

    color_vecs, structure_vecs, combined_vecs = [], [], []
    id_map = []

    for img in tqdm(valid_images):
        if img["id"] in seen_ids:
            continue

        try:
            r = requests.get(img["image_url"], timeout=10)
            if not r.headers.get("Content-Type", "").startswith("image/"):
                continue

            pil_color = Image.open(BytesIO(r.content)).convert("RGB")
            pil_grey = pil_color.convert("L").convert("RGB")

            color_tensor = preprocess(pil_color).unsqueeze(0).to(device)
            grey_tensor = preprocess(pil_grey).unsqueeze(0).to(device)

            with torch.no_grad():
                color_feat = model.encode_image(color_tensor)
                color_feat /= color_feat.norm(dim=-1, keepdim=True)

                grey_feat = model.encode_image(grey_tensor)
                grey_feat /= grey_feat.norm(dim=-1, keepdim=True)

                combined_feat = (color_feat + grey_feat) / 2
                combined_feat /= combined_feat.norm(dim=-1, keepdim=True)

            color_vecs.append(color_feat.cpu().numpy().flatten())
            structure_vecs.append(grey_feat.cpu().numpy().flatten())
            combined_vecs.append(combined_feat.cpu().numpy().flatten())
            id_map.append(img["id"])
            seen_ids.add(img["id"])

            if len(seen_ids) % 100 == 0:
                with open(progress_path, "w") as f:
                    json.dump(list(seen_ids), f)

        except (UnidentifiedImageError, Exception) as e:
            print(f"❌ Failed on {img['image_url']}: {e}")
            continue

    def save_index(vectors, index_path, map_path):
        arr = np.stack(vectors).astype(np.float32)
        faiss.normalize_L2(arr)
        idx = faiss.IndexFlatIP(arr.shape[1])
        idx.add(arr)
        faiss.write_index(idx, index_path)
        with open(map_path, "w") as f:
            json.dump(id_map, f)

    def conditional_save(name, vectors, index_path, map_path):
        if vectors:
            print(f"💾 Saving {name} index with {len(vectors)} vectors...")
            save_index(vectors, index_path, map_path)
        else:
            print(f"⚠️ Skipping {name} index — no new vectors.")

    print("💾 Saving all FAISS indexes...")
    conditional_save("color", color_vecs, "/data/clip_color.index", "/data/id_map_color.json")
    conditional_save("structure", structure_vecs, "/data/clip_structure.index", "/data/id_map_structure.json")
    conditional_save("combined", combined_vecs, "/data/clip_combined.index", "/data/id_map_combined.json")

    with open(progress_path, "w") as f:
        json.dump(list(seen_ids), f)

    print("✅ All indexes and progress saved to /data")

# This line is crucial for Modal to trigger the job on deploy
if __name__ == "__main__":
    with app.run():
        build_all_indexes.remote()
