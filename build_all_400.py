from modal import App, Image, Secret
import os, json
import numpy as np
import faiss
import torch
import clip
import requests
from io import BytesIO
from PIL import Image as PILImage, UnidentifiedImageError
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "faiss-cpu", "torch", "numpy", "ftfy", "regex", "tqdm", "requests",
        "Pillow", "supabase", "python-dotenv"
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
)

app = App(
    name="build-multi-index-faiss",
    image=image,
    secrets=[Secret.from_name("supabase-creds")]
)

@app.function(gpu="A10G", timeout=3600)
def build_index_supabase():
    load_dotenv()
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ Using device: {device}")

    print("📦 Fetching product images from Supabase...")
    resp = supabase.table("product_images") \
        .select("id, image_url") \
        .or_("and(image_url.ilike.%ikon%,image_url.ilike.%.jpg),"
             "and(image_url.ilike.%ikon%,image_url.ilike.%.jpeg),"
             "and(image_url.ilike.%ikon%,image_url.ilike.%.png),"
             "and(image_url.ilike.%volt%,image_url.ilike.%.jpg),"
             "and(image_url.ilike.%volt%,image_url.ilike.%.jpeg),"
             "and(image_url.ilike.%volt%,image_url.ilike.%.png)") \
        .limit(400).execute()

    all_images = resp.data
    print(f"✅ Retrieved {len(all_images)} images")

    def is_valid_image(img):
        return img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))

    valid_images = [img for img in all_images if is_valid_image(img)]
    print(f"🔄 {len(valid_images)} valid images to embed.")

    color_vecs, structure_vecs, combined_vecs = [], [], []
    id_map = []

    for img in tqdm(valid_images):
        try:
            r = requests.get(img["image_url"], timeout=10)
            if not r.headers.get("Content-Type", "").startswith("image/"):
                continue

            pil_color = PILImage.open(BytesIO(r.content)).convert("RGB")
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
            id_map.append({
                "id": img["id"],
                "image_url": img["image_url"]
            })

        except (UnidentifiedImageError, Exception) as e:
            print(f"❌ Failed on {img['image_url']}: {e}")
            continue

    os.makedirs("/tmp/faiss", exist_ok=True)

    def save_index(name, vectors, id_entries):
        if not vectors:
            print(f"⚠️ No vectors to save for {name}. Skipping.")
            return

        np_vectors = np.stack(vectors).astype(np.float32)
        faiss.normalize_L2(np_vectors)
        index = faiss.IndexFlatIP(np_vectors.shape[1])
        index.add(np_vectors)

        index_path = f"/tmp/faiss/clip_{name}.index"
        id_map_path = f"/tmp/faiss/id_map_{name}.json"
        readable_path = f"/tmp/faiss/readable_{name}_id_map.json"

        faiss.write_index(index, index_path)

        with open(readable_path, "w") as f:
            json.dump([{**entry, "index": i} for i, entry in enumerate(id_entries)], f, indent=2)

        with open(id_map_path, "w") as f:
            json.dump([entry["id"] for entry in id_entries], f)

        for file_path in [index_path, id_map_path, readable_path]:
            with open(file_path, "rb") as f:
                file_name = os.path.basename(file_path)
                supabase.storage.from_("faiss").upload(
                    file=f,
                    path=file_name,
                    file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
                )

        print(f"✅ Uploaded {name} index with {len(vectors)} vectors.")

    save_index("color", color_vecs, id_map)
    save_index("structure", structure_vecs, id_map)
    save_index("combined", combined_vecs, id_map)

    print("🎉 All FAISS indexes built cleanly and uploaded.")
