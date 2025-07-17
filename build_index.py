from modal import App, Image, Secret
import os, json, tempfile, argparse
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

# Modal app definition
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
    name="build-faiss-to-supabase",
    image=image,
    secrets=[Secret.from_name("supabase-creds")]
)

@app.function(gpu="A10G", timeout=3600)
def build_index_supabase(batch_size: int = 500):
    load_dotenv()

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("📦 Fetching product images from Supabase...")
    all_images = []
    page = 0
    #while True:
    #    resp = supabase.table("product_images").select("id, image_url").range(page * 1000, (page + 1) * 1000 - 1).execute()
    #### Remove below this if it does not work!
    resp = supabase.table("product_images") \
        .select("id, image_url") \
        .or_("image_url.ilike.%ikon%,image_url.ilike.%volt%") \
        .limit(400) \
        .execute()

    all_images = resp.data

    #### Change above this if it does not work!
    #batch = resp.data
    #if not batch:
    #    break
    #all_images.extend(batch)
    #page += 1

    print(f"✅ Retrieved {len(all_images)} images")

    def fetch_existing_ids(index_name):
        try:
            file_name = f"id_map_{index_name}.json"
            res = supabase.storage.from_("faiss").download(file_name)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(res)
                tmp_path = tmp.name
            with open(tmp_path, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()

    def load_existing_index(index_name):
        try:
            file_name = f"clip_{index_name}.index"
            res = supabase.storage.from_("faiss").download(file_name)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(res)
                tmp_path = tmp.name
            return faiss.read_index(tmp_path)
        except Exception:
            return None

    processed_ids = fetch_existing_ids("combined")
    valid_images = [img for img in all_images if img["image_url"].lower().endswith((".jpg", ".jpeg", ".png")) and img["id"] not in processed_ids]
    print(f"🔄 {len(valid_images)} images remaining to embed.")

    for i in range(0, len(valid_images), batch_size):
        batch_images = valid_images[i:i + batch_size]

        color_vecs, structure_vecs, combined_vecs = [], [], []
        id_map = []

        for img in tqdm(batch_images):
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
                id_map.append(img["id"])

            except (UnidentifiedImageError, Exception) as e:
                print(f"❌ Failed on {img['image_url']}: {e}")
                continue

        os.makedirs("/tmp/faiss", exist_ok=True)

        def save_and_upload(name, new_vectors):
            if not new_vectors:
                print(f"⚠️ Skipping {name} — no vectors.")
                return

            index_path = f"/tmp/faiss/clip_{name}.index"
            idmap_path = f"/tmp/faiss/id_map_{name}.json"

            new_arr = np.stack(new_vectors).astype(np.float32)
            #faiss.normalize_L2(new_arr)

            existing_index = load_existing_index(name)
            if existing_index:
                existing_index.add(new_arr)
                idx = existing_index
            else:
                idx = faiss.IndexFlatIP(new_arr.shape[1])
                idx.add(new_arr)

            faiss.write_index(idx, index_path)
            updated_ids = processed_ids.union(set(id_map))
            with open(idmap_path, "w") as f:
                json.dump(list(updated_ids), f)

            print(f"⬆️ Uploading {name} index to Supabase Storage...")

            try:
                for file_path in [index_path, idmap_path]:
                    with open(file_path, "rb") as f:
                        file_name = os.path.basename(file_path)
                        supabase.storage.from_("faiss").upload(
                            file=f,
                            path=file_name,
                            file_options={
                                "content-type": "application/octet-stream",
                                "x-upsert": "true"
                            }
                        )
                print(f"✅ {name} index and ID map uploaded.")
            except Exception as e:
                print(f"❌ Upload failed for {name}: {e}")
                raise
        # 💾 Save local .npy files for debugging
        np.save("clip_color_embeddings.npy", np.stack(color_vecs))
        np.save("clip_structure_embeddings.npy", np.stack(structure_vecs))
        np.save("clip_combined_embeddings.npy", np.stack(combined_vecs))


        save_and_upload("color", color_vecs)
        save_and_upload("structure", structure_vecs)
        save_and_upload("combined", combined_vecs)

        processed_ids.update(id_map)  # Update in-memory checkpoint

    print("🎉 All FAISS indexes built and uploaded to Supabase.")
