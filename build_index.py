from modal import App, Image, Secret, Volume
import os, json
import numpy as np
import faiss
import torch
import clip
import requests
from io import BytesIO
from tqdm import tqdm
from PIL import Image as PILImage, UnidentifiedImageError
from supabase import create_client
from dotenv import load_dotenv
import boto3
from botocore.client import Config

# Modal setup
volume = Volume.from_name("faiss-index-storage", create_if_missing=True)

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "faiss-cpu", "torch", "numpy", "ftfy", "regex", "tqdm", "requests",
        "Pillow", "supabase", "python-dotenv", "boto3"
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
    load_dotenv()

    # R2 setup
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

    r2 = boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto',
    )

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

    progress_path = "/data/progress.json"
    seen_ids = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            seen_ids = set(json.load(f))

    color_vecs, structure_vecs, combined_vecs = [], [], []
    id_map = []

    for img in tqdm(valid_images):
        if img["id"] in seen_ids:
            continue

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
            seen_ids.add(img["id"])

            if len(seen_ids) % 100 == 0:
                with open(progress_path, "w") as f:
                    json.dump(list(seen_ids), f)

        except (UnidentifiedImageError, Exception) as e:
            print(f"❌ Failed on {img['image_url']}: {e}")
            continue

    def save_and_upload(name, vectors):
        if not vectors:
            print(f"⚠️ Skipping {name} — no vectors.")
            return

        arr = np.stack(vectors).astype(np.float32)
        faiss.normalize_L2(arr)
        idx = faiss.IndexFlatIP(arr.shape[1])
        idx.add(arr)

        index_path = f"/data/clip_{name}.index"
        idmap_path = f"/data/id_map_{name}.json"

        faiss.write_index(idx, index_path)
        with open(idmap_path, "w") as f:
            json.dump(id_map, f)

        print(f"⬆️ Uploading {name} index to R2...")
        r2.upload_file(index_path, R2_BUCKET_NAME, f"faiss/clip_{name}.index")
        r2.upload_file(idmap_path, R2_BUCKET_NAME, f"faiss/id_map_{name}.json")
        print(f"✅ {name} index uploaded.")

    save_and_upload("color", color_vecs)
    save_and_upload("structure", structure_vecs)
    save_and_upload("combined", combined_vecs)

    with open(progress_path, "w") as f:
        json.dump(list(seen_ids), f)

    print("🎉 All indexes built and uploaded successfully.")

if __name__ == "__main__":
    with app.run():
        build_all_indexes.remote()
