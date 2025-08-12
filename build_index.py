import os
import json
import tempfile
from collections import defaultdict
from urllib.parse import urlparse
from io import BytesIO

import modal
import requests
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ---------------------------
# Modal image with dependencies
# ---------------------------
image = (
    modal.Image.debian_slim()
    .pip_install(
        "faiss-cpu",
        "pillow",
        "requests",
        "tqdm",
        "supabase",
        "sentence-transformers"
    )
)

stub = modal.Stub("faiss-index-builder", image=image)

# ---------------------------
# Config
# ---------------------------
BATCH_SIZE = 200
INDEX_TYPES = ["color", "structure", "combined"]
LOCAL_FAISS_DIR = "/tmp/faiss"  # Persistent for the life of container
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROGRESS_FILE = os.path.join(LOCAL_FAISS_DIR, "progress.json")

# ---------------------------
# Main function
# ---------------------------
@stub.function(
    image=image,
    timeout=3600,
    gpu="A10G",
    mounts=[
        modal.Mount.from_local_dir(".", remote_path="/root/app")  # Adjust path if needed
    ]
)
def build_index_supabase():
    # Supabase config
    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Model
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Ensure local dir
    os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

    # ---------------------------
    # Helpers
    # ---------------------------
    def download_json(bucket, path):
        try:
            res = supabase.storage.from_(bucket).download(path)
            if res:
                return json.loads(res.decode("utf-8"))
        except Exception:
            pass
        return None

    def upload_json(bucket, path, data):
        supabase.storage.from_(bucket).upload(path, json.dumps(data).encode("utf-8"), {"upsert": "true"})

    def download_faiss_index(bucket, path):
        try:
            res = supabase.storage.from_(bucket).download(path)
            if res:
                tmp_file = tempfile.NamedTemporaryFile(delete=False)
                tmp_file.write(res)
                tmp_file.flush()
                index = faiss.read_index(tmp_file.name)
                tmp_file.close()
                return index
        except Exception:
            pass
        return None

    def upload_faiss_index(bucket, path, index):
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        faiss.write_index(index, tmp_file.name)
        tmp_file.close()
        with open(tmp_file.name, "rb") as f:
            supabase.storage.from_(bucket).upload(path, f.read(), {"upsert": "true"})

    def is_valid_image_url(url: str):
        return url.lower().endswith(VALID_EXTENSIONS)

    def fetch_image(url):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image/"):
                return None
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        except (requests.RequestException, UnidentifiedImageError, OSError):
            return None

    def extract_brand_from_url(url: str):
        """Extracts the brand/folder name from an image URL."""
        path = urlparse(url).path.strip("/")
        if "/" in path:
            return path.split("/")[0].lower()
        return "unknown"

    # ---------------------------
    # Load progress.json
    # ---------------------------
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
    else:
        progress = {t: None for t in INDEX_TYPES}

    # ---------------------------
    # Load id_maps & indexes
    # ---------------------------
    id_map_by_type = {}
    indexed_ids_by_type = {}
    faiss_indexes = {}

    for t in INDEX_TYPES:
        local_map_path = os.path.join(LOCAL_FAISS_DIR, f"id_map_{t}.json")
        remote_map = download_json("faiss", f"id_map_{t}.json") or []
        local_map = []
        if os.path.exists(local_map_path):
            with open(local_map_path, "r") as f:
                local_map = json.load(f)
        merged_map = list(set(remote_map) | set(local_map))
        id_map_by_type[t] = merged_map
        indexed_ids_by_type[t] = set(merged_map)

        local_index_path = os.path.join(LOCAL_FAISS_DIR, f"clip_{t}.index")
        remote_index = download_faiss_index("faiss", f"clip_{t}.index")
        if os.path.exists(local_index_path):
            local_index = faiss.read_index(local_index_path)
            if remote_index:
                remote_index.merge_from(local_index)
            faiss_indexes[t] = remote_index
        else:
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(512)

    # ---------------------------
    # Get all images from DB
    # ---------------------------
    rows = supabase.table("product_images").select("id,image_url").execute().data
    print(f"📊 Total images in DB: {len(rows)}")

    # ---------------------------
    # Main embedding loop
    # ---------------------------
    new_ids_by_type = {t: [] for t in INDEX_TYPES}
    brands_added = defaultdict(int)  # For end-of-run report

    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding Batches"):
        batch = rows[start:start + BATCH_SIZE]
        skipped_count = 0
        embedded_count = 0
        per_type_added = {t: 0 for t in INDEX_TYPES}

        for img in batch:
            img_id = img["id"]
            url = img["image_url"]

            if not is_valid_image_url(url):
                skipped_count += 1
                continue

            missing_types = []
            for t in INDEX_TYPES:
                if img_id not in indexed_ids_by_type[t]:
                    missing_types.append(t)
                elif progress[t] and img_id <= progress[t]:
                    continue

            if not missing_types:
                skipped_count += 1
                continue

            image_obj = fetch_image(url)
            if image_obj is None:
                skipped_count += 1
                continue

            vector = clip_model.encode([image_obj], convert_to_numpy=True, normalize_embeddings=True)[0]
            brand_name = extract_brand_from_url(url)

            for t in missing_types:
                faiss_indexes[t].add(np.expand_dims(vector, axis=0))
                indexed_ids_by_type[t].add(img_id)
                new_ids_by_type[t].append(img_id)
                per_type_added[t] += 1
                progress[t] = img_id

            brands_added[brand_name] += 1
            embedded_count += 1

        tqdm.write(
            f"Batch {start // BATCH_SIZE + 1}: "
            f"Total={len(batch)}, Skipped={skipped_count}, Embedded={embedded_count}, "
            + ", ".join(f"{t}:{per_type_added[t]}" for t in INDEX_TYPES)
        )

        # Save local progress & indexes after each batch
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)
        for t in INDEX_TYPES:
            if new_ids_by_type[t]:
                id_map_by_type[t].extend(new_ids_by_type[t])
                with open(os.path.join(LOCAL_FAISS_DIR, f"id_map_{t}.json"), "w") as f:
                    json.dump(id_map_by_type[t], f)
                faiss.write_index(faiss_indexes[t], os.path.join(LOCAL_FAISS_DIR, f"clip_{t}.index"))
                new_ids_by_type[t] = []

    # ---------------------------
    # Final upload
    # ---------------------------
    print("☁️ Uploading all data to Supabase...")
    for t in INDEX_TYPES:
        upload_json("faiss", f"id_map_{t}.json", id_map_by_type[t])
        upload_faiss_index("faiss", f"clip_{t}.index", faiss_indexes[t])

    # ---------------------------
    # Brand coverage report
    # ---------------------------
    print("\n📊 Brands with new vectors this run:")
    for brand, count in sorted(brands_added.items(), key=lambda x: x[1], reverse=True):
        print(f"{brand}: {count} new vectors")

    print("\n🎉 All FAISS indexes built locally with bulletproof per-index resilience, resume capability, and uploaded at the end.")
