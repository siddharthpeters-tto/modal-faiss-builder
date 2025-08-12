import os
import json
import tempfile
from collections import defaultdict
from urllib.parse import urlparse
from io import BytesIO

from modal import App, Image, Secret
import requests
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage, UnidentifiedImageError
from supabase import create_client

# ---------------------------
# Modal image with dependencies (no new deps added)
# ---------------------------
image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "faiss-cpu",
        "torch",
        "numpy",
        "ftfy",
        "regex",
        "tqdm",
        "requests",
        "Pillow",
        "supabase",
        "python-dotenv"
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
)

app = App(name="build-multi-index-faiss", image=image, secrets=[Secret.from_name("supabase-creds")])

# ---------------------------
# Config
# ---------------------------
BATCH_SIZE = 200
INDEX_TYPES = ["color", "structure", "combined"]
LOCAL_FAISS_DIR = "/tmp/faiss"  # Persistent for the life of the container
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROGRESS_FILE = os.path.join(LOCAL_FAISS_DIR, "progress.json")

# ---------------------------
# Main function
# ---------------------------
@app.function(
    image=image,
    timeout=3600,
    gpu="A10G"
)
def build_index_supabase():
    import torch
    import clip  # loaded from git install above

    # Supabase config
    from dotenv import load_dotenv
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Device & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ Using device: {device}")

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
        supabase.storage.from_(bucket).upload(
            path,
            json.dumps(data).encode("utf-8"),
            {"x-upsert": "true", "content-type": "application/json"}
        )

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
            supabase.storage.from_(bucket).upload(
                path, f.read(), {"x-upsert": "true", "content-type": "application/octet-stream"}
            )

    def is_valid_image_url(url: str):
        return url.lower().endswith(VALID_EXTENSIONS)

    def fetch_image(url):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image/"):
                return None
            img = PILImage.open(BytesIO(r.content)).convert("RGB")
            return img
        except (requests.RequestException, UnidentifiedImageError, OSError):
            return None

    def extract_brand_from_url(url: str):
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
            # Dimension is 512 for ViT-B/32 image embeddings
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(512)

    # ---------------------------
    # Get all images from DB

    # --- Get all image IDs and total count ---
    print("Fetching all image IDs from Supabase to establish processing order...")
    all_image_ids = []
    current_offset = 0
    id_fetch_limit = 1000  # Supabase default limit for select queries

    try:
        while True:
            # Fetch IDs in batches
            ids_resp = supabase.table("product_images") \
                .select("id") \
                .order("id") \
                .range(current_offset, current_offset + id_fetch_limit - 1) \
                .execute()

            if not ids_resp.data:
                break  # No more IDs to fetch

            all_image_ids.extend([item['id'] for item in ids_resp.data])
            current_offset += id_fetch_limit
            print(f"Fetched {len(all_image_ids)} IDs so far...")

        total_images = len(all_image_ids)
        print(f"There are a total of {total_images} images in your Supabase table.")
    except Exception as e:
        print(f"Could not retrieve all image IDs or total count: {e}. Cannot proceed without a definitive list of IDs.")
        return  # Exit if we can't get the full list of IDs

    total_processed_count = 0

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

            # Only process if missing in at least one index
            missing_types = [t for t in INDEX_TYPES if img_id not in indexed_ids_by_type[t]]
            if not missing_types:
                skipped_count += 1
                continue

            pil_color = fetch_image(url)
            if pil_color is None:
                skipped_count += 1
                continue

            # Prepare tensors
            color_tensor = preprocess(pil_color).unsqueeze(0).to(device)

            # Structure (grayscale fed through same preprocess)
            pil_grey = pil_color.convert("L").convert("RGB")
            grey_tensor = preprocess(pil_grey).unsqueeze(0).to(device)

            with torch.no_grad():
                color_feat = model.encode_image(color_tensor)
                color_feat /= color_feat.norm(dim=-1, keepdim=True)

                grey_feat = model.encode_image(grey_tensor)
                grey_feat /= grey_feat.norm(dim=-1, keepdim=True)

                combined_feat = (color_feat + grey_feat) / 2
                combined_feat /= combined_feat.norm(dim=-1, keepdim=True)

            # Convert to numpy float32
            color_vec = color_feat.cpu().numpy().astype(np.float32)
            grey_vec = grey_feat.cpu().numpy().astype(np.float32)
            combined_vec = combined_feat.cpu().numpy().astype(np.float32)

            # Add to relevant indexes only
            for t in missing_types:
                if t == "color":
                    faiss_indexes[t].add(color_vec)
                elif t == "structure":
                    faiss_indexes[t].add(grey_vec)
                else:  # combined
                    faiss_indexes[t].add(combined_vec)

                indexed_ids_by_type[t].add(img_id)
                new_ids_by_type[t].append(img_id)
                per_type_added[t] += 1
                progress[t] = img_id  # last processed id per index type

            brand_name = extract_brand_from_url(url)
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
    # Final upload (single commit to cloud)
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
