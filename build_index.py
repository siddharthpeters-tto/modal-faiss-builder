import os
import json
import tempfile
import time
from collections import defaultdict
from urllib.parse import urlparse
from io import BytesIO

from modal import App, Image, Secret, Mount

import requests
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage, UnidentifiedImageError
from supabase import create_client

# NEW: sharding helpers
from faiss_sharding import (
    ShardState,
    maybe_rotate_and_upload_shard,
    flush_open_shard,
)

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
BATCH_SIZE = 1000  # increased for fewer round-trips; checkpointed each batch
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
    gpu="A10G",
    mounts=[Mount.from_local_file("faiss_sharding.py", remote_path="/root/faiss_sharding.py")]
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
    model.eval()
    print(f"✅ Using device: {device}")

    # Ensure local dir
    os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

    # ---------------------------
    # Helpers
    # ---------------------------
    def with_retries(fn, *args, **kwargs):
        retries = kwargs.pop("retries", 5)
        delay = kwargs.pop("delay", 2.0)
        last_err = None
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                time.sleep(delay * (2 ** i))
        raise last_err

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
            path=path,
            file=json.dumps(data).encode("utf-8"),
            file_options={"contentType": "application/json", "upsert": True},
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
        tmp = tempfile.NamedTemporaryFile(delete=False)
        faiss.write_index(index, tmp.name); tmp.close()
        with open(tmp.name, "rb") as f:
            supabase.storage.from_(bucket).upload(
                path=path,
                file=f.read(),
                file_options={"contentType": "application/octet-stream", "upsert": True},
            )

    def is_valid_image_url(url: str):
        return url.lower().endswith(VALID_EXTENSIONS)

    def fetch_image(url):
        try:
            r = requests.get(url, timeout=20)
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
    # Load progress.json (prefer local within same container, else remote)
    # ---------------------------
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            local_progress = json.load(f)
    else:
        local_progress = None
    remote_progress = download_json("faiss", "progress.json")

    if local_progress is not None:
        progress = local_progress
    elif remote_progress is not None:
        progress = remote_progress
    else:
        progress = {t: None for t in INDEX_TYPES}

    # ---------------------------
    # Load id_maps & indexes
    # ---------------------------
    id_map_by_type = {}
    indexed_ids_by_type = {}
    faiss_indexes = {}

    # Embedding dim for ViT-B/32 image features is 512 across all three variants
    DIM_BY_TYPE = {"color": 512, "structure": 512, "combined": 512}

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
            # If still None, create a fresh IP index
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(DIM_BY_TYPE[t])
        else:
            # Dimension is 512 for ViT-B/32 image embeddings
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(DIM_BY_TYPE[t])

    # NEW: initialize sharding state (separate from monolithic in-RAM indexes)
    shard_state = ShardState(DIM_BY_TYPE)

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
    # Batch checkpoint helper (commit to Supabase after each batch)
    # ---------------------------
    def checkpoint_batch(id_map_by_type, new_ids_by_type, faiss_indexes, progress):
        # 1) Write local progress and any touched index files
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)

        for t in INDEX_TYPES:
            if new_ids_by_type[t]:
                id_map_by_type[t].extend(new_ids_by_type[t])
                with open(os.path.join(LOCAL_FAISS_DIR, f"id_map_{t}.json"), "w") as f:
                    json.dump(id_map_by_type[t], f)
                # Keep writing monolithic local file (resume safety), but DO NOT upload it to Supabase
                faiss.write_index(faiss_indexes[t], os.path.join(LOCAL_FAISS_DIR, f"clip_{t}.index"))
                new_ids_by_type[t] = []

        # 2) Upload to Supabase with retries (JSONs only; shards are uploaded continuously by shard rotation)
        with_retries(upload_json, "faiss", "progress.json", progress)
        for t in INDEX_TYPES:
            local_map_path = os.path.join(LOCAL_FAISS_DIR, f"id_map_{t}.json")
            if os.path.exists(local_map_path):
                with open(local_map_path, "r") as f:
                    current_map = json.load(f)
                with_retries(upload_json, "faiss", f"id_map_{t}.json", current_map)
        # NOTE: intentionally NOT uploading clip_{t}.index to avoid 413

    # ---------------------------
    # Main embedding loop
    # ---------------------------
    new_ids_by_type = {t: [] for t in INDEX_TYPES}
    brands_added = defaultdict(int)  # For end-of-run report

    for start in tqdm(range(0, len(all_image_ids), BATCH_SIZE), desc="Embedding Batches"):
        batch_ids = all_image_ids[start:start + BATCH_SIZE]
        # Fetch actual image data for this batch
        resp = supabase.table("product_images") \
            .select("id,image_url") \
            .in_("id", batch_ids) \
            .execute()
        batch = resp.data
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

                # NEW: also add to the current open shard and rotate/upload if needed
                shard_state.ensure_open(t)
                if t == "color":
                    shard_state.current_ix[t].add(color_vec)
                elif t == "structure":
                    shard_state.current_ix[t].add(grey_vec)
                else:
                    shard_state.current_ix[t].add(combined_vec)
                maybe_rotate_and_upload_shard(supabase, t, shard_state)

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

        for t in INDEX_TYPES:
            flush_open_shard(supabase, t, shard_state)

        # ✅ Robust checkpoint: save + upload after each batch (JSON and shards only)
        checkpoint_batch(id_map_by_type, new_ids_by_type, faiss_indexes, progress)

    # ---------------------------
    # Finalize: flush any open shard(s) and push final JSONs
    # ---------------------------
    print("☁️ Flushing any open shards and finalizing JSONs to Supabase...")
    for t in INDEX_TYPES:
        flush_open_shard(supabase, t, shard_state)
        with_retries(upload_json, "faiss", f"id_map_{t}.json", id_map_by_type[t])
    with_retries(upload_json, "faiss", "progress.json", progress)

    # ---------------------------
    # Brand coverage report
    # ---------------------------
    print("\n📊 Brands with new vectors this run:")
    for brand, count in sorted(brands_added.items(), key=lambda x: x[1], reverse=True):
        print(f"{brand}: {count} new vectors")

    print("\n🎉 All FAISS indexes built with per-batch remote checkpoints, sharded uploads, and full resume capability.")
