import os
import json
import tempfile
import time
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

from faiss_sharding import (
    ShardState,
    maybe_rotate_and_upload_shard,
    flush_open_shard,
)

# ---------------------------
# Modal image with dependencies (keeps old production setup)
# ---------------------------
image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "faiss-cpu==1.12.0",
        "torch==2.8.0",
        "numpy",
        "ftfy",
        "regex",
        "tqdm",
        "requests",
        "Pillow",
        "supabase",
        "python-dotenv",
        "git+https://github.com/openai/CLIP.git@main",  # ← use CLIP from GitHub, no torch pin conflict
    )
)

app = App(name="build-color-index-faiss", image=image, secrets=[Secret.from_name("supabase-creds")])

# ---------------------------
# Config
# ---------------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
INDEX_TYPES = ["color"]  # scalable to ["color", "structure", "combined"]
DIM_BY_TYPE = {"color": 512}
LOCAL_FAISS_DIR = "/tmp/faiss"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROGRESS_FILE = os.path.join(LOCAL_FAISS_DIR, "progress.json")

# ship helper into container if needed
image = image.add_local_file("faiss_sharding.py", remote_path="/root/faiss_sharding.py")


@app.function(
    image=image,
    timeout=3600,
    gpu="A10G",  # keep GPU for prod speed
)
def build_index_supabase():
    import torch
    import clip
    from dotenv import load_dotenv

    load_dotenv()

    MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))  # 0 = no limit
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"✅ Using device: {device}")

    os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

    # ---------- resiliency helpers (from old) ----------
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
            file_options={"contentType": "application/json", "upsert": "true"},
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

    def is_valid_image_url(url: str):
        return url and url.lower().endswith(VALID_EXTENSIONS)

    def fetch_image(url):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image/"):
                return None
            return PILImage.open(BytesIO(r.content)).convert("RGB")
        except (requests.RequestException, UnidentifiedImageError, OSError):
            return None

    def extract_brand_from_url(url: str):
        path = urlparse(url).path.strip("/")
        if "/" in path:
            return path.split("/")[0].lower()
        return "unknown"

    # ---------- progress (prefer local, then remote) ----------
    local_progress = None
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            local_progress = json.load(f)
    remote_progress = download_json("faiss", "progress.json")
    progress = local_progress or remote_progress or {t: None for t in INDEX_TYPES}

    # ---------- id_map & index (merge local + remote) ----------
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
        merged_map = list(remote_map) + [x for x in local_map if x not in remote_map]
        id_map_by_type[t] = merged_map
        indexed_ids_by_type[t] = set(merged_map)

        local_index_path = os.path.join(LOCAL_FAISS_DIR, f"clip_{t}.index")
        remote_index = download_faiss_index("faiss", f"clip_{t}.index")
        if os.path.exists(local_index_path):
            local_index = faiss.read_index(local_index_path)
            if remote_index:
                remote_index.merge_from(local_index)
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(DIM_BY_TYPE[t])
        else:
            faiss_indexes[t] = remote_index or faiss.IndexFlatIP(DIM_BY_TYPE[t])

    shard_state = ShardState(DIM_BY_TYPE)

    # ---------- fetch candidate images (append-only + progress gate) ----------
    print("Fetching image IDs and URLs from Supabase…")
    try:
        query = supabase.table("product_images").select("id,image_url").order("id")
        if MAX_IMAGES:
            query = query.limit(MAX_IMAGES)
        ids_resp = query.execute()
        rows = ids_resp.data or []
    except Exception as e:
        print(f"Error fetching IDs: {e}")
        return

    # Filter: not yet indexed + beyond last progress
    last_done = progress.get("color")
    ids_to_process = [r for r in rows if r["id"] not in indexed_ids_by_type["color"]]
    if last_done:
        ids_to_process = [r for r in ids_to_process if r["id"] > last_done]

    print(f"Processing {len(ids_to_process)} new images…")

    # ---------- checkpointing ----------
    new_ids_by_type = {t: [] for t in INDEX_TYPES}

    def checkpoint_batch():
        # extend id_map with any new ids (per type), write both id_map and faiss index locally
        for t in INDEX_TYPES:
            if new_ids_by_type[t]:
                with open(os.path.join(LOCAL_FAISS_DIR, f"id_map_{t}.json"), "w") as f:
                    json.dump(id_map_by_type[t], f)
                faiss.write_index(faiss_indexes[t], os.path.join(LOCAL_FAISS_DIR, f"clip_{t}.index"))

                # --- NEW: sanity check to keep id_map and index in sync
                try:
                    assert faiss_indexes[t].ntotal == len(id_map_by_type[t]), (
                        f"Mismatch for {t}: index has {faiss_indexes[t].ntotal}, id_map has {len(id_map_by_type[t])}"
                    )
                except AssertionError as ae:
                    print(f"⚠️ {ae}")
                new_ids_by_type[t] = []

        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)

        # upload artifacts (with retries) to cloud
        with_retries(upload_json, "faiss", "progress.json", progress)
        for t in INDEX_TYPES:
            with_retries(upload_json, "faiss", f"id_map_{t}.json", id_map_by_type[t])

    # ---------- main embedding loop ----------
    brands_added = defaultdict(int)

    for start in tqdm(range(0, len(ids_to_process), BATCH_SIZE), desc="Embedding Batches"):
        batch_rows = ids_to_process[start:start + BATCH_SIZE]

        skipped_count = 0
        embedded_count = 0
        per_type_added = {t: 0 for t in INDEX_TYPES}

        # prepare a map id->url (we already have both from rows)
        for row in batch_rows:
            img_id = row["id"]
            url = row["image_url"]

            if not is_valid_image_url(url) or img_id in indexed_ids_by_type["color"]:
                skipped_count += 1
                continue

            pil_img = fetch_image(url)
            if pil_img is None:
                skipped_count += 1
                continue

            # --- NEW: safer normalization with clamp to avoid divide-by-zero
            with torch.no_grad():
                feat = model.encode_image(preprocess(pil_img).unsqueeze(0).to(device))
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            vec = feat.cpu().numpy().astype(np.float32)

            # ✅ Ensure unit normalization before adding to IP index (cosine)
            faiss.normalize_L2(vec)

            # 🔍 Sanity check: print norm of the vector (should be ~1.0)
            norm = np.linalg.norm(vec)
            if start == 0 and embedded_count < 3:  # only print a couple per first batch to avoid spam
                print(f"Sanity check norm for image {img_id}: {norm:.4f}")

            # add to global in-memory index + open shard
            faiss_indexes["color"].add(vec)
            shard_state.ensure_open("color")
            shard_state.current_ix["color"].add(vec)

            # 🔑 Append id immediately when vector goes into shard
            shard_state.current_ids["color"].append(img_id)

            maybe_rotate_and_upload_shard(supabase, "color", shard_state, id_map_by_type)  # ✅ add id_map_by_type

            indexed_ids_by_type["color"].add(img_id)
            new_ids_by_type["color"].append(img_id)
            per_type_added["color"] += 1
            progress["color"] = img_id

            brands_added[extract_brand_from_url(url)] += 1
            embedded_count += 1

        # per-batch log (includes per-type numbers)
        tqdm.write(
            f"Batch {start // BATCH_SIZE + 1}: Total={len(batch_rows)}, Skipped={skipped_count}, "
            f"Embedded={embedded_count}, color:{per_type_added['color']}"
        )

        flush_open_shard(supabase, "color", shard_state, id_map_by_type)  # ✅ add id_map_by_type
        checkpoint_batch()

    print("☁️ Flushing shards…")
    flush_open_shard(supabase, "color", shard_state, id_map_by_type)  # ✅ add id_map_by_type
    with_retries(upload_json, "faiss", f"id_map_color.json", id_map_by_type["color"])
    with_retries(upload_json, "faiss", "progress.json", progress)

    print("\n📊 Brands processed:")
    for brand, count in sorted(brands_added.items(), key=lambda x: x[1], reverse=True):
        print(f"{brand}: {count} new vectors")

    print("\n🎉 Color index build complete with sharding, checkpointing, and new safety checks.")
    
    # ✅ Ensure id_map is written in strict shard order
    id_map_path = os.path.join("/root", "id_map_color.json")
    with open(id_map_path, "w") as f:
        json.dump(id_map_by_type["color"], f)

    with_retries(upload_json, "faiss", "id_map_color.json", id_map_by_type["color"])
    print(f"Final id_map_color.json uploaded with {len(id_map_by_type['color'])} entries")

