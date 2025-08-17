import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import io
import re
import json
import math
import time
import random
import tempfile
import bisect
from typing import List, Tuple

import numpy as np
from collections import defaultdict
import requests
from PIL import Image

# --- Supabase ---
from supabase import create_client, Client

# --- FAISS / CLIP ---
import faiss  # type: ignore
import torch
import clip  # openai/CLIP

"""
Alignment checker for sharded FAISS indexes stored in Supabase Storage.

What it does
------------
1) Lists shard *.index files in a bucket/prefix on Supabase.
2) Downloads shards to a cross‑platform temp folder.
3) Loads a single global id_map (positions -> image_id) from JSON or NPY.
4) Builds a combined FAISS index via IndexShards(successive_ids=True).
5) Samples image_ids (optionally restricted to a brand) and:
   - fetches the image_url from Supabase
   - encodes it with CLIP
   - searches the combined index (top‑1)
   - compares the *expected* shard (by id_map position) vs the *found* shard.

Notes
-----
• This script assumes your id_map is the concatenation order used when you
  built shards (i.e., shard_00000 vectors first, then 00001, ...).
• Set env vars below; reasonable defaults provided.
"""
from dotenv import load_dotenv
load_dotenv()

# =============================
# Config via environment vars
# =============================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "faiss")              # e.g. "faiss"
# Put empty default prefix (root) since your shards + id_map are at bucket root
PREFIX = os.getenv("FAISS_PREFIX", "")
INDEX_REGEX = os.getenv("INDEX_REGEX", r"clip_color_shard_\d{5}\.index")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "id_map_color.json")
SAMPLES = int(os.getenv("SAMPLES", "20"))
TEST_LIMIT = int(os.getenv("TEST_LIMIT", "0"))  # 0 = no limit
TEST_BRAND = os.getenv("TEST_BRAND", "").strip()  # optional brand name filter
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))

# Tables / columns (adjust to your schema)
IMAGES_TABLE = os.getenv("IMAGES_TABLE", "product_images")
IMG_ID_COL = os.getenv("IMG_ID_COL", "id")
IMG_URL_COL = os.getenv("IMG_URL_COL", "image_url")

# Metric assumed by your FAISS index ("ip" or "l2"). If you built cosine, it's typically IP with L2‑normalized vectors
METRIC = os.getenv("FAISS_METRIC", "ip").lower()

# =====================================
# Helpers: Supabase client & filesystem
# =====================================

def supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY env vars.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

sb: Client = supabase_client()

TMP_DIR = os.path.join(tempfile.gettempdir(), "faiss_shards")
os.makedirs(TMP_DIR, exist_ok=True)


def list_bucket_files(prefix: str) -> List[str]:
    """Recursively list files under a prefix (folder) in Supabase Storage.
    IMPORTANT: 'prefix' is the path *inside* the bucket, e.g. 'clip_color' or 'clip/combined'.
    Do NOT include the bucket name here.
    """
    files: List[str] = []

    def _walk(p: str):
        entries = sb.storage.from_(BUCKET_NAME).list(path=p)
        for it in entries:
            name = it.get("name")
            if not name:
                continue
            full = f"{p}/{name}" if p else name
            # Heuristic: folders in Supabase list have no mimetype and size 0
            is_folder = it.get("metadata", {}).get("mimetype") in (None, "") and not name.lower().endswith(('.index', '.json', '.npy', '.faiss'))
            if is_folder:
                _walk(full)
            else:
                files.append(full)

    _walk(prefix.rstrip('/'))
    return files


def download_file(path: str) -> str:
    data = sb.storage.from_(BUCKET_NAME).download(path)
    if data is None:
        raise RuntimeError(f"Failed to download: {path}")
    local_path = os.path.join(TMP_DIR, os.path.basename(path))
    with open(local_path, "wb") as f:
        f.write(data if isinstance(data, (bytes, bytearray)) else bytes(data))
    print(f"⬇️  Saved {path} → {local_path}")
    return local_path


# --- Add near the top, after env loads ---
print("🔧 Debug config:",
      f"\n  BUCKET_NAME = {BUCKET_NAME}",
      f"\n  PREFIX      = '{PREFIX}'",
      f"\n  INDEX_REGEX = {INDEX_REGEX}",
      f"\n  ID_MAP_PATH = {ID_MAP_PATH}",
      f"\n  IMAGES_TABLE= {IMAGES_TABLE}",
      sep="")

# --- Replace list_bucket_files with a more verbose version ---
def list_bucket_files(prefix: str) -> List[str]:
    """Recursively list files under a prefix in Supabase Storage."""
    files: List[str] = []

    def _walk(p: str):
        entries = sb.storage.from_(BUCKET_NAME).list(path=p)
        if entries is None:
            print(f"⚠️  list() returned None for path='{p}'")
            return
        if not entries:
            print(f"ℹ️  No entries under '{p}'")
        for it in entries:
            name = it.get("name")
            if not name:
                continue
            full = f"{p}/{name}" if p else name
            # Folder heuristic: Supabase marks folders with no mimetype
            mimetype = (it.get("metadata") or {}).get("mimetype")
            is_folder = (mimetype in (None, "")) and not name.lower().endswith(('.index', '.json', '.npy', '.faiss'))
            if is_folder:
                _walk(full)
            else:
                files.append(full)

    path = prefix.rstrip('/')
    print(f"🔎 Walking prefix '{path or '<root>'}'...")
    _walk(path)
    return files

# --- Replace load_shards() with a version that falls back to the root when empty ---
def load_shards() -> Tuple[faiss.Index, List[int]]:
    def _show(head):
        for f in head[:50]:
            print(" •", f)
        if len(head) > 50:
            print(f" … (+{len(head)-50} more)")

    files = list_bucket_files(PREFIX)
    if not files:
        print(f"❗ No files under prefix '{PREFIX}'. Trying bucket root as a fallback…")
        files = list_bucket_files("")  # fallback to root
        if not files:
            raise RuntimeError(
                f"No files found in bucket '{BUCKET_NAME}' even at root. "
                f"Check your bucket name / permissions."
            )
        print("📁 Files at bucket root:")
        _show(files)
    else:
        print(f"📁 Files under prefix '{PREFIX or '<root>'}':")
        _show(files)

    pat = re.compile(INDEX_REGEX)
    shard_paths = sorted([p for p in files if pat.search(os.path.basename(p))])

    if not shard_paths:
        basenames = [os.path.basename(x) for x in files]
        raise RuntimeError(
            "No shard indexes matched your INDEX_REGEX.\n"
            f"INDEX_REGEX: {INDEX_REGEX}\n"
            f"Basenames seen (first 30): {basenames[:30]}"
        )

    print(f"📦 Found {len(shard_paths)} shard(s) matching {INDEX_REGEX}:")
    for s in shard_paths:
        print("   -", s)

    shard_local = [download_file(p) for p in shard_paths]
    shard_indexes = [faiss.read_index(p) for p in shard_local]

    d = shard_indexes[0].d
    index = faiss.IndexShards(d, True, True)
    offsets = [0]
    total = 0
    for sh in shard_indexes:
        index.add_shard(sh)
        total += sh.ntotal
        offsets.append(total)
    print(f"🧠 Combined index built. Total vectors: {total}")
    return index, offsets

# --- Make load_id_map() resilient to id_map_* names when ID_MAP_PATH is wrong ---
def load_id_map() -> List[str]:
    global ID_MAP_PATH
    if not ID_MAP_PATH:
        raise RuntimeError("ID_MAP_PATH not set.")

    try:
        local = download_file(ID_MAP_PATH)
    except Exception:
        print(f"❗ Couldn't download '{ID_MAP_PATH}'. Attempting auto-detect (id_map*.json|npy) at prefix '{PREFIX or '<root>'}'…")
        candidates = [p for p in list_bucket_files(PREFIX) if re.search(r"id_map.*\.(json|npy)$", os.path.basename(p), re.I)]
        if not candidates:
            # last-chance: root
            candidates = [p for p in list_bucket_files("") if re.search(r"id_map.*\.(json|npy)$", os.path.basename(p), re.I)]
        if not candidates:
            raise RuntimeError("No id_map file found matching 'id_map*.json|npy'.")
        # Prefer json, then npy
        candidates.sort(key=lambda x: (not x.lower().endswith(".json"), x))
        ID_MAP_PATH = candidates[0]
        print(f"✅ Using detected id_map: {ID_MAP_PATH}")
        local = download_file(ID_MAP_PATH)

    if local.lower().endswith('.json'):
        with open(local, 'r', encoding='utf-8') as f:
            m = json.load(f)
        if isinstance(m, dict) and 'id_map' in m:
            m = m['id_map']
        if not isinstance(m, list):
            raise RuntimeError("id_map.json must be a JSON list or {'id_map': [...]}.")
        return [str(x) for x in m]
    elif local.lower().endswith('.npy'):
        arr = np.load(local, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    else:
        raise RuntimeError("ID_MAP_PATH must be .json or .npy")


# ==================
# Brand filter (opt)
# ==================

def eligible_ids_by_brand(id_map: List[str]) -> List[str]:
    # Keep id_map intact; only narrow the *sampling set* if brand filter provided
    if not TEST_BRAND:
        return id_map

    # Join to find brand names; adjust to your schema
    # product_images -> product_variants -> products -> brands
    sel = (
        f"{IMG_ID_COL},{IMG_URL_COL},"
        "product_variants!inner(product_id,products!inner(brand_id,brands!inner(name)))"
    )
    resp = sb.table(IMAGES_TABLE).select(sel).in_(IMG_ID_COL, id_map).execute()
    rows = resp.data or []
    keep = []
    for r in rows:
        try:
            brand_name = r["product_variants"]["products"]["brands"]["name"]
            if str(brand_name).lower() == TEST_BRAND.lower():
                keep.append(r[IMG_ID_COL])
        except Exception:
            pass
    print(f"🔍 Filtered to brand '{TEST_BRAND}': {len(keep)} ids eligible.")
    return keep


# =======================
# Image download & encode
# =======================

def load_clip(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def encode_image(url: str, model, preprocess, device: str) -> np.ndarray:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vec = feat.cpu().numpy().astype('float32')
        faiss.normalize_L2(vec)  # ✅ ensure cosine-compatible
    return vec


# ==================
# Shard math helper
# ==================

def shard_for_pos(pos: int, offsets: List[int]) -> int:
    """Return 1‑based shard index for a global position using cumulative offsets."""
    return bisect.bisect_right(offsets, pos)


# =======
#  Main
# =======
if __name__ == "__main__":
    print("📦 Listing shards in Supabase...")
    index, shard_offsets = load_shards()
    print(f"🔍 FAISS metric_type = {index.metric_type} (0=L2, 1=IP)")

    id_map = load_id_map()
    print(f"🗺️  id_map loaded with {len(id_map)} entries.")

    # Build *eligible* sampling set
    eligible = eligible_ids_by_brand(id_map)
    if TEST_LIMIT and len(eligible) > TEST_LIMIT:
        eligible = eligible[:TEST_LIMIT]
        print(f"🧪 Test limited to first {TEST_LIMIT} eligible ids.")

    if not eligible:
        raise SystemExit("No eligible ids to test.")

    # Sample without disturbing id_map order
    sample_ids = random.sample(eligible, min(SAMPLES, len(eligible)))
    print(f"🎯 Sampling {len(sample_ids)} ids for alignment checks...")

    # CLIP
    model, preprocess, device = load_clip()
    print(f"🧩 CLIP ready on {device}.")

    mismatches = 0
    for i, qid in enumerate(sample_ids, 1):
        # Fetch image URL
        row = sb.table(IMAGES_TABLE).select(f"{IMG_URL_COL}").eq(IMG_ID_COL, qid).single().execute().data
        if not row or not row.get(IMG_URL_COL):
            print(f"[{i}/{len(sample_ids)}] ⚠️ Missing image_url for id {qid}")
            continue
        url = row[IMG_URL_COL]

        try:
            qvec = encode_image(url, model, preprocess, device)

            # Get top-5 neighbors instead of only top-1
            D, I = index.search(qvec, 5)

            # --- COSINE PATCH (simplified) ---
            metric_type = index.metric_type  # 0=L2, 1=IP
            topk_positions = [int(p) for p in I[0]]
            topk_sims = []
            for val in D[0]:
                cos = float(val)
                topk_sims.append(cos)

            pos_in_idmap = id_map.index(qid)
            expected_shard = shard_for_pos(pos_in_idmap, shard_offsets)

            ok = (pos_in_idmap in topk_positions)
            status = "✅" if ok else "❌"
            if not ok:
                mismatches += 1

            try:
                rank = topk_positions.index(pos_in_idmap) + 1
                cos_val = topk_sims[rank - 1]
                rank_info = f"rank={rank}"
            except ValueError:
                cos_val = None
                rank_info = "not in top-5"

            found_shard = shard_for_pos(topk_positions[0], shard_offsets)
            print(
                f"[{i}/{len(sample_ids)}] {status} id={qid} | "
                f"pos(id_map)={pos_in_idmap} → shard#{expected_shard} | "
                f"top1_pos={topk_positions[0]} → shard#{found_shard} | "
                f"{rank_info} | cos={round(cos_val, 4) if cos_val is not None else 'N/A'}"
            )
        except Exception as e:
            print(f"[{i}/{len(sample_ids)}] ❌ id={qid} | ERROR {e}")