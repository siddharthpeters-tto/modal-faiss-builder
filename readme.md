# FAISS Sharded Index Builder & Validator

This repo contains scripts to **embed product images, build a sharded FAISS index via Modal**, and **validate index alignment locally** against Supabase.

---

## Components

* **`build_index.py`**
  Modal app that:

  * Fetches product image metadata from Supabase.
  * Encodes images with CLIP embeddings.
  * Builds a **sharded FAISS index** (CPU-based).
  * Uploads `.index` shards and `id_map.json` to Supabase storage.

* **`faiss_sharding.py`**
  Utilities for managing shard offsets and index splitting.

* **`check_alignment.py`**
  Local verification tool to:

  * Confirm index ↔ id\_map consistency.
  * Sample random images, encode them, and search the index.
  * Validate that IDs resolve to the expected shard and position.

---

## Workflow

### 1. Build Embeddings & Shards on Modal

```bash
modal run build_index.py
```

This will:

* Encode product images via CLIP.
* Save shard files (`clip_color_shard_00000.index`, …) and `id_map_color.json` to Supabase storage.

---

### 2. Validate Alignment Locally

```bash
python check_alignment.py
```

Checks include:

* **Sanity check:** every index position maps to the correct ID.
* **Random search test:** encodes sample images and verifies they return correctly.
* **Shard verification:** confirms mapping between global IDs and shard offsets.

---

## Notes

* Cosine similarity normalization is applied so values are consistent regardless of FAISS metric type (L2 or IP).
* Duplicate detection logic has been removed for clarity.
* Make sure environment variables (`SUPABASE_URL`, `SUPABASE_KEY`, `BUCKET_NAME`, `PREFIX`) are set in `.env`.

---

## Example Outputs

### From `build_index.py` (Modal)

```
Batch 1: Total=500, Embedded=500
Batch 2: Total=500, Embedded=500
📊 Brands processed:
aw: 137 new vectors
unknown: 863 new vectors
```

### From `check_alignment.py`

```
[42/200] ✅ id=abc123 | pos(id_map)=102 → shard#2 | top1_pos=102 → shard#2 | rank=1 | cos=0.9987
⚠️ Mismatch for color: index has 1000, id_map has 2000
```

---

## Quickstart

1. **Clone repo & install deps**

   ```bash
   git clone https://github.com/<your-repo>.git
   cd <your-repo>
   pip install -r requirements.txt
   ```

2. **Setup `.env`** with your Supabase credentials and FAISS config:

   ```
   SUPABASE_URL=...
   SUPABASE_KEY=...
   BUCKET_NAME=faiss
   PREFIX=clip_color
   ```

3. **Run Modal build**

   ```bash
   modal run build_index.py
   ```

4. **Validate locally**

   ```bash
   python check_alignment.py
   ```

---

## Repo Structure

```
build_index.py       # Modal app for embedding + FAISS sharding
faiss_sharding.py    # Shard helpers
check_alignment.py   # Local validator
readme.md            # This file
```

