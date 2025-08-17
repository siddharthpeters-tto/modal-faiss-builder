# FAISS Index Sharding on Modal with Supabase Storage

This project embeds product images, stores results in **FAISS indexes**, and uploads them to **Supabase Storage** in shards to avoid file size limits. It is designed for **resilience**, allowing runs to resume from progress if interrupted.

## Key Features

- **Three index types** supported in architecture: `color`, `structure`, and `combined` (current runs may use `color` only).
- **Sharding**: Each index is split into shards (~1,000 images each) to keep files below Supabase’s ~50 MB per-file limit.
- **Immediate uploads**: Shards are uploaded to Supabase after each batch — no reliance on large local or ephemeral storage.
- **Resumable runs**: Progress is tracked in `progress.json` stored in Supabase, enabling restarts without reprocessing.
- **Batch processing**: Embedding runs in configurable batches for efficiency.
- **Error handling**: Skips failed images but records completed ones.
- **Merged local + remote state**: Index and ID map merging from both local disk and Supabase ensures no progress loss.
- **GPU acceleration**: Runs on GPU (`A10G`) in Modal for faster embedding.
- **Brand tracking**: Tracks number of vectors added per brand for monitoring.
- **Safety checks**: Asserts ID map and FAISS vector counts match after each checkpoint.
- **Safer normalization**: Uses `clamp(min=1e-12)` to avoid divide-by-zero issues.
- **Alignment verification**: `check_alignment.py` script verifies that index vectors and ID map are perfectly aligned, detects duplicates, and runs random match checks.

## Structure

- `build_400.py` – Production-grade build script with merged enhancements (multi-index capable, resumable, robust sharding, and safety checks).
- `faiss_sharding.py` – Manages shard creation, rotation, uploads, manifest management, and loading sharded indexes.
- `check_alignment.py` – Verifies alignment between FAISS index and ID map, detects duplicates, and validates match accuracy.
- `.env.example` – Environment variable template.

## Requirements

- **Modal** account & CLI (`pip install modal`)
- **Supabase** project with `product_images` table and storage bucket for FAISS files
- Dependencies: `faiss-cpu`, `torch`, `tqdm`, `supabase-py`, `numpy`, `Pillow`, `clip-by-openai`, etc.

## Setup

1. Create a Modal secret:
```bash
modal secret create supabase-creds \
  SUPABASE_URL="YOUR_SUPABASE_URL" \
  SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
```
2. Clone and configure:
```bash
git clone https://github.com/YOUR_USERNAME/modal-faiss-builder.git
cd modal-faiss-builder
```

## Running

Deploy:
```bash
modal deploy build_400.py
```
Run with environment-configured limits (e.g., MAX_IMAGES=400 in `.env`):
```bash
modal run build_index.py::build_index_supabase
```

Process:
1. Load unprocessed IDs from Supabase, filtering out already indexed ones and IDs before last checkpoint.
2. Embed in batches.
3. Append vectors to the current shard per index type.
4. Rotate & upload shard if it reaches the size threshold.
5. Flush shards at batch end.
6. Save `progress.json` and updated ID maps to Supabase (and locally).
7. Verify FAISS vector count matches ID map count after each checkpoint.

## Recovery

If interrupted, the script resumes from `progress.json`, skipping already processed IDs.

## Alignment Check

Run `check_alignment.py` to validate that all vectors in the FAISS index correspond exactly to the correct IDs in the ID map, with no duplicates or mismatches.

## Supabase Storage Layout
```
supabase-storage/
  faiss/
    clip_color_shard_00001.index
    clip_structure_shard_00001.index
    clip_combined_shard_00001.index
    id_map_color.json
    id_map_structure.json
    id_map_combined.json
    progress.json
```
