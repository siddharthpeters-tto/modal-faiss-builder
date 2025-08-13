# FAISS Index Sharding on Modal with Supabase Storage

This project embeds product images, stores results in **FAISS indexes**, and uploads them to **Supabase Storage** in shards to avoid file size limits. It is designed for **resilience**, allowing runs to resume from progress if interrupted.

## Key Features

- **Three index types**: `color`, `structure`, and `combined`.
- **Sharding**: Each index is split into shards (~1,000 images each) to keep files below Supabase’s ~50 MB per-file limit.
- **Immediate uploads**: Shards are uploaded to Supabase after each batch — no reliance on large local or ephemeral storage.
- **Resumable runs**: Progress is tracked in `progress.json` stored in Supabase, enabling restarts without reprocessing.
- **Batch processing**: Embedding runs in batches of 1,000 images for efficiency.
- **Error handling**: Skips failed images but records completed ones.

## Structure

- `build_index.py` – Main script to fetch unprocessed images, batch embed, and write shards.
- `faiss_sharding.py` – Manages shard creation, rotation, uploads, and manifest management.
- `.env.example` – Environment variable template.

## Requirements

- **Modal** account & CLI (`pip install modal`)
- **Supabase** project with `images` table and storage bucket for FAISS files
- Dependencies: `faiss-cpu`, `tqdm`, `supabase-py`, `numpy`, `openai`, `Pillow`, etc.

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
modal deploy build_index.py
```
Or run locally via Modal:
```bash
modal run build_index.py
```

Process:
1. Load unprocessed IDs from Supabase.
2. Embed in batches of 1,000.
3. Append vectors to the current shard per index type.
4. Rotate & upload shard if it reaches the size threshold.
5. Flush shards at batch end.
6. Save `progress.json` in Supabase.

## Recovery

If interrupted, the script resumes from `progress.json`, skipping already processed IDs.

## Supabase Storage Layout
```
supabase-storage/
  faiss/
    color_shard_001.faiss
    structure_shard_001.faiss
    combined_shard_001.faiss
  progress.json
```
