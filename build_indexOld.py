from modal import App, Image, Secret
import os, json
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
import math  # Import math for ceil function

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
    name="build-multi-index-faiss",
    image=image,
    secrets=[Secret.from_name("supabase-creds")]
)

@app.function(gpu="A10G", timeout=3600)
def build_index_supabase():
    load_dotenv()
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ Using device: {device}")

    # --- Configuration for pagination and resumption ---
    BATCH_SIZE = 200
    INDEX_TYPES = ["color", "structure", "combined"]
    PROGRESS_FILE_NAME = "progress.json"

    # Initialize or load last processed ID (UUID string) for each index type from Supabase Storage
    last_processed_ids = {}
    try:
        resp = supabase.storage.from_("faiss").download(PROGRESS_FILE_NAME)
        progress_data = json.loads(resp.decode('utf-8'))
        for index_type in INDEX_TYPES:
            last_processed_ids[index_type] = progress_data.get(index_type, None)  # Store UUID string or None
            if last_processed_ids[index_type]:
                print(f"Resuming {index_type} index from image ID: {last_processed_ids[index_type]}")
            else:
                print(f"Starting {index_type} index from the beginning.")
    except Exception as e:
        print(f"No existing {PROGRESS_FILE_NAME} found or error loading: {e}. Starting all indexes from the beginning.")
        for index_type in INDEX_TYPES:
            last_processed_ids[index_type] = None  # Start from the beginning if file not found or error

    # --- Load existing indexed IDs once, to skip re-embedding ---
    # We load per-index ID maps (not just a union) so we only skip an image if
    # it already exists in *all* three indexes. If one is missing it, we'll embed
    # once and the save step will add it only to the missing index.
    indexed_ids_by_type = {}
    for name in INDEX_TYPES:
        try:
            data = supabase.storage.from_("faiss").download(f"id_map_{name}.json")
            indexed_ids_by_type[name] = set(json.loads(data.decode("utf-8")))
            print(f"🔎 Loaded {len(indexed_ids_by_type[name])} existing IDs for '{name}' index.")
        except Exception as e:
            tqdm.write(f"No existing id_map for {name} or error loading: {e}. Treating as empty.")
            indexed_ids_by_type[name] = set()

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

    # --- Determine starting point for processing ---
    # Force a full scan from the beginning to avoid missing newly inserted UUIDs
    # that might sort *before* the saved checkpoints. We'll skip quickly using
    # the id_map sets above, so we don't re-embed existing images.
    start_index = 0

    total_processed_count = 0
    resumed_total_batches = math.ceil((total_images - start_index) / BATCH_SIZE) if total_images > 0 else 0

    # --- Main Loop for Pagination using ID chunks ---
    for batch_num, i in enumerate(tqdm(range(start_index, total_images, BATCH_SIZE), desc="Processing Batches"), start=1):
        current_id_chunk = all_image_ids[i: i + BATCH_SIZE]
        if not current_id_chunk:
            break  # No more IDs to process

        # Fetch images using 'in' clause for the current chunk of IDs
        tqdm.write(f"📦 Processing batch {batch_num}/{resumed_total_batches} (IDs: {current_id_chunk[0]} to {current_id_chunk[-1]})…")
        resp = supabase.table("product_images") \
            .select("id, image_url") \
            .in_("id", current_id_chunk) \
            .execute()

        batch_images = resp.data
        if not batch_images:
            tqdm.write(f"⚠️ No images retrieved for chunk starting with ID {current_id_chunk[0]}. Skipping this batch.")
            continue

        tqdm.write(f"✅ Retrieved {len(batch_images)} images in this batch.")

        def is_valid_image(img):
            return img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))

        # Build the list of images we actually need to embed (skip those present in all indexes)
        current_batch_for_processing = []
        for img in batch_images:
            if not is_valid_image(img):
                continue
            # Skip if this ID already exists in *all* index maps
            already_in_all = all(img["id"] in indexed_ids_by_type[t] for t in INDEX_TYPES)
            if already_in_all:
                continue
            current_batch_for_processing.append(img)

        tqdm.write(f"🆕 {len(current_batch_for_processing)} new valid images to embed in this batch.")
        if not current_batch_for_processing:
            continue  # Move to the next batch

        color_vecs, structure_vecs, combined_vecs = [], [], []
        id_map_for_batch = []  # Local map for this batch
        last_id_in_current_batch = None  # Track the last processed ID (UUID) in the current batch

        for img in tqdm(current_batch_for_processing, desc=f"Embedding Batch {batch_num}/{resumed_total_batches}"):
            try:
                r = requests.get(img["image_url"], timeout=10)
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    tqdm.write(f"Skipping {img['image_url']}: Not an image content type.")
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
                id_map_for_batch.append({
                    "id": img["id"],
                    "image_url": img["image_url"]
                })
                last_id_in_current_batch = img["id"]

            except (UnidentifiedImageError, Exception) as e:
                tqdm.write(f"❌ Failed on {img['image_url']}: {e}")
                continue

        os.makedirs("/tmp/faiss", exist_ok=True)

        def save_index_and_update_progress(name, vectors_from_current_batch, id_entries_from_current_batch, last_processed_uuid_for_batch):
            # Load existing ID map for this specific index type
            id_map_path = f"/tmp/faiss/id_map_{name}.json"
            existing_id_map_data = []
            try:
                resp = supabase.storage.from_("faiss").download(f"id_map_{name}.json")
                existing_id_map_data = json.loads(resp.decode('utf-8'))
            except Exception as e:
                tqdm.write(f"No existing {name} id map found or error loading: {e}. Creating new.")

            existing_ids_set = set(existing_id_map_data)

            # Filter vectors and IDs to only include those not already in this index's ID map
            vectors_to_add = []
            ids_to_add_to_map = []
            for i, entry in enumerate(id_entries_from_current_batch):
                entry_id_uuid = entry["id"]  # UUID string
                if entry_id_uuid not in existing_ids_set:
                    vectors_to_add.append(vectors_from_current_batch[i])
                    ids_to_add_to_map.append(entry_id_uuid)
                else:
                    pass

            if not vectors_to_add:
                tqdm.write(f"⚠️ No NEW vectors to save for {name} in this batch. Skipping FAISS update for this index.")
                # Still update progress if this index type is caught up
                if last_processed_uuid_for_batch:  # Only update if there was at least one image in this batch
                    last_processed_ids[name] = last_processed_uuid_for_batch
                    with open(PROGRESS_FILE_NAME, "w") as f:
                        json.dump(last_processed_ids, f)
                    with open(PROGRESS_FILE_NAME, "rb") as f:
                        supabase.storage.from_("faiss").upload(
                            file=f,
                            path=PROGRESS_FILE_NAME,
                            file_options={"content-type": "application/json", "x-upsert": "true"}
                        )
                return

            np_vectors_to_add = np.stack(vectors_to_add).astype(np.float32)
            faiss.normalize_L2(np_vectors_to_add)

            index_path = f"/tmp/faiss/clip_{name}.index"
            try:
                # Attempt to download existing index to append
                resp = supabase.storage.from_("faiss").download(f"clip_{name}.index")
                with open(index_path, "wb") as f:
                    f.write(resp)
                index = faiss.read_index(index_path)
                tqdm.write(f"Loaded existing {name} index with {index.ntotal} vectors.")
            except Exception as e:
                tqdm.write(f"No existing {name} index found or error loading: {e}. Creating new index.")
                index = faiss.IndexFlatIP(np_vectors_to_add.shape[1])

            index.add(np_vectors_to_add)  # Add only the truly new vectors
            faiss.write_index(index, index_path)

            # Extend the existing ID map with only the truly new IDs
            existing_id_map_data.extend(ids_to_add_to_map)

            with open(id_map_path, "w") as f:
                json.dump(existing_id_map_data, f)

            # Only upload the index and the id_map.json
            for file_path in [index_path, id_map_path]:
                with open(file_path, "rb") as f:
                    file_name = os.path.basename(file_path)
                    supabase.storage.from_("faiss").upload(
                        file=f,
                        path=file_name,
                        file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
                    )

            # Update global last_processed_ids for this index type
            last_processed_ids[name] = last_processed_uuid_for_batch

            # Update progress.json in Supabase Storage
            with open(PROGRESS_FILE_NAME, "w") as f:
                json.dump(last_processed_ids, f)  # Save the current state of all indexes

            with open(PROGRESS_FILE_NAME, "rb") as f:
                supabase.storage.from_("faiss").upload(
                    file=f,
                    path=PROGRESS_FILE_NAME,
                    file_options={"content-type": "application/json", "x-upsert": "true"}
                )

            tqdm.write(f"✅ Uploaded {name} index and updated progress with {len(vectors_to_add)} NEW vectors. Last processed ID for {name}: {last_processed_uuid_for_batch}")

        # Process and save for each index type
        if color_vecs:
            save_index_and_update_progress("color", color_vecs, id_map_for_batch, last_id_in_current_batch)
        if structure_vecs:
            save_index_and_update_progress("structure", structure_vecs, id_map_for_batch, last_id_in_current_batch)
        if combined_vecs:
            save_index_and_update_progress("combined", combined_vecs, id_map_for_batch, last_id_in_current_batch)

        total_processed_count += len(current_batch_for_processing)

    print(f"🎉 All FAISS indexes built cleanly and uploaded. Total images processed: {total_processed_count}")
