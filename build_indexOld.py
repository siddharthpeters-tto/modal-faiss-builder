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
import math # Import math for ceil function

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
            last_processed_ids[index_type] = progress_data.get(index_type, None) # Store UUID string or None
            if last_processed_ids[index_type]:
                print(f"Resuming {index_type} index from image ID: {last_processed_ids[index_type]}")
            else:
                print(f"Starting {index_type} index from the beginning.")
    except Exception as e:
        print(f"No existing {PROGRESS_FILE_NAME} found or error loading: {e}. Starting all indexes from the beginning.")
        for index_type in INDEX_TYPES:
            last_processed_ids[index_type] = None # Start from the beginning if file not found or error

    # --- Get all image IDs and total count ---
    print("Fetching all image IDs from Supabase to establish processing order...")
    all_image_ids = []
    current_offset = 0
    id_fetch_limit = 1000 # Supabase default limit for select queries

    try:
        while True:
            # Fetch IDs in batches
            ids_resp = supabase.table("product_images") \
                .select("id") \
                .order("id") \
                .range(current_offset, current_offset + id_fetch_limit - 1) \
                .execute()

            if not ids_resp.data:
                break # No more IDs to fetch

            all_image_ids.extend([item['id'] for item in ids_resp.data])
            current_offset += id_fetch_limit
            print(f"Fetched {len(all_image_ids)} IDs so far...")

        total_images = len(all_image_ids)
        print(f"There are a total of {total_images} images in your Supabase table.")
    except Exception as e:
        print(f"Could not retrieve all image IDs or total count: {e}. Cannot proceed without a definitive list of IDs.")
        return # Exit if we can't get the full list of IDs

    # Calculate total number of batches
    #total_batches = math.ceil(total_images / BATCH_SIZE) if total_images > 0 else 0

    # --- Determine starting point for processing ---
    start_index = 0
    if last_processed_ids and all(lp_id is not None for lp_id in last_processed_ids.values()):
        # If all index types have a last processed ID, find the earliest one to resume from
        # This ensures we don't skip images if one index type is behind
        earliest_resume_id = min(filter(None, last_processed_ids.values())) # Use filter(None, ...) to ignore None values
        try:
            # Find the index of the earliest resume ID in the sorted list of all IDs
            start_index = all_image_ids.index(earliest_resume_id)
            print(f"Resuming processing from image ID {earliest_resume_id} (index {start_index} in total list).")
        except ValueError:
            print(f"Warning: Earliest resume ID {earliest_resume_id} not found in current list of all IDs. Starting from beginning.")
            start_index = 0 # Fallback if the ID is somehow missing

    total_processed_count = 0

    resumed_total_batches = math.ceil((total_images - start_index) / BATCH_SIZE) if total_images > 0 else 0


    # --- Main Loop for Pagination using ID chunks ---
    # Use enumerate to get the current batch number
    for batch_num, i in enumerate(tqdm(range(start_index, total_images, BATCH_SIZE), desc="Processing Batches"), start=1):
        current_id_chunk = all_image_ids[i : i + BATCH_SIZE]
        if not current_id_chunk:
            break # No more IDs to process

        # Update the tqdm description to show batch progress
        tqdm.write(f"📦 Processing batch {batch_num}/{resumed_total_batches} (IDs: {current_id_chunk[0]} to {current_id_chunk[-1]})...")
        # Fetch images using 'in' clause for the current chunk of IDs
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
            # Keep this check to ensure only valid image formats are processed
            return img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))

        current_batch_for_processing = []
        # Filter out images already processed based on last_processed_ids for each index type
        # And ensure they are valid image formats
        for img in batch_images:
            process_this_image = False
            # An image needs to be processed if its ID is 'after' the last processed ID for *any* index type
            # We determine "after" by checking its position in the `all_image_ids` list
            # It's crucial that img["id"] exists in all_image_ids for this to work.
            try:
                img_global_index = all_image_ids.index(img["id"]) # Get its position in the master list
            except ValueError:
                tqdm.write(f"Warning: Image ID {img['id']} found in batch but not in master list. Skipping.")
                continue # Skip this image if it's not in our master ordered list

            for index_type in INDEX_TYPES:
                last_id_for_type = last_processed_ids.get(index_type)
                if last_id_for_type is None: # If no progress for this type, process it
                    process_this_image = True
                    break
                else:
                    # Find the global index of the last processed ID for this type
                    try:
                        last_id_global_index = all_image_ids.index(last_id_for_type)
                        if img_global_index > last_id_global_index:
                            process_this_image = True
                            break
                    except ValueError:
                        # If last_id_for_type is not found (e.g., deleted from DB), re-process from current image
                        process_this_image = True
                        break

            if process_this_image and is_valid_image(img):
                current_batch_for_processing.append(img)
            else:
                pass # Explicitly pass for the else block to avoid indentation error

        # This print statement was incorrectly indented. It should be outside the inner for loop.
        tqdm.write(f"� {len(current_batch_for_processing)} new valid images to embed in this batch.")

        if not current_batch_for_processing:
            continue # Move to the next batch if all images in this one were already processed or invalid

        color_vecs, structure_vecs, combined_vecs = [], [], []
        id_map_for_batch = [] # Renamed to avoid confusion with global id_map
        last_id_in_current_batch = None # To track the last processed ID (UUID) in the current batch

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
                # Update last_id_in_current_batch with the UUID of the last successfully processed image
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
                entry_id_uuid = entry["id"] # ID is already a UUID string
                if entry_id_uuid not in existing_ids_set:
                    vectors_to_add.append(vectors_from_current_batch[i])
                    ids_to_add_to_map.append(entry_id_uuid) # Store as UUID string in the map
                else:
                    pass # tqdm.write(f"Skipping ID {entry_id_uuid} for {name} index: already in existing map.") # Uncomment for detailed debug


            if not vectors_to_add:
                tqdm.write(f"⚠️ No NEW vectors to save for {name} in this batch. Skipping FAISS update for this index.")
                # Still update progress if this index type is caught up
                if last_processed_uuid_for_batch: # Only update if there was at least one image in this batch
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

            index.add(np_vectors_to_add) # Add only the truly new vectors
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
                json.dump(last_processed_ids, f) # Save the current state of all indexes

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
        # The loop automatically advances to the next batch based on `range(start_index, total_images, BATCH_SIZE)`
        # No need to manually increment offset here.

    print(f"🎉 All FAISS indexes built cleanly and uploaded. Total images processed: {total_processed_count}")

