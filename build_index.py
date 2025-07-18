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

    # --- Configuration for pagination and resumption ---
    BATCH_SIZE = 200
    INDEX_TYPES = ["color", "structure", "combined"]
    PROGRESS_FILE_NAME = "progress.json"

    # Initialize or load last processed ID for each index type from Supabase Storage
    last_processed_ids = {}
    try:
        # Attempt to download the progress file
        resp = supabase.storage.from_("faiss").download(PROGRESS_FILE_NAME)
        progress_data = json.loads(resp.decode('utf-8'))
        for index_type in INDEX_TYPES:
            last_processed_ids[index_type] = progress_data.get(index_type, 0)
            print(f"Resuming {index_type} index from image ID: {last_processed_ids[index_type]}")
    except Exception as e:
        print(f"No existing {PROGRESS_FILE_NAME} found or error loading: {e}. Starting all indexes from the beginning.")
        for index_type in INDEX_TYPES:
            last_processed_ids[index_type] = 0 # Start from the beginning if file not found or error

    # --- Main Loop for Pagination ---
    offset = 0
    total_processed_count = 0
    # Start offset from the minimum of last_processed_ids to ensure we don't miss any images
    # This assumes image IDs are sequential or at least monotonically increasing.
    # If IDs are not sequential, a different strategy (e.g., fetching all IDs and iterating) might be needed.
    # For now, we'll assume they are sequential enough for range-based pagination.
    if last_processed_ids:
        offset = min(last_processed_ids.values()) # Start fetching from the earliest unprocessed image

    while True:
        print(f"📦 Fetching product images from Supabase (offset: {offset}, limit: {BATCH_SIZE})...")
        # Fetch a batch of images, ordered by ID to ensure consistent pagination
        # Removed the .or_() filter to fetch all images
        resp = supabase.table("product_images") \
            .select("id, image_url") \
            .order("id") \
            .range(offset, offset + BATCH_SIZE - 1) \
            .execute()

        batch_images = resp.data
        if not batch_images:
            print("🎉 No more images to process. Exiting.")
            break

        print(f"✅ Retrieved {len(batch_images)} images in this batch.")

        def is_valid_image(img):
            # Keep this check to ensure only valid image formats are processed
            return img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))

        current_batch_for_processing = []
        # Filter out images already processed based on last_processed_ids for each index type
        for img in batch_images:
            process_this_image = False # Assume not to process until proven otherwise
            # An image needs to be processed if its ID is greater than the last processed ID for *any* index type
            # This ensures we don't skip images if one index type is behind.
            for index_type in INDEX_TYPES:
                if img["id"] > last_processed_ids.get(index_type, 0):
                    process_this_image = True
                    break # Only need one index to be behind to process this image

            if process_this_image and is_valid_image(img):
                current_batch_for_processing.append(img)

        print(f"🔄 {len(current_batch_for_processing)} new valid images to embed in this batch.")

        if not current_batch_for_processing:
            offset += BATCH_SIZE
            continue # Move to the next batch if all images in this one were already processed or invalid

        color_vecs, structure_vecs, combined_vecs = [], [], []
        id_map = []
        last_id_in_current_batch = 0 # To track the last processed ID in the current batch

        for img in tqdm(current_batch_for_processing):
            try:
                r = requests.get(img["image_url"], timeout=10)
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    print(f"Skipping {img['image_url']}: Not an image content type.")
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
                id_map.append({
                    "id": img["id"],
                    "image_url": img["image_url"]
                })
                last_id_in_current_batch = max(last_id_in_current_batch, img["id"]) # Update last_id_in_current_batch

            except (UnidentifiedImageError, Exception) as e:
                print(f"❌ Failed on {img['image_url']}: {e}")
                continue

        os.makedirs("/tmp/faiss", exist_ok=True)

        def save_index_and_update_progress(name, vectors_from_current_batch, id_entries_from_current_batch, current_batch_last_id):
            # Load existing ID map for this specific index type
            id_map_path = f"/tmp/faiss/id_map_{name}.json"
            existing_id_map_data = []
            try:
                resp = supabase.storage.from_("faiss").download(f"id_map_{name}.json")
                existing_id_map_data = json.loads(resp.decode('utf-8'))
            except Exception as e:
                print(f"No existing {name} id map found or error loading: {e}. Creating new.")

            existing_ids_set = set(existing_id_map_data)

            # Filter vectors and IDs to only include those not already in this index's ID map
            vectors_to_add = []
            ids_to_add_to_map = []
            for i, entry in enumerate(id_entries_from_current_batch):
                if entry["id"] not in existing_ids_set:
                    vectors_to_add.append(vectors_from_current_batch[i])
                    ids_to_add_to_map.append(entry["id"])
                else:
                    print(f"Skipping ID {entry['id']} for {name} index: already in existing map.")


            if not vectors_to_add:
                print(f"⚠️ No NEW vectors to save for {name} in this batch. Skipping FAISS update for this index.")
                # Still update progress if this index type is caught up
                last_processed_ids[name] = current_batch_last_id
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
                print(f"Loaded existing {name} index with {index.ntotal} vectors.")
            except Exception as e:
                print(f"No existing {name} index found or error loading: {e}. Creating new index.")
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
            last_processed_ids[name] = current_batch_last_id

            # Update progress.json in Supabase Storage
            with open(PROGRESS_FILE_NAME, "w") as f:
                json.dump(last_processed_ids, f) # Save the current state of all indexes

            with open(PROGRESS_FILE_NAME, "rb") as f:
                supabase.storage.from_("faiss").upload(
                    file=f,
                    path=PROGRESS_FILE_NAME,
                    file_options={"content-type": "application/json", "x-upsert": "true"}
                )

            print(f"✅ Uploaded {name} index and updated progress with {len(vectors_to_add)} NEW vectors. Last processed ID for {name}: {current_batch_last_id}")


        # Process and save for each index type
        # It's important to update the progress.json only AFTER all index types for the batch are successfully processed
        # However, to ensure progress is saved even if one index type fails for a batch,
        # we'll update the progress.json after each index type's save_index_and_update_progress call.
        # This means if the process crashes, the next run will pick up from the last successfully saved index type.
        if color_vecs:
            save_index_and_update_progress("color", color_vecs, id_map, last_id_in_current_batch)
        if structure_vecs:
            save_index_and_update_progress("structure", structure_vecs, id_map, last_id_in_current_batch)
        if combined_vecs:
            save_index_and_update_progress("combined", combined_vecs, id_map, last_id_in_current_batch)

        total_processed_count += len(current_batch_for_processing)
        offset += BATCH_SIZE # Move to the next batch

    print(f"🎉 All FAISS indexes built cleanly and uploaded. Total images processed: {total_processed_count}")

