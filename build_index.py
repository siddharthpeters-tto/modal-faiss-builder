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
import math

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
    name="build-multi-index-faiss-resilient",
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

    BATCH_SIZE = 200
    INDEX_TYPES = ["color", "structure", "combined"]
    PROGRESS_FILE_NAME = "progress.json"
    LOCAL_FAISS_DIR = "/tmp/faiss"
    os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

    # Load progress
    last_processed_ids = {}
    progress_path = os.path.join(LOCAL_FAISS_DIR, PROGRESS_FILE_NAME)
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            last_processed_ids = json.load(f)
    else:
        try:
            resp = supabase.storage.from_("faiss").download(PROGRESS_FILE_NAME)
            last_processed_ids = json.loads(resp.decode('utf-8'))
        except:
            last_processed_ids = {t: None for t in INDEX_TYPES}

    # Load id_maps per type
    indexed_ids_by_type = {}
    for name in INDEX_TYPES:
        local_map = os.path.join(LOCAL_FAISS_DIR, f"id_map_{name}.json")
        if os.path.exists(local_map):
            with open(local_map, "r") as f:
                indexed_ids_by_type[name] = set(json.load(f))
        else:
            try:
                data = supabase.storage.from_("faiss").download(f"id_map_{name}.json")
                indexed_ids_by_type[name] = set(json.loads(data.decode("utf-8")))
            except:
                indexed_ids_by_type[name] = set()

    # Load indexes per type
    indexes = {}
    for name in INDEX_TYPES:
        local_idx = os.path.join(LOCAL_FAISS_DIR, f"clip_{name}.index")
        if os.path.exists(local_idx):
            indexes[name] = faiss.read_index(local_idx)
        else:
            try:
                resp = supabase.storage.from_("faiss").download(f"clip_{name}.index")
                with open(local_idx, "wb") as f:
                    f.write(resp)
                indexes[name] = faiss.read_index(local_idx)
            except:
                indexes[name] = None

    # Fetch IDs from Supabase
    all_image_ids = []
    offset = 0
    while True:
        ids_resp = supabase.table("product_images").select("id").order("id").range(offset, offset+999).execute()
        if not ids_resp.data:
            break
        all_image_ids.extend([item['id'] for item in ids_resp.data])
        offset += 1000
    total_images = len(all_image_ids)
    print(f"📊 Total images: {total_images}")

    # Process batches
    for batch_num, i in enumerate(tqdm(range(0, total_images, BATCH_SIZE)), start=1):
        current_ids = all_image_ids[i:i+BATCH_SIZE]
        resp = supabase.table("product_images").select("id, image_url").in_("id", current_ids).execute()
        batch_images = resp.data

        current_batch = [img for img in batch_images if img["image_url"].lower().endswith((".jpg", ".jpeg", ".png"))]
        if not current_batch:
            continue

        color_vecs, structure_vecs, combined_vecs = [], [], []
        id_map_for_batch = []
        last_id_in_batch = None

        for img in current_batch:
            already_in_all = all(img["id"] in indexed_ids_by_type[t] for t in INDEX_TYPES)
            if already_in_all:
                continue
            try:
                r = requests.get(img["image_url"], timeout=10)
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    continue
                pil_color = PILImage.open(BytesIO(r.content)).convert("RGB")
                pil_grey = pil_color.convert("L").convert("RGB")

                color_tensor = preprocess(pil_color).unsqueeze(0).to(device)
                grey_tensor = preprocess(pil_grey).unsqueeze(0).to(device)

                with torch.no_grad():
                    color_feat = model.encode_image(color_tensor); color_feat /= color_feat.norm(dim=-1, keepdim=True)
                    grey_feat = model.encode_image(grey_tensor); grey_feat /= grey_feat.norm(dim=-1, keepdim=True)
                    combined_feat = (color_feat + grey_feat) / 2; combined_feat /= combined_feat.norm(dim=-1, keepdim=True)

                color_vecs.append(color_feat.cpu().numpy().flatten())
                structure_vecs.append(grey_feat.cpu().numpy().flatten())
                combined_vecs.append(combined_feat.cpu().numpy().flatten())

                id_map_for_batch.append({"id": img["id"], "image_url": img["image_url"]})
                last_id_in_batch = img["id"]
            except (UnidentifiedImageError, Exception):
                continue

        def save_local_index(name, vecs, id_entries, last_id):
            id_map_path = os.path.join(LOCAL_FAISS_DIR, f"id_map_{name}.json")
            existing_ids = set()
            if os.path.exists(id_map_path):
                with open(id_map_path, "r") as f:
                    existing_ids = set(json.load(f))

            vectors_to_add, ids_to_add = [], []
            for idx, entry in enumerate(id_entries):
                if entry["id"] not in existing_ids:
                    vectors_to_add.append(vecs[idx])
                    ids_to_add.append(entry["id"])

            if not vectors_to_add:
                last_processed_ids[name] = last_id
                with open(progress_path, "w") as f:
                    json.dump(last_processed_ids, f)
                return

            np_vecs = np.stack(vectors_to_add).astype(np.float32)
            faiss.normalize_L2(np_vecs)

            index_path = os.path.join(LOCAL_FAISS_DIR, f"clip_{name}.index")
            if indexes[name] is None:
                indexes[name] = faiss.IndexFlatIP(np_vecs.shape[1])
            indexes[name].add(np_vecs)
            faiss.write_index(indexes[name], index_path)

            existing_ids.update(ids_to_add)
            with open(id_map_path, "w") as f:
                json.dump(list(existing_ids), f)

            last_processed_ids[name] = last_id
            with open(progress_path, "w") as f:
                json.dump(last_processed_ids, f)

        if color_vecs:
            save_local_index("color", color_vecs, id_map_for_batch, last_id_in_batch)
        if structure_vecs:
            save_local_index("structure", structure_vecs, id_map_for_batch, last_id_in_batch)
        if combined_vecs:
            save_local_index("combined", combined_vecs, id_map_for_batch, last_id_in_batch)

    print("☁️ Uploading all data to Supabase...")
    for name in INDEX_TYPES:
        with open(os.path.join(LOCAL_FAISS_DIR, f"clip_{name}.index"), "rb") as f:
            supabase.storage.from_("faiss").upload(path=f"clip_{name}.index", file=f, file_options={"x-upsert": "true"})
        with open(os.path.join(LOCAL_FAISS_DIR, f"id_map_{name}.json"), "rb") as f:
            supabase.storage.from_("faiss").upload(path=f"id_map_{name}.json", file=f, file_options={"x-upsert": "true"})
    with open(progress_path, "rb") as f:
        supabase.storage.from_("faiss").upload(path=PROGRESS_FILE_NAME, file=f, file_options={"x-upsert": "true"})

    print("🎉 All FAISS indexes built locally with full resiliency and uploaded at the end.")