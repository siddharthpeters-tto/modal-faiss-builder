import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from supabase import create_client
import boto3
from botocore.client import Config

# === Load credentials ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")

# === Setup clients ===
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

r2 = boto3.client(
    's3',
    endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto',
)

# === Fetch records from Supabase view ===
def fetch_non_image_records():
    print("📥 Fetching from Supabase view: non_image_product_images")
    try:
        response = supabase.table("non_image_product_images").select("id, image_url").limit(1000).execute()
        data = response.data or []

        if not data:
            print("✅ Your product_images table is clean. No non-image files found.")
        else:
            print(f"🚨 Found {len(data)} non-image file(s) in product_images.\n")

        return data

    except Exception as e:
        print("❌ Error fetching from Supabase:", e)
        return []

# === Review results and (optionally) delete ===
def review_and_optionally_delete(entries):
    for row in entries:
        supabase_id = row["id"]
        url = row["image_url"]
        r2_key = urlparse(url).path.lstrip("/")

        print("—" * 50)
        print(f"🆔 Supabase ID: {supabase_id}")
        print(f"🔗 Image URL: {url}")
        print(f"📦 R2 Key: {r2_key}")

        # === DELETE from Supabase (commented out)
        #supabase.table("product_images").delete().eq("id", supabase_id).execute()
        #print("🗑 Deleted from Supabase")

        # === DELETE from R2 (commented out)
        #r2.delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
        #print("🗑 Deleted from R2")

    if entries:
        print("\n✅ Review complete. Nothing was deleted — all delete lines are commented out.")
        print("📝 When ready, uncomment the deletion lines above to apply changes.")

# === Run it all ===
if __name__ == "__main__":
    entries = fetch_non_image_records()
    if entries:
        review_and_optionally_delete(entries)
