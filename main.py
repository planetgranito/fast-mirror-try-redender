# main.py
import os
import json
import random
import base64
import logging
from io import BytesIO
from pathlib import Path
from datetime import timedelta
from typing import List

from openai import OpenAI
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# genai
import google.genai as genai

# GCS
from google.cloud import storage
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# --- Config (env vars) ---
BASE_DIR = Path(__file__).parent
STATIC_FOLDER = Path(os.getenv("STATIC_FOLDER", BASE_DIR / "static"))
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)

# GCS settings (required)
GCP_SA_JSON = os.getenv("GCP_SERVICE_ACCOUNT_KEY")  # paste JSON in Render secret
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CATALOGUE_PREFIX = os.getenv("GCS_CATALOGUE_PREFIX", "catalogue/")
GCS_GENERATED_PREFIX = os.getenv("GCS_GENERATED_PREFIX", "generated/")

# AI keys (optional, but your local app used them)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize OpenAI client (guarded)
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning("OpenAI client init failed: %s", e)

# init genai client (guarded)
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning("GenAI client init failed: %s", e)

# --- GCS client helper ---
def get_gcs_client():
    """
    Create a google.cloud.storage.Client.
    Priority:
      1) If GCP_SERVICE_ACCOUNT_KEY env var (full JSON string) is present -> use it.
      2) Otherwise rely on Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS file path or ADC).
    """
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
    if sa_json:
        # JSON provided in env var (useful for Render secrets)
        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(info)
        project = info.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        return storage.Client(project=project, credentials=creds)

    # Fallback to Application Default Credentials (file pointed by GOOGLE_APPLICATION_CREDENTIALS,
    # or other ADC mechanisms). This is the recommended local testing method.
    # storage.Client() will automatically read GOOGLE_APPLICATION_CREDENTIALS if set.
    return storage.Client()

"""def get_gcs_client():
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SERVICE_ACCOUNT_KEY env var not set")
    info = json.loads(GCP_SA_JSON)
    creds = service_account.Credentials.from_service_account_info(info)
    project = info.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
    return storage.Client(project=project, credentials=creds)"""

def list_gcs_image_keys(prefix: str = GCS_CATALOGUE_PREFIX, max_results: int = 1000) -> List[str]:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = client.list_blobs(bucket.name, prefix=prefix, max_results=max_results)
    keys = []
    for b in blobs:
        # skip directory markers and 0-byte blobs
        if b.name.endswith("/") or (b.size is not None and b.size == 0):
            continue
        if b.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            keys.append(b.name)
    return keys

def generate_signed_url_for_key(key: str, expires_hours: int = 24) -> str:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(key)
    return blob.generate_signed_url(expiration=timedelta(hours=expires_hours))

def download_blob_bytes(key: str) -> bytes:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(key)
    return blob.download_as_bytes()

def upload_bytes_to_gcs(data: bytes, dest_key: str, content_type: str = "image/png"):
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(dest_key)
    blob.upload_from_string(data, content_type=content_type)
    return dest_key

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Keep static for index.html (optional)
app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")


@app.get("/catalogue")
def catalogue(n: int = 8, expires_hours: int = 1):
    """Return up to n signed URLs for random catalogue images from GCS."""
    if not GCS_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not set")
    keys = list_gcs_image_keys(prefix=GCS_CATALOGUE_PREFIX)
    if not keys:
        raise HTTPException(status_code=404, detail="No catalogue images found in GCS.")
    chosen = random.sample(keys, min(n, len(keys)))
    urls = [generate_signed_url_for_key(k, expires_hours=expires_hours) for k in chosen]
    return {"images": urls, "keys": chosen}


@app.post("/try-on")
async def try_on(
    model_photo: UploadFile = File(...),
    dress_key_or_name: str = Form(...),
    expires_hours: int = Form(24),
):
    """
    model_photo: uploaded image of the person
    dress_key_or_name: either a filename like 'image1.jpg' or a full key 'catalogue/image1.jpg'
    """
    if not GCS_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not set")

    # read model image bytes
    model_photo_bytes = await model_photo.read()
    try:
        model_img = Image.open(BytesIO(model_photo_bytes)).convert("RGBA")
    except Exception as e:
        logger.exception("Invalid model photo: %s", e)
        raise HTTPException(status_code=400, detail="Invalid model photo uploaded")

    # resolve dress key
    dress_key = dress_key_or_name
    if "/" not in dress_key:
        dress_key = (GCS_CATALOGUE_PREFIX.rstrip("/") + "/" + dress_key).lstrip("/")

    # download dress bytes from GCS
    try:
        dress_bytes = download_blob_bytes(dress_key)
        dress_img = Image.open(BytesIO(dress_bytes)).convert("RGBA")
    except Exception as e:
        logger.exception("Failed to download dress from GCS: %s", e)
        raise HTTPException(status_code=404, detail=f"Could not find dress '{dress_key_or_name}' in catalogue")

    # --- call Gemini/GenAI (keeps your existing pattern) ---
    if not genai_client:
        # If genai not configured, return helpful error
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set or genai client not initialized")

    prompt = (
        "You are a virtual stylist. Take the T-shirt from the first image and realistically place it on the person in the second image. "
        "Make sure it looks like an authentic, full-body fashion photo. The T-shirt must appear naturally fitted, with correct proportions, "
        "wrinkles, fabric texture, and realistic shadows and lighting, as if the person is actually wearing it in a professional photoshoot."
    )

    try:
        # re-use your same call shape — the SDK sometimes accepts PIL images; adjust if your SDK requires bytes
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[dress_img, model_img, prompt],
        )
        image_parts = [part for part in response.candidates[0].content.parts if getattr(part, "inline_data", None)]
        if not image_parts:
            raise ValueError("No image data found in Gemini response.")
        image_bytes = image_parts[0].inline_data.data
    except Exception as e:
        logger.exception("GenAI generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Image generation failed. See server logs.")

    # upload generated bytes to GCS
    gen_filename = f"result_{Path(dress_key).stem}_{random.randint(1000,9999)}.png"
    gen_key = (GCS_GENERATED_PREFIX.rstrip("/") + "/" + gen_filename).lstrip("/")
    try:
        upload_bytes_to_gcs(image_bytes, gen_key, content_type="image/png")
    except Exception as e:
        logger.exception("Failed to upload generated image to GCS: %s", e)
        raise HTTPException(status_code=500, detail="Failed to upload generated image to GCS.")

    # generate signed URL for result
    result_url = generate_signed_url_for_key(gen_key, expires_hours=expires_hours)

    # optional: call OpenAI for description (guarded)
    description = None
    if openai_client:
        try:
            # create base64 and call OpenAI as you did earlier
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            desc_prompt = (
                "Write one single sentence (no line breaks) describing this t-shirt. It must contain 39 to 45 words in total, "
                "single line only. Make it elegant and customer-friendly, describing the beauty and style appeal."
            )
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": desc_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ]
                    }
                ],
            )
            description = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("OpenAI description generation failed: %s", e)
            description = None

    return {"result_key": gen_key, "result_url": result_url, "description": description}


@app.get("/")
def index():
    index_file = STATIC_FOLDER / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok", "note": "index.html not found; frontend can use GCS signed URLs."}


# Only for local testing (do not use reload=True on Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)
