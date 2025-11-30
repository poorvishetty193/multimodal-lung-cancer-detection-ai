from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import random
import time
import io
from minio import Minio
from PIL import Image
import os

# ---------------------------
#  MinIO Client
# ---------------------------
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ROOT_USER", "minio"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False
)

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI()

# ---------------------------
# Pydantic Request Schema
# ---------------------------
class PredictRequest(BaseModel):
    job_id: str
    ct_object: str   # example: "uploads/<uuid>/image.png"
    metadata: dict = {}


# ---------------------------
# Helper: Download from MinIO
# ---------------------------
def load_image_from_minio(path: str) -> np.ndarray:
    """
    Downloads a PNG/JPG from MinIO and returns numpy image.
    """
    bucket, object_name = path.split("/", 1)

    resp = minio_client.get_object(bucket, object_name)
    data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("L")  # grayscale
    img = img.resize((256, 256))
    return np.array(img).astype(np.float32)


# ---------------------------
# CT Processing (Dummy / Replace later)
# ---------------------------
def fake_ct_model(volume: np.ndarray):
    """
    A placeholder CT analysis function.
    Replace with your real PyTorch model later.
    """

    # Simulate 1–3 nodules
    nodules = []
    for _ in range(random.randint(1, 3)):
        nodules.append({
            "x": random.randint(20, 200),
            "y": random.randint(20, 200),
            "z": random.randint(10, 60),
            "diameter_mm": random.uniform(5, 20),
            "confidence": round(random.uniform(0.70, 0.98), 2),
            "nodule_probs": {
                "adenocarcinoma": round(random.random(), 2),
                "squamous": round(random.random(), 2),
                "large_cell": round(random.random(), 2),
                "small_cell": round(random.random(), 2)
            }
        })

    # fake embedding vector
    embedding = [random.random() for _ in range(128)]

    lung_summary = {
        "volume_cc": round(random.uniform(2000, 4500), 2)
    }

    return nodules, embedding, lung_summary


# ---------------------------
# /predict Endpoint
# ---------------------------
@app.post("/predict")
async def predict(req: PredictRequest):

    # Simulate processing time
    time.sleep(1.5)

    # Load CT image
    ct_array = load_image_from_minio(req.ct_object)

    # Process (fake model)
    nodules, embedding, lung_summary = fake_ct_model(ct_array)

    # Response
    return {
        "job_id": req.job_id,
        "nodules": nodules,
        "lung_mask_summary": lung_summary,
        "ct_embedding": embedding
    }


# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0",
        port=int(os.getenv("PORT", 8101))
    )
