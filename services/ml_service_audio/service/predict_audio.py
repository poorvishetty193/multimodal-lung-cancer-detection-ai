# predict_audio.py
from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid
import os
from infer_audio import predict_audio

app = FastAPI()

class PredictRequest(BaseModel):
    job_id: str
    audio_object: str  # MinIO path

MINIO_ENDPOINT = "http://minio:9000"
MINIO_BUCKET = "uploads"

# For downloading from MinIO
from minio import Minio
minio_client = Minio(
    "minio:9000",
    access_key=os.getenv("MINIO_ROOT_USER", "minio"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minio123"),
    secure=False
)

def download_from_minio(obj_path):
    local = f"/tmp/{uuid.uuid4()}.wav"
    bucket, key = obj_path.split("/", 1)
    minio_client.fget_object(bucket, key, local)
    return local

@app.post("/predict")
async def predict(req: PredictRequest):
    # Download audio from MinIO
    audio_local = download_from_minio(req.audio_object)

    # Run ML inference
    result = predict_audio(audio_local)

    return {
        "job_id": req.job_id,
        "audio_prediction": result["predicted_label"],
        "audio_probabilities": result["probabilities"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
