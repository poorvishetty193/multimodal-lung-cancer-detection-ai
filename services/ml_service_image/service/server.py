from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
from models import load_model
from io import BytesIO
from minio import Minio
import os

app = FastAPI()

# -----------------------------------------
# ENV VARS
# -----------------------------------------
MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", "/app/models/image_classifier.pt")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET = os.getenv("STORAGE_BUCKET", "uploads")

# -----------------------------------------
# INIT MODEL
# -----------------------------------------
CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]

model = load_model(MODEL_PATH, n_classes=len(CLASSES))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------------------
# INIT MINIO CLIENT
# -----------------------------------------
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# -----------------------------------------
# REQUEST MODEL
# -----------------------------------------
class PredictRequest(BaseModel):
    job_id: str
    image_object: str  # example: "uploads/<job>/<file>"
    metadata: dict | None = None


# -----------------------------------------
# ENDPOINT
# -----------------------------------------
@app.post("/predict")
async def predict(req: PredictRequest):

    # image_object = "uploads/<job_id>/<filename>"
    object_key = req.image_object.replace("uploads/", "")

    # -------------------------------------
    # DOWNLOAD FROM MINIO
    # -------------------------------------
    try:
        response = minio_client.get_object(BUCKET, object_key)
        img_bytes = response.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to load image from MinIO: {str(e)}"}

    # -------------------------------------
    # INFERENCE
    # -------------------------------------
    try:
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze()

        probs_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        embedding = probs.tolist()

        return {
            "job_id": req.job_id,
            "image_probs": probs_dict,
            "image_embedding": embedding,
        }

    except Exception as e:
        return {"error": f"Model prediction error: {str(e)}"}


# -----------------------------------------
# LOCAL RUN
# -----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8110)
