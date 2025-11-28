from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from infer import load_model, predict_image

app = FastAPI()

model = None


class PredictRequest(BaseModel):
    job_id: str
    image_object: str  # MinIO path


@app.on_event("startup")
def startup():
    global model
    if os.path.exists("image_model.pth"):
        model = load_model()
        print("Loaded trained PNG/JPG model.")
    else:
        print("⚠ No trained model found. Using random weights.")


def download_from_minio(path: str):
    """
    MinIO object path: uploads/job_id/filename.png
    This downloads into /tmp inside container.
    """
    minio_url = os.getenv("MINIO_ENDPOINT", "minio:9000")
    bucket = "uploads"
    local_path = f"/tmp/{os.path.basename(path)}"

    url = f"http://{minio_url}/{bucket}/{path.replace('uploads/', '')}"

    r = requests.get(url)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(r.content)

    return local_path


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        local = download_from_minio(req.image_object)
        result = predict_image(local, model=model)

        return {
            "job_id": req.job_id,
            "image_result": result
        }
    except Exception as e:
        return {"error": str(e)}
