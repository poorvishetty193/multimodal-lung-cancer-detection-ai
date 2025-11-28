# services/ml_service_ct/service/server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from infer import load_model, predict
import torch
import numpy as np
import os
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "checkpoints/model_ep29.pt"
model = load_model(MODEL_PATH)

@app.post("/predict")
async def predict_endpoint(payload: dict):
    # Expect payload {"job_id":..., "ct_object": "<minio path or presigned URL>", "metadata": {...}}
    # For a simple demo, we accept pre-loaded volume. In production, fetch from MinIO URL.
    # Here we expect ct_object to be a local path for testing.
    job_id = payload.get("job_id")
    ct_object = payload.get("ct_object")
    # load local file path (for real deployment fetch via presigned_url)
    import numpy as np, torch
    vol = np.load(ct_object) if ct_object.endswith(".npy") else np.zeros((128,128,128), dtype=np.float32)
    vol = (vol - vol.mean())/(vol.std()+1e-8)
    t = torch.tensor(vol[np.newaxis,...]).float()  # C x D x H x W
    res = predict(model, t)
    res["job_id"] = job_id
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8101)))
