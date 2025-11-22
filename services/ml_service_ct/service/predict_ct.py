from fastapi import FastAPI
from pydantic import BaseModel
import random
import time

app = FastAPI()

class PredictRequest(BaseModel):
    job_id: str

@app.post("/predict")
async def predict(req: PredictRequest):
    # simulate CT pipeline
    time.sleep(2)

    nodules = [
        {
            "x": 50,
            "y": 60,
            "z": 30,
            "diameter_mm": 8,
            "confidence": 0.87,
            "nodule_probs": {
                "adenocarcinoma": 0.6,
                "squamous": 0.2,
                "large_cell": 0.1,
                "small_cell": 0.1
            }
        },
        {
            "x": 120,
            "y": 80,
            "z": 45,
            "diameter_mm": 12,
            "confidence": 0.92,
            "nodule_probs": {
                "adenocarcinoma": 0.2,
                "squamous": 0.7,
                "large_cell": 0.05,
                "small_cell": 0.05
            }
        }
    ]

    return {
        "job_id": req.job_id,
        "nodules": nodules,
        "lung_mask_summary": {"volume_cc": 2500},
        "ct_embedding": [random.random() for _ in range(128)]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
