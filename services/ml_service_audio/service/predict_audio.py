from fastapi import FastAPI
from pydantic import BaseModel
import random
import time

app = FastAPI()

class PredictRequest(BaseModel):
    job_id: str

@app.post("/predict")
async def predict(req: PredictRequest):
    time.sleep(1)

    probs = {
        "adenocarcinoma": 0.3,
        "squamous": 0.1,
        "large_cell": 0.05,
        "small_cell": 0.05,
        "anomaly_score": 0.4
    }

    return {
        "job_id": req.job_id,
        "audio_probs": probs,
        "audio_embedding": [random.random() for _ in range(64)]
    }

# Uvicorn server start when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
