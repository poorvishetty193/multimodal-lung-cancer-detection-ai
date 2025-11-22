from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

class MetadataModel(BaseModel):
    age: int | None = 50
    smoking_history_pack_years: float | None = 0.0

class PredictRequest(BaseModel):
    job_id: str
    metadata: MetadataModel | None = None

@app.post("/predict")
async def predict(req: PredictRequest):
    metadata = req.metadata or MetadataModel()

    age = metadata.age or 50
    pack_years = metadata.smoking_history_pack_years or 0

    # Simple rule-based baseline
    base = 0.01 + max(0, (age - 40) * 0.005) + min(0.3, pack_years * 0.002)

    probs = {
        "adenocarcinoma": base * 0.5,
        "squamous": base * 0.3,
        "large_cell": base * 0.1,
        "small_cell": base * 0.1,
    }

    time.sleep(0.3)

    return {
        "job_id": req.job_id,
        "metadata_probs": probs,
        "meta_embedding": [base] * 16,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8103)
