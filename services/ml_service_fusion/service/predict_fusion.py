from fastapi import FastAPI
from pydantic import BaseModel
import time
import random

app = FastAPI()

class CTModel(BaseModel):
    nodules: list | None = None

class AudioModel(BaseModel):
    audio_probs: dict | None = None

class MetadataModel(BaseModel):
    metadata_probs: dict | None = None

class PredictRequest(BaseModel):
    job_id: str
    ct: dict | None = None
    audio: dict | None = None
    metadata: dict | None = None


@app.post("/predict")
async def predict(req: PredictRequest):

    def safe_get(p, k, default=0.0):
        return p.get(k, default) if isinstance(p, dict) else default

    cancer_types = ["adenocarcinoma", "squamous", "large_cell", "small_cell"]
    fused = {}

    ct = req.ct or {}
    audio = req.audio or {}
    metadata = req.metadata or {}

    # Fusion calculation
    for c in cancer_types:
        vals = []

        # CT nodules → pick max probability per cancer type
        nodules = ct.get("nodules", []) if isinstance(ct, dict) else []
        if nodules:
            vals.append(max([n.get("nodule_probs", {}).get(c, 0) for n in nodules]))

        # Audio
        vals.append(safe_get(audio.get("audio_probs", {}), c, 0))

        # Metadata
        vals.append(safe_get(metadata.get("metadata_probs", {}), c, 0))

        fused[c] = sum(vals) / max(1, len(vals))

    # normalize
    total = sum(fused.values()) or 1.0
    for k in fused:
        fused[k] = fused[k] / total

    risk_score = max(fused.values())

    time.sleep(0.2)

    return {
        "job_id": req.job_id,
        "final_probs": fused,
        "risk_score": risk_score,
        "fusion_embedding": [random.random() for _ in range(32)]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8104)
