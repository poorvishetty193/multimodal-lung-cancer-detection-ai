from fastapi import FastAPI
from pydantic import BaseModel
import random
import time

app = FastAPI()

# ---------------------------------------------
# Pydantic Models
# ---------------------------------------------
class PredictRequest(BaseModel):
    job_id: str
    ct: dict | None = None
    audio: dict | None = None
    metadata: dict | None = None
    image: dict | None = None   # NEW IMAGE MODALITY


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def safe_prob(container, key, default=0.0):
    """Return probability or default."""
    if isinstance(container, dict):
        return container.get(key, default)
    return default


def max_prob_from_nodules(nodules, cancer_type):
    """Return the highest probability for a cancer type among CT nodules."""
    if not isinstance(nodules, list):
        return 0.0
    best = 0.0
    for n in nodules:
        if not isinstance(n, dict):
            continue
        p = n.get("nodule_probs", {}).get(cancer_type, 0.0)
        if p > best:
            best = p
    return best


# ---------------------------------------------
# Fusion Endpoint
# ---------------------------------------------
@app.post("/predict")
async def predict(req: PredictRequest):

    cancer_types = ["adenocarcinoma", "squamous", "large_cell", "small_cell"]
    fused = {}

    ct = req.ct or {}
    audio = req.audio or {}
    metadata = req.metadata or {}
    image = req.image or {}  # NEW

    # -----------------------------------------
    # FUSION — combine all 4 modalities
    # -----------------------------------------
    for c in cancer_types:
        votes = []

        # ---- CT ----
        if "nodules" in ct:
            votes.append(max_prob_from_nodules(ct.get("nodules", []), c))

        # ---- Audio ----
        audio_probs = audio.get("audio_probs", {})
        votes.append(safe_prob(audio_probs, c, 0))

        # ---- Metadata ----
        metadata_probs = metadata.get("metadata_probs", {})
        votes.append(safe_prob(metadata_probs, c, 0))

        # ---- Image (NEW) ----
        image_probs = image.get("image_probs", {})
        votes.append(safe_prob(image_probs, c, 0))

        # average fusion
        fused[c] = sum(votes) / max(len(votes), 1)

    # -----------------------------------------
    # NORMALIZATION
    # -----------------------------------------
    total = sum(fused.values()) or 1.0
    fused = {k: v / total for k, v in fused.items()}

    # Risk score = highest probability
    risk_score = max(fused.values())

    # -----------------------------------------
    # RESPONSE
    # -----------------------------------------
    return {
        "job_id": req.job_id,
        "final_probs": fused,
        "risk_score": risk_score,
        "fusion_embedding": [random.random() for _ in range(32)],
    }


# ---------------------------------------------
# LOCAL RUNNER
# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8104)
