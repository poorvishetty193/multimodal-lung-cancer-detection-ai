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
    if isinstance(container, dict):
        return container.get(key, default)
    return default

def max_prob_from_nodules(nodules, cancer_type):
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

def normalize_image_probs(raw_probs: dict) -> dict:
    """
    Accept various image model class names and map them to canonical fusion keys:
      - adenocarcinoma -> adenocarcinoma
      - squamous / squamous.cell.carcinoma -> squamous
      - large.cell.carcinoma / large cell carcinoma -> large_cell
      - small.cell.carcinoma / small cell carcinoma -> small_cell
    Any unknown keys are ignored.
    """
    mapping = {}
    for k, v in (raw_probs or {}).items():
        key = k.lower().strip()
        key = key.replace(".", " ").replace("_", " ")
        # normalize words
        if "adeno" in key:
            mapping.setdefault("adenocarcinoma", 0.0)
            mapping["adenocarcinoma"] = max(mapping["adenocarcinoma"], float(v))
        elif "squamous" in key:
            mapping.setdefault("squamous", 0.0)
            mapping["squamous"] = max(mapping["squamous"], float(v))
        elif "large" in key and "cell" in key:
            mapping.setdefault("large_cell", 0.0)
            mapping["large_cell"] = max(mapping["large_cell"], float(v))
        elif "small" in key and "cell" in key:
            mapping.setdefault("small_cell", 0.0)
            mapping["small_cell"] = max(mapping["small_cell"], float(v))
        # else ignore unknown
    return mapping

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

    # attempt to normalize image probabilities
    image_probs_raw = image.get("image_probs", {}) if isinstance(image, dict) else {}
    image_probs = normalize_image_probs(image_probs_raw)

    # -----------------------------------------
    # FUSION — combine all 4 modalities
    # -----------------------------------------
    for c in cancer_types:
        votes = []

        # ---- CT ----
        if "nodules" in ct:
            votes.append(max_prob_from_nodules(ct.get("nodules", []), c))

        # ---- Audio ----
        audio_probs = audio.get("audio_probs", {}) if isinstance(audio, dict) else {}
        votes.append(safe_prob(audio_probs, c, 0))

        # ---- Metadata ----
        metadata_probs = metadata.get("metadata_probs", {}) if isinstance(metadata, dict) else {}
        votes.append(safe_prob(metadata_probs, c, 0))

        # ---- Image (normalized) ----
        votes.append(safe_prob(image_probs, c, 0))

        # average fusion (use equal weighting)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8104)
