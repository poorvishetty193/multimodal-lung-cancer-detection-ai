from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.inference_service import InferenceService

router = APIRouter()

# -------------------------
# 📌 1) Pydantic request model
# -------------------------
class InferenceRequest(BaseModel):
    ct_score: float
    audio_score: float
    age: int
    sex: str


# -------------------------
# 📌 2) Initialize service
# -------------------------
inference_service = InferenceService()


# -------------------------
# 📌 3) POST /inference endpoint
# -------------------------
@router.post("/inference")
async def run_inference(request: InferenceRequest):
    try:
        result = inference_service.run_inference(
            ct_score=request.ct_score,
            audio_score=request.audio_score,
            metadata={"age": request.age, "sex": request.sex}
        )
        return {"status": "success", "report": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
