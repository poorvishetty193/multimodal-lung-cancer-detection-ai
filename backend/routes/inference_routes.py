from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.inference_model import InferenceRequest, InferenceResponse
from backend.services.inference_service import inference_service
from backend.utils.file_utils import save_upload
import uuid

router = APIRouter()

@router.post("/upload")
def upload_files(ct_zip: UploadFile = File(...), audio_wav: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    ct_path = save_upload(ct_zip, f"uploads/{uid}_ct.zip")
    audio_path = save_upload(audio_wav, f"uploads/{uid}_audio.wav")

    return {"ct_path": ct_path, "audio_path": audio_path}


@router.post("/run", response_model=InferenceResponse)
def run_inference(req: InferenceRequest):
    try:
        result = inference_service.run_full_pipeline(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
