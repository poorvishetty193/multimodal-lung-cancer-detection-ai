from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import json
from storage import upload_bytes, presigned_url
from models.schemas import JobCreateResponse  # keep existing response schema
from orchestrator.agent_controller import AgentController
from core.logger import logger
import redis
import os
import time
from typing import Optional

router = APIRouter()
r = redis.Redis.from_url("redis://redis:6379/1", decode_responses=True)

JOB_TTL = 60 * 60 * 24  # 1 day

# allowed CT mime-type / extensions (keep DICOM/NIfTI/zip plus images)
_CT_EXT_WHITELIST = (".nii", ".nii.gz", ".zip", ".dcm", ".png", ".jpg", ".jpeg")
_CT_MIMES = ("image/png", "image/jpeg", "application/zip", "application/x-nifti", "application/dicom")
_AUDIO_MIMES = ("audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a")

def _looks_like_ct(filename: str, content_type: Optional[str]) -> bool:
    if filename:
        fn = filename.lower()
        for ext in _CT_EXT_WHITELIST:
            if fn.endswith(ext):
                return True
    if content_type:
        if any(m in content_type for m in ("image/", "application/")):
            return True
    return False

def _looks_like_audio(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.startswith("audio/"):
        return True
    if filename:
        fn = filename.lower()
        return fn.endswith((".mp3", ".wav", ".m4a", ".flac", ".aac"))
    return False

@router.post("/submit", response_model=JobCreateResponse)
async def submit_scan(
    file: UploadFile = File(...),
    audio_file: UploadFile = File(None),
    metadata: str = Form(...),
    smoking_pack_years: str = Form("0")
):
    """
    Accepts:
      - file: main uploaded file (CT image .png/.jpg/.jpeg or DICOM/NIfTI/ZIP)
      - audio_file: optional audio (wav/mp3/m4a)
      - metadata: JSON string (other clinical fields)
      - smoking_pack_years: numeric form field (optional)
    Returns job id and queued status.
    """
    job_id = str(uuid4())

    # parse metadata JSON
    try:
        md = json.loads(metadata) if metadata else {}
    except Exception:
        raise HTTPException(status_code=400, detail="metadata must be a valid JSON string")

    # normalize smoking pack-years
    try:
        md["smoking_history_pack_years"] = float(smoking_pack_years or 0)
    except Exception:
        md["smoking_history_pack_years"] = 0.0

    # Validate primary file is CT (or acceptable)
    if not _looks_like_ct(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="Primary file must be a CT (nii/zip/dcm) or an image (png/jpg)")

    # Upload primary file to MinIO
    content = await file.read()
    object_name = f"{job_id}/{file.filename}"
    try:
        upload_bytes("uploads", object_name, content, content_type=file.content_type or "application/octet-stream")
        logger.info(f"Uploaded file to uploads/{object_name}")
    except Exception as e:
        logger.exception("Failed to upload CT file")
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    ct_obj_path = f"uploads/{object_name}"

    # Handle optional audio
    audio_obj_path = None
    if audio_file:
        # check audio looks okay (not mandatory)
        if not _looks_like_audio(audio_file.filename, audio_file.content_type):
            raise HTTPException(status_code=400, detail="audio_file must be a valid audio format (mp3/wav/m4a)")
        audio_content = await audio_file.read()
        audio_name = f"{job_id}/{audio_file.filename}"
        try:
            upload_bytes("uploads", audio_name, audio_content, content_type=audio_file.content_type or "application/octet-stream")
            logger.info(f"Uploaded audio to uploads/{audio_name}")
            audio_obj_path = f"uploads/{audio_name}"
        except Exception as e:
            logger.exception("Failed to upload audio file")
            raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    # build objects mapping exactly how worker expects
    objects = {"ct": ct_obj_path, "audio": audio_obj_path}

    # create job record in Redis
    job_key = f"job:{job_id}"
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "progress": "0.0",
        "objects": json.dumps(objects),
        "metadata": json.dumps(md),
        "results": json.dumps({}),
    }

    # Use hset mapping (hmset is deprecated in some redis libs)
    r.hset(job_key, mapping=job_record)
    r.expire(job_key, JOB_TTL)

    # push to queue for worker
    r.lpush("job_queue", job_id)

    logger.info(f"Created job {job_id} (ct={ct_obj_path}, audio={audio_obj_path})")
    return {"job_id": job_id, "status": "queued"}
