from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import json
from storage import upload_bytes
from models.schemas import JobCreateResponse
from core.logger import logger
import redis
import os
from typing import Optional

router = APIRouter()

r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/1"), decode_responses=True)

JOB_TTL = 60 * 60 * 24  # 1 day

# Allowed primary file types (CT + image)
_CT_EXT = (".nii", ".nii.gz", ".zip", ".dcm", ".png", ".jpg", ".jpeg")
_AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac")

def is_ct_file(filename: str, content_type: Optional[str]) -> bool:
    if not filename:
        return False
    fname = filename.lower()
    if fname.endswith(_CT_EXT):
        return True
    if content_type and (
        content_type.startswith("image/")
        or "dicom" in content_type.lower()
        or "zip" in content_type.lower()
    ):
        return True
    return False

def is_audio_file(filename: str, content_type: Optional[str]) -> bool:
    if not filename:
        return False
    fname = filename.lower()
    if fname.endswith(_AUDIO_EXT):
        return True
    if content_type and content_type.startswith("audio/"):
        return True
    return False

@router.post("/submit", response_model=JobCreateResponse)
async def submit_scan(
    file: UploadFile = File(...),
    audio_file: UploadFile = File(None),
    metadata: str = Form("{}"),
    smoking_pack_years: str = Form("0"),
):
    """
    Accepts:
      - file: required CT/ZIP/DICOM/PNG/JPG
      - audio_file: optional
      - metadata: JSON
      - smoking_pack_years: optional numeric field
    """
    job_id = str(uuid4())

    # ---- Parse metadata JSON ----
    try:
        md = json.loads(metadata) if metadata else {}
    except Exception:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON")

    # ---- Add smoking pack-years to metadata ----
    try:
        md["smoking_history_pack_years"] = float(smoking_pack_years or 0)
    except Exception:
        md["smoking_history_pack_years"] = 0.0

    # ---- Validate primary file type ----
    if not is_ct_file(file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Primary file must be CT (.nii/.nii.gz/.zip/.dcm) or image (.png/.jpg/.jpeg)",
        )

    # ---- Upload primary CT/Image file ----
    try:
        content = await file.read()
        object_name = f"{job_id}/{file.filename}"
        upload_bytes("uploads", object_name, content, content_type=file.content_type or "application/octet-stream")
        logger.info(f"Uploaded primary file to uploads/{object_name}")
    except Exception as e:
        logger.exception("Primary file upload error")
        raise HTTPException(status_code=500, detail="Failed to upload primary file")

    ct_path = f"uploads/{object_name}"

    audio_path = None
    if audio_file:
        if not is_audio_file(audio_file.filename, audio_file.content_type):
            raise HTTPException(status_code=400, detail="audio_file must be mp3/wav/m4a etc")
        try:
            audio_content = await audio_file.read()
            audio_object = f"{job_id}/{audio_file.filename}"
            upload_bytes("uploads", audio_object, audio_content, content_type=audio_file.content_type or "audio/wav")
            audio_path = f"uploads/{audio_object}"
            logger.info(f"Uploaded audio to uploads/{audio_object}")
        except Exception as e:
            logger.exception("Optional audio upload failed")
            raise HTTPException(status_code=500, detail="Failed to upload audio")

    # Determine whether the primary file is an image or CT
    filename_l = file.filename.lower()
    is_image = filename_l.endswith((".png", ".jpg", ".jpeg"))
    is_ct = filename_l.endswith((".nii", ".nii.gz", ".zip", ".dcm"))

    objects = {
        "ct": ct_path if is_ct else None,
        "image": ct_path if is_image else None,
        "audio": audio_path,
    }

    job_key = f"job:{job_id}"
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "progress": "0.0",
        "objects": json.dumps(objects),
        "metadata": json.dumps(md),
        "results": json.dumps({}),
    }

    r.hset(job_key, mapping=job_record)
    r.expire(job_key, JOB_TTL)
    r.lpush("job_queue", job_id)

    logger.info(f"Job created: {job_id} (objects: {objects})")
    return {"job_id": job_id, "status": "queued"}
