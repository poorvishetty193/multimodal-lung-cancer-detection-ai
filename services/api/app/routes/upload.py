from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from uuid import uuid4
import json
from storage import upload_bytes, presigned_url
from models.schemas import JobCreateResponse, MetadataModel
from orchestrator.agent_controller import AgentController
from core.logger import logger
import os
import base64
import redis
import time
from typing import Optional
import requests

router = APIRouter()
r = redis.Redis.from_url("redis://redis:6379/1", decode_responses=True)

# For job queueing we use Celery tasks via HTTP forward to worker's task entrypoint (the worker will poll rabbitmq).
# But to keep this service independent, we'll write job metadata to Redis and then call the Celery task via HTTP (worker will pick up).
# NOTE: In this starter the actual Celery enqueuing is performed by the worker container polling Redis keys.
# We'll set job state here.
JOB_TTL = 60 * 60 * 24  # 1 day

@router.post("/submit", response_model=JobCreateResponse)
async def submit_scan(
    file: UploadFile = File(...),
    metadata: str = Form(...),
):
    """
    Accepts a single file (DICOM, NIfTI, ZIP of DICOMs, audio), plus metadata JSON string.
    Returns a job id and queued status.
    """
    job_id = str(uuid4())
    try:
        md = json.loads(metadata)
    except Exception:
        raise HTTPException(status_code=400, detail="metadata must be a valid JSON string")

    content = await file.read()
    object_name = f"{job_id}/{file.filename}"
    # save to MinIO
    upload_bytes("uploads", object_name, content, content_type=file.content_type or "application/octet-stream")
    logger.info(f"Uploaded file to uploads/{object_name}")

    # create job record in Redis
    job_key = f"job:{job_id}"
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "progress": "0.0",
        "objects": json.dumps({"ct": f"uploads/{object_name}" if file.filename.lower().endswith(('.nii','.nii.gz','.zip','.dcm')) else None,
                                "audio": f"uploads/{object_name}" if file.content_type and "audio" in file.content_type else None}),
        "metadata": json.dumps(md),
        "results": json.dumps({}),
    }
    r.hmset(job_key, job_record)
    r.expire(job_key, JOB_TTL)

    # place a simple signal key so worker picks it up
    r.lpush("job_queue", job_id)

    return {"job_id": job_id, "status": "queued"}
