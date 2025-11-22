from fastapi import APIRouter, HTTPException
from models.schemas import JobStatus
import redis, json
from core.logger import logger

router = APIRouter()
r = redis.Redis.from_url("redis://redis:6379/1", decode_responses=True)

@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    key = f"job:{job_id}"
    if not r.exists(key):
        raise HTTPException(status_code=404, detail="job not found")
    data = r.hgetall(key)
    # parse fields
    status = data.get("status")
    progress = float(data.get("progress", "0.0"))
    results = {}
    try:
        results = json.loads(data.get("results", "{}"))
    except:
        results = {"raw": data.get("results")}
    return {"job_id": job_id, "status": status, "progress": progress, "results": results}

@router.post("/{job_id}/pause")
def pause_job(job_id: str):
    key = f"job:{job_id}"
    if not r.exists(key):
        raise HTTPException(status_code=404, detail="job not found")
    r.hset(key, "status", "paused")
    logger.info(f"Job {job_id} paused")
    return {"job_id": job_id, "status": "paused"}

@router.post("/{job_id}/resume")
def resume_job(job_id: str):
    key = f"job:{job_id}"
    if not r.exists(key):
        raise HTTPException(status_code=404, detail="job not found")
    r.hset(key, "status", "queued")
    # push to queue again
    r.lpush("job_queue", job_id)
    logger.info(f"Job {job_id} resumed")
    return {"job_id": job_id, "status": "queued"}
