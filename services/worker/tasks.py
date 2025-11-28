"""
Simple worker that polls Redis job_queue list and executes the local worker loop.
"""

import time, os, json, requests
import redis
from loguru import logger
from minio import Minio

# CHANGE HERE ↓↓↓
from orchestrator.agent_controller import AgentController

logger.add("/app/logs/worker.log", rotation="10 MB", retention="7 days")
r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/1"), decode_responses=True)

minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
    secure=False
)

# CHANGE HERE ↓↓↓
agent = AgentController()

JOB_QUEUE = "job_queue"
POLL_INTERVAL = 2

def process_job(job_id):
    key = f"job:{job_id}"
    logger.info(f"Starting job {job_id}")

    r.hset(key, "status", "running")
    r.hset(key, "progress", "0.0")

    data = r.hgetall(key)
    objects = json.loads(data.get("objects", "{}"))
    metadata = json.loads(data.get("metadata", "{}"))

    results = agent.run_all(job_id, objects, metadata, redis_client=r, job_key=key)

    r.hset(key, "results", json.dumps(results))
    r.hset(key, "status", "completed")
    r.hset(key, "progress", "100.0")

    logger.info(f"Completed job {job_id}")

def poll_loop():
    logger.info("Worker poll loop started")
    while True:
        job_id = r.rpop(JOB_QUEUE)
        if job_id:
            key = f"job:{job_id}"

            if not r.exists(key):
                logger.warning(f"Job {job_id} not found")
                continue

            if r.hget(key, "status") == "paused":
                r.lpush(JOB_QUEUE, job_id)
                time.sleep(POLL_INTERVAL)
                continue

            try:
                process_job(job_id)
            except Exception as e:
                logger.exception("Job processing failed")
                r.hset(key, "status", "failed")
                r.hset(key, "results", json.dumps({"error": str(e)}))
        else:
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    poll_loop()
