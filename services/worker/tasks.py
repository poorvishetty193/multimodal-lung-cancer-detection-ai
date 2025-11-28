"""
Simple worker that polls Redis job_queue list and executes the local worker loop.
"""

import time
import os
import json
import redis
from loguru import logger
from minio import Minio

# -------------------------------------------------------
# FIX: Import AgentController from worker-local folder
# -------------------------------------------------------
from orchestrator.agent_controller import AgentController

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logger.add("/app/logs/worker.log", rotation="10 MB", retention="7 days")

# -------------------------------------------------------
# Redis
# -------------------------------------------------------
r = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379/1"), 
    decode_responses=True
)

# -------------------------------------------------------
# MinIO (not used directly in worker yet but kept)
# -------------------------------------------------------
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minio"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
    secure=False
)

# -------------------------------------------------------
# Agent — calls ML services over HTTP
# -------------------------------------------------------
agent = AgentController()

JOB_QUEUE = "job_queue"
POLL_INTERVAL = 2


def process_job(job_id: str):
    """Executes a single job by calling AgentController."""
    key = f"job:{job_id}"

    logger.info(f"➡️  Worker: Starting job {job_id}")

    # Mark as running
    r.hset(key, mapping={
        "status": "running",
        "progress": "0.0"
    })

    # Load job data
    data = r.hgetall(key)
    objects = json.loads(data.get("objects", "{}"))
    metadata = json.loads(data.get("metadata", "{}"))

    # Execute orchestrator (runs CT, Audio, Meta, Fusion)
    try:
        results = agent.run_all(job_id, objects, metadata, redis_client=r, job_key=key)

    except Exception as e:
        logger.exception("❌ Error in AgentController.run_all")
        r.hset(key, "status", "failed")
        r.hset(key, "results", json.dumps({"error": str(e)}))
        return

    # Save results
    r.hset(key, mapping={
        "status": "completed",
        "progress": "100.0",
        "results": json.dumps(results)
    })

    logger.info(f"✅ Worker: Completed job {job_id}")


def poll_loop():
    """Main polling loop of the worker."""

    logger.info("🔄 Worker poll loop started...")

    while True:
        job_id = r.rpop(JOB_QUEUE)

        if job_id:
            key = f"job:{job_id}"

            # Validate job exists
            if not r.exists(key):
                logger.warning(f"⚠️ Job {job_id} does not exist")
                continue

            # Skip paused
            if r.hget(key, "status") == "paused":
                logger.info(f"⏸ Job {job_id} paused — requeueing")
                r.lpush(JOB_QUEUE, job_id)
                time.sleep(POLL_INTERVAL)
                continue

            # Process the job
            process_job(job_id)

        else:
            # No job available
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()
