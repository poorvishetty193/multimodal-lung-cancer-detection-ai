"""
Stubbed AgentController for worker that calls ML microservices via HTTP.
This mirrors services/api/app/orchestrator/agent_controller.py but adapted for the worker.
"""
import os, requests, json, time
from loguru import logger

class AgentControllerStub:
    def __init__(self):
        self.ct_url = os.getenv("ML_CT_URL", "http://ml_ct:8101/predict")
        self.audio_url = os.getenv("ML_AUDIO_URL", "http://ml_audio:8102/predict")
        self.meta_url = os.getenv("ML_META_URL", "http://ml_meta:8103/predict")
        self.fusion_url = os.getenv("ML_FUSION_URL", "http://ml_fusion:8104/predict")

    def run_all(self, job_id, storage_objects, metadata, redis_client=None, job_key=None):
        results = {}
        # update progress
        if redis_client and job_key:
            redis_client.hset(job_key, "progress", "5.0")

        try:
            r_ct = requests.post(self.ct_url, json={"job_id": job_id, "ct_object": storage_objects.get("ct"), "metadata": metadata}, timeout=300)
            results['ct'] = r_ct.json()
        except Exception as e:
            logger.exception("CT failed")
            results['ct'] = {"error": str(e)}
        if redis_client and job_key:
            redis_client.hset(job_key, "progress", "35.0")

        try:
            r_audio = requests.post(self.audio_url, json={"job_id": job_id, "audio_object": storage_objects.get("audio"), "metadata": metadata}, timeout=120)
            results['audio'] = r_audio.json()
        except Exception as e:
            logger.exception("Audio failed")
            results['audio'] = {"error": str(e)}
        if redis_client and job_key:
            redis_client.hset(job_key, "progress", "65.0")

        try:
            r_meta = requests.post(self.meta_url, json={"job_id": job_id, "metadata": metadata}, timeout=30)
            results['metadata'] = r_meta.json()
        except Exception as e:
            logger.exception("Meta failed")
            results['metadata'] = {"error": str(e)}
        if redis_client and job_key:
            redis_client.hset(job_key, "progress", "80.0")

        # Fusion
        try:
            r_fuse = requests.post(self.fusion_url, json={"job_id": job_id, "ct": results.get("ct"), "audio": results.get("audio"), "metadata": results.get("metadata")}, timeout=60)
            results['fusion'] = r_fuse.json()
        except Exception as e:
            logger.exception("Fusion failed")
            results['fusion'] = {"error": str(e)}
        if redis_client and job_key:
            redis_client.hset(job_key, "progress", "95.0")

        return results
