import os
import logging
import requests

# ---------------------------------
# Simple local logger
# ---------------------------------
logger = logging.getLogger("worker_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

# ---------------------------------
# Load settings from environment
# ---------------------------------
class Settings:
    ML_CT_URL = os.getenv("ML_CT_URL", "http://ml_ct:8101/predict")
    ML_AUDIO_URL = os.getenv("ML_AUDIO_URL", "http://ml_audio:8102/predict")
    ML_META_URL = os.getenv("ML_META_URL", "http://ml_meta:8103/predict")
    ML_FUSION_URL = os.getenv("ML_FUSION_URL", "http://ml_fusion:8104/predict")

settings = Settings()


class AgentController:
    """
    Worker-side controller.
    Calls each ML microservice and returns results.
    No dependency on API source code.
    """

    def __init__(self):
        self.ct_url = settings.ML_CT_URL
        self.audio_url = settings.ML_AUDIO_URL
        self.meta_url = settings.ML_META_URL
        self.fusion_url = settings.ML_FUSION_URL

    def run_all(self, job_id, objects, metadata, redis_client=None, job_key=None):
        logger.info(f"[Agent] Running job {job_id}")
        results = {}

        # ---- CT ----
        try:
            payload = {"job_id": job_id, "ct_object": objects.get("ct"), "metadata": metadata}
            r = requests.post(self.ct_url, json=payload, timeout=200)
            r.raise_for_status()
            results["ct"] = r.json()
        except Exception as e:
            results["ct"] = {"error": str(e)}

        # ---- Audio (optional) ----
        try:
            if objects.get("audio"):
                payload = {"job_id": job_id, "audio_object": objects.get("audio")}
                r = requests.post(self.audio_url, json=payload, timeout=120)
                r.raise_for_status()
                results["audio"] = r.json()
            else:
                results["audio"] = {"note": "audio not provided"}
        except Exception as e:
            results["audio"] = {"error": str(e)}

        # ---- Metadata ----
        try:
            payload = {"job_id": job_id, "metadata": metadata}
            r = requests.post(self.meta_url, json=payload, timeout=30)
            r.raise_for_status()
            results["metadata"] = r.json()
        except Exception as e:
            results["metadata"] = {"error": str(e)}

        # ---- Fusion ----
        try:
            payload = {
                "job_id": job_id,
                "ct": results.get("ct"),
                "audio": results.get("audio"),
                "metadata": results.get("metadata"),
            }
            r = requests.post(self.fusion_url, json=payload, timeout=120)
            r.raise_for_status()
            results["fusion"] = r.json()
        except Exception as e:
            results["fusion"] = {"error": str(e)}

        logger.info(f"[Agent] Completed job {job_id}")
        return results
