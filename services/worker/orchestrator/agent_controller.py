import os
import sys
import requests

# === FIX FOR PYLANCE IMPORT ERRORS ===
API_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../api/app"))
if API_PATH not in sys.path:
    sys.path.append(API_PATH)

from core.logger import logger
from core.config import settings


class AgentController:

    def __init__(self):
        self.ct_url = settings.ML_CT_URL
        self.audio_url = settings.ML_AUDIO_URL
        self.meta_url = settings.ML_META_URL
        self.fusion_url = settings.ML_FUSION_URL

    def run_all(self, job_id, objects, metadata, redis_client=None, job_key=None):
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
                "ct": results["ct"],
                "audio": results["audio"],
                "metadata": results["metadata"],
            }
            r = requests.post(self.fusion_url, json=payload, timeout=120)
            r.raise_for_status()
            results["fusion"] = r.json()
        except Exception as e:
            results["fusion"] = {"error": str(e)}

        logger.info(f"AgentController finished job {job_id}")
        return results
