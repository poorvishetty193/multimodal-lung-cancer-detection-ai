import os
import logging
import requests

logger = logging.getLogger("worker_agent")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class Settings:
    ML_CT_URL = os.getenv("ML_CT_URL", "http://ml_ct:8101/predict")
    # use 8110 as the image service default (adjust if your container uses different port)
    ML_IMAGE_URL = os.getenv("ML_IMAGE_URL", "http://ml_image:8110/predict")
    ML_AUDIO_URL = os.getenv("ML_AUDIO_URL", "http://ml_audio:8102/predict")
    ML_META_URL = os.getenv("ML_META_URL", "http://ml_meta:8103/predict")
    ML_FUSION_URL = os.getenv("ML_FUSION_URL", "http://ml_fusion:8104/predict")

settings = Settings()

class AgentController:
    def __init__(self):
        self.ct_url = settings.ML_CT_URL
        self.image_url = settings.ML_IMAGE_URL
        self.audio_url = settings.ML_AUDIO_URL
        self.meta_url = settings.ML_META_URL
        self.fusion_url = settings.ML_FUSION_URL

    def run_all(self, job_id, objects, metadata, redis_client=None, job_key=None):
        """
        objects: dict that may contain keys 'ct', 'image', 'audio' each with MinIO path like "uploads/<job>/<file>"
        metadata: dict
        returns dict with keys 'ct','image','audio','metadata','fusion' (or error entries)
        """
        results = {}

        # ---------------- CT pipeline (if provided) ----------------
        if objects.get("ct"):
            try:
                payload = {"job_id": job_id, "ct_object": objects["ct"], "metadata": metadata}
                r = requests.post(self.ct_url, json=payload, timeout=200)
                r.raise_for_status()
                results["ct"] = r.json()
            except Exception as e:
                logger.exception("CT service call failed")
                results["ct"] = {"error": str(e)}
        else:
            results["ct"] = {"note": "no CT provided"}

        # ---------------- Image pipeline (if provided) ----------------
        if objects.get("image"):
            try:
                # we send image_object (MinIO path) and metadata — image service should fetch from MinIO
                payload = {"job_id": job_id, "image_object": objects["image"], "metadata": metadata}
                r = requests.post(self.image_url, json=payload, timeout=200)
                r.raise_for_status()
                results["image"] = r.json()
            except Exception as e:
                logger.exception("Image service call failed")
                results["image"] = {"error": str(e)}
        else:
            results["image"] = {"note": "no image provided"}

        # ---------------- Audio ----------------
        if objects.get("audio"):
            try:
                payload = {"job_id": job_id, "audio_object": objects["audio"], "metadata": metadata}
                r = requests.post(self.audio_url, json=payload, timeout=120)
                r.raise_for_status()
                results["audio"] = r.json()
            except Exception as e:
                logger.exception("Audio service call failed")
                results["audio"] = {"error": str(e)}
        else:
            results["audio"] = {"note": "audio not provided"}

        # ---------------- Metadata ----------------
        try:
            payload = {"job_id": job_id, "metadata": metadata}
            r = requests.post(self.meta_url, json=payload, timeout=30)
            r.raise_for_status()
            results["metadata"] = r.json()
        except Exception as e:
            logger.exception("Metadata service call failed")
            results["metadata"] = {"error": str(e)}

        # ---------------- Fusion ----------------
        try:
            payload = {
                "job_id": job_id,
                "ct": results.get("ct"),
                "image": results.get("image"),
                "audio": results.get("audio"),
                "metadata": results.get("metadata"),
            }
            r = requests.post(self.fusion_url, json=payload, timeout=120)
            r.raise_for_status()
            results["fusion"] = r.json()
        except Exception as e:
            logger.exception("Fusion service call failed")
            results["fusion"] = {"error": str(e)}

        return results
