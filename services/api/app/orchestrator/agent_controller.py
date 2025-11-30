import requests
from core.config import settings
from core.logger import logger


class AgentController:
    """
    Orchestrates all modality services (CT / Image, Audio, Metadata)
    and then calls the Fusion service.

    Works with:
    - Optional audio
    - PNG/JPG/JPEG images
    - DICOM ZIP / CT ZIP
    - Metadata (age, pack-years, symptoms)
    """

    def __init__(self):
        self.ct_url = settings.ML_CT_URL
        self.audio_url = settings.ML_AUDIO_URL
        self.meta_url = settings.ML_META_URL
        self.fusion_url = settings.ML_FUSION_URL

    def _call(self, url, payload, timeout, label):
        """Internal helper for clean request/exception handling."""
        try:
            logger.info(f"[AgentController] Calling {label} service → {url}")
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.exception(f"{label} service failed")
            return {"error": str(e)}

    def run_all(self, job_id: str, objects: dict, metadata: dict, job_control=None):
        """
        objects = {
            "ct": "uploads/jobid/file",
            "audio": "uploads/jobid/file" OR None
        }
        metadata = parsed dict from the upload
        """

        logger.info(f"[AgentController] Starting job {job_id}")
        results = {}

        # ---------------------------------------------------------
        #   CT / IMAGE MODEL  (mandatory)
        # ---------------------------------------------------------
        ct_payload = {
            "job_id": job_id,
            "ct_object": objects.get("ct"),
            "metadata": metadata
        }
        results["ct"] = self._call(
            self.ct_url,
            ct_payload,
            timeout=200,
            label="CT/Image"
        )

        # ---------------------------------------------------------
        #   AUDIO MODEL (optional)
        # ---------------------------------------------------------
        audio_obj = objects.get("audio")
        if audio_obj:
            audio_payload = {
                "job_id": job_id,
                "audio_object": audio_obj
            }
            results["audio"] = self._call(
                self.audio_url,
                audio_payload,
                timeout=120,
                label="Audio"
            )
        else:
            logger.info("[AgentController] No audio provided for this job.")
            results["audio"] = {"note": "Audio file not provided"}

        # ---------------------------------------------------------
        #   METADATA MODEL
        # ---------------------------------------------------------
        meta_payload = {
            "job_id": job_id,
            "metadata": metadata
        }
        results["metadata"] = self._call(
            self.meta_url,
            meta_payload,
            timeout=40,
            label="Metadata"
        )

        # ---------------------------------------------------------
        #   FUSION MODEL
        # ---------------------------------------------------------
        fusion_payload = {
            "job_id": job_id,
            "ct": results["ct"],
            "audio": results["audio"],
            "metadata": results["metadata"]
        }
        results["fusion"] = self._call(
            self.fusion_url,
            fusion_payload,
            timeout=120,
            label="Fusion"
        )

        logger.info(f"[AgentController] Completed job {job_id}")
        return results
