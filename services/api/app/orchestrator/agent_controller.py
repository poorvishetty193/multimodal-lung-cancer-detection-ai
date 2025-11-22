import requests
from core.config import settings
from orchestrator.a2a_protocol import make_message
from core.logger import logger
import time, json

class AgentController:
    """
    Orchestrates modality agents (CT, Audio, Metadata) in parallel,
    waits for results, calls fusion, and returns the final result.
    Note: this starter uses HTTP to call microservices. Replace with gRPC or direct calls as needed.
    """

    def __init__(self):
        self.ct_url = settings.ML_CT_URL
        self.audio_url = settings.ML_AUDIO_URL
        self.meta_url = settings.ML_META_URL
        self.fusion_url = settings.ML_FUSION_URL

    def run_all(self, job_id: str, storage_objects: dict, metadata: dict, job_control=None):
        """
        storage_objects: dict with keys 'ct', 'audio' each containing a presigned URL or object path
        metadata: parsed metadata dict
        job_control: callable that returns dict(job_state, should_pause)
        """
        logger.info(f"AgentController: starting run_all for job {job_id}")
        results = {}

        # Launch parallel HTTP requests (simple approach)
        # CT
        try:
            ct_payload = {"job_id": job_id, "ct_object": storage_objects.get("ct"), "metadata": metadata}
            logger.info("Calling CT service")
            r_ct = requests.post(self.ct_url, json=ct_payload, timeout=300)
            r_ct.raise_for_status()
            results['ct'] = r_ct.json()
        except Exception as e:
            logger.exception("CT service failed")
            results['ct'] = {"error": str(e)}

        # Audio
        try:
            audio_payload = {"job_id": job_id, "audio_object": storage_objects.get("audio"), "metadata": metadata}
            logger.info("Calling Audio service")
            r_audio = requests.post(self.audio_url, json=audio_payload, timeout=120)
            r_audio.raise_for_status()
            results['audio'] = r_audio.json()
        except Exception as e:
            logger.exception("Audio service failed")
            results['audio'] = {"error": str(e)}

        # Metadata model
        try:
            meta_payload = {"job_id": job_id, "metadata": metadata}
            logger.info("Calling Metadata service")
            r_meta = requests.post(self.meta_url, json=meta_payload, timeout=30)
            r_meta.raise_for_status()
            results['metadata'] = r_meta.json()
        except Exception as e:
            logger.exception("Metadata service failed")
            results['metadata'] = {"error": str(e)}

        # Call fusion with gathered results
        try:
            fusion_payload = {"job_id": job_id, "ct": results.get('ct'), "audio": results.get('audio'), "metadata": results.get('metadata')}
            logger.info("Calling Fusion service")
            r_fuse = requests.post(self.fusion_url, json=fusion_payload, timeout=60)
            r_fuse.raise_for_status()
            results['fusion'] = r_fuse.json()
        except Exception as e:
            logger.exception("Fusion service failed")
            results['fusion'] = {"error": str(e)}

        logger.info(f"AgentController: finished run_all for job {job_id}")
        return results
