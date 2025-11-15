# multi-agent-system/agents/audio_agent.py
import logging
from typing import Dict, Any

from ml_service.audio_pipeline.audio_inference import audio_full_inference

logger = logging.getLogger(__name__)


class AudioAgent:
    """
    Audio-Agent: runs the audio inference pipeline and returns score + embedding
    """

    def __init__(self, device: str = "cpu", weight_path=None):
        self.device = device
        self.weight_path = weight_path

    def run(self, audio_path: str) -> Dict[str, Any]:
        logger.info("AudioAgent: running audio inference for %s", audio_path)
        res = audio_full_inference(path=audio_path, weight_path=self.weight_path, device=self.device)
        return {
            "audio_score": float(res.get("audio_score", 0.0)),
            "audio_embedding": res.get("audio_embedding"),
            "log_mel": res.get("log_mel"),
            "provenance": {"source": "audio_agent", "path": audio_path}
        }
