# multi-agent-system/agents/fusion_agent.py
import logging
from typing import Dict, Any

from ml_service.fusion_model.fusion_inference import fusion_full_inference

logger = logging.getLogger(__name__)


class FusionAgent:
    """
    Fusion-Agent: receives embeddings from CT/Audio/Metadata and returns final fusion score.
    """

    def __init__(self, device: str = "cpu", fusion_weights=None, metadata_weights=None):
        self.device = device
        self.fusion_weights = fusion_weights
        self.metadata_weights = metadata_weights

    def run(self, ct_emb, audio_emb, metadata_raw) -> Dict[str, Any]:
        # ct_emb: numpy array (256)
        # audio_emb: numpy array (256)
        # metadata_raw: dict
        logger.info("FusionAgent: running fusion inference")
        score = fusion_full_inference(
            ct_emb=ct_emb,
            audio_emb=audio_emb,
            metadata_raw=metadata_raw,
            fusion_weights=self.fusion_weights,
            metadata_weights=self.metadata_weights,
            device=self.device
        )
        return {
            "fusion_score": float(score),
            "provenance": {"source": "fusion_agent"}
        }
