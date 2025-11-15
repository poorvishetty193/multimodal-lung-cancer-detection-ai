# multi-agent-system/agents/ct_agent.py
import logging
from typing import Dict, Any

from ml_service.ct_pipeline.ct_inference import ct_full_inference

logger = logging.getLogger(__name__)


class CTAgent:
    """
    CT-Agent: runs CT pipeline and returns CT embedding + segmentation + score.
    """

    def __init__(self, device: str = "cpu", unet_weights=None, classifier_weights=None):
        self.device = device
        self.unet_weights = unet_weights
        self.classifier_weights = classifier_weights

    def run(self, ct_path: str, is_dicom: bool = True) -> Dict[str, Any]:
        """
        Input: path to DICOM folder or NIfTI
        Output:
            {
              "ct_embedding": np.array (256)  # if using classifier internals; otherwise classifier probability boxed
              "ct_score": float,
              "mask": ndarray,
              "preprocessed": ndarray,
              "provenance": {...}
            }
        """
        logger.info("CTAgent: running full CT inference for %s", ct_path)
        res = ct_full_inference(
            path=ct_path,
            is_dicom=is_dicom,
            unet_weights=self.unet_weights,
            classifier_weights=self.classifier_weights,
            device=self.device
        )

        # For this agent we treat classifier output as ct_score and create a placeholder embedding
        # If you later extract internal embedding from classifier, replace this with real vector.
        ct_score = res.get("cancer_risk", 0.0)

        # Create a simple embedding by replicating the score (placeholder) — replace with real embedding later
        import numpy as np
        ct_embedding = np.full((256,), float(ct_score), dtype=float)

        return {
            "ct_score": float(ct_score),
            "ct_embedding": ct_embedding,
            "mask": res.get("lung_mask_pred"),
            "preprocessed": res.get("preprocessed_ct"),
            "provenance": {"source": "ct_agent", "path": ct_path}
        }
