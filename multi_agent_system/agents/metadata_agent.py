# multi-agent-system/agents/metadata_agent.py
import logging
from typing import Dict, Any

from ml_service.metadata_pipeline.preprocess_metadata import preprocess_metadata
from ml_service.metadata_pipeline.metadata_model import MetadataMLP
import torch

logger = logging.getLogger(__name__)


class MetadataAgent:
    """
    Metadata-Agent: preprocess metadata and return metadata embedding
    """

    def __init__(self, device: str = "cpu", weight_path=None):
        self.device = device
        # lazy load encoder
        self.encoder = None
        self.weight_path = weight_path

    def _load_encoder(self, input_dim=13, embed_dim=32):
        if self.encoder is None:
            self.encoder = MetadataMLP(input_dim=input_dim, embed_dim=embed_dim).to(self.device)
            if self.weight_path:
                self.encoder.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            self.encoder.eval()

    def run(self, metadata_raw: Dict[str, Any]) -> Dict[str, Any]:
        vec = preprocess_metadata(metadata_raw)
        self._load_encoder(input_dim=len(vec), embed_dim=32)
        x = torch.tensor(vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder(x).squeeze(0).cpu().numpy()
        return {
            "metadata_embedding": emb,
            "metadata_vector": vec,
            "provenance": {"source": "metadata_agent"}
        }
