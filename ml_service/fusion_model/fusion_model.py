"""
Fusion Model — combines CT, Audio, and Metadata embeddings.
Produces final cancer-risk score (0–1).
"""

import torch
import torch.nn as nn
import numpy as np


# -----------------------------------------------------------
#  FUSION MODEL ARCHITECTURE
# -----------------------------------------------------------

class FusionModel(nn.Module):
    def __init__(self,
                 ct_dim=256,
                 audio_dim=256,
                 meta_dim=32,
                 hidden_dim=256):
        super(FusionModel, self).__init__()

        # Project embeddings to same dimensionality
        self.ct_proj = nn.Linear(ct_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)

        # Fusion head
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, ct_emb, audio_emb, meta_emb):
        ct = self.ct_proj(ct_emb)
        audio = self.audio_proj(audio_emb)
        meta = self.meta_proj(meta_emb)

        combined = torch.cat([ct, audio, meta], dim=1)
        out = self.fusion_net(combined)

        return torch.sigmoid(out).squeeze(1)


# -----------------------------------------------------------
#  FUSION PREDICT FUNCTION (USED BY BACKEND)
# -----------------------------------------------------------

def fusion_predict(
        ct_embedding: np.ndarray,
        audio_embedding: np.ndarray,
        metadata_embedding: np.ndarray,
        weight_path: str = None,
        device: str = "cpu"
    ):
    """
    Final prediction function used by backend.

    Inputs:
        - ct_embedding (256,)
        - audio_embedding (256,)
        - metadata_embedding (32,)
    Returns:
        - float → probability (0–1)
    """

    # Convert to tensors
    ct_tensor = torch.tensor(ct_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    audio_tensor = torch.tensor(audio_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    meta_tensor = torch.tensor(metadata_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    # Load fusion model
    model = FusionModel().to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Loaded FusionModel weights.")
    else:
        print("[WARNING] No fusion weights provided — using untrained fusion model.")

    # Run inference
    with torch.no_grad():
        prob = model(ct_tensor, audio_tensor, meta_tensor).item()

    return float(prob)


# -----------------------------------------------------------
#  QUICK TEST
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Testing Fusion Model...")

    ct = np.random.randn(256)
    audio = np.random.randn(256)
    meta = np.random.randn(32)

    result = fusion_predict(ct, audio, meta)
    print("Predicted Fusion Risk:", result)
