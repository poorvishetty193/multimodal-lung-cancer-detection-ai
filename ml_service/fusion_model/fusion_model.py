# fusion_model.py
"""
Fusion Model — combines CT, Audio, and Metadata embeddings.
Final output: cancer risk probability (0–1)

Embeddings (default sizes):
- CT embedding: 256
- Audio embedding: 256
- Metadata embedding: 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self,
                 ct_dim=256,
                 audio_dim=256,
                 meta_dim=32,
                 hidden_dim=256):
        super(FusionModel, self).__init__()

        # Linear layers to unify embedding dimensions
        self.ct_proj = nn.Linear(ct_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)

        # Combined fusion network
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
        """
        All embeddings should have shape: (B, dim)
        """

        ct = self.ct_proj(ct_emb)
        audio = self.audio_proj(audio_emb)
        meta = self.meta_proj(meta_emb)

        combined = torch.cat([ct, audio, meta], dim=1)

        out = self.fusion_net(combined)
        return torch.sigmoid(out).squeeze(1)  # probability
        

if __name__ == "__main__":
    # Quick test
    model = FusionModel()

    ct = torch.randn(2, 256)
    audio = torch.randn(2, 256)
    meta = torch.randn(2, 32)

    prob = model(ct, audio, meta)
    print("Output:", prob.shape)
