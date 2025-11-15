# metadata_model.py
"""
Metadata encoder model.
Input: structured metadata vector (length ~ 1 + 3 + 1 + symptoms)
Output: metadata embedding (32-dim by default)
"""

import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super(MetadataMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # example test
    from preprocess_metadata import preprocess_metadata
    
    raw = {
        "age": 55,
        "sex": "female",
        "smoking_history": 1,
        "symptoms": ["cough", "short_breath"]
    }

    feat = preprocess_metadata(raw)
    model = MetadataMLP(input_dim=len(feat))

    x = torch.tensor(feat).unsqueeze(0)
    e = model(x)
    print("Embedding shape:", e.shape)
