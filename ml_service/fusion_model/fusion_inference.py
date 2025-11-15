# fusion_inference.py
"""
Full Fusion Inference Pipeline
Combines:
- CT embedding
- Audio embedding
- Metadata embedding
Returns:
- final cancer risk score
"""

import torch
import numpy as np

from ml_service.fusion_model.fusion_model import FusionModel
from ml_service.metadata_pipeline.preprocess_metadata import preprocess_metadata
from ml_service.metadata_pipeline.metadata_model import MetadataMLP


def load_fusion_model(weight_path=None, device="cpu"):
    print("[INFO] Loading Fusion Model...")

    model = FusionModel().to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Fusion weights loaded.")
    else:
        print("[WARNING] No fusion weights provided — using untrained model.")

    return model


def load_metadata_encoder(weight_path=None, device="cpu"):
    print("[INFO] Loading Metadata MLP...")

    # Example feature length: 1 age + 3 sex + 1 smoking + 8 symptoms = 13
    input_dim = 13

    model = MetadataMLP(input_dim=input_dim, embed_dim=32).to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Metadata encoder loaded.")

    return model


def fusion_full_inference(ct_emb,
                          audio_emb,
                          metadata_raw,
                          fusion_weights=None,
                          metadata_weights=None,
                          device="cpu"):
    """
    ct_emb: numpy array (256)
    audio_emb: numpy array (256)
    metadata_raw: dict
    """

    # Load models
    fusion_model = load_fusion_model(fusion_weights, device)
    metadata_encoder = load_metadata_encoder(metadata_weights, device)

    # Preprocess metadata
    meta_vec = preprocess_metadata(metadata_raw)
    meta_tensor = torch.tensor(meta_vec).float().unsqueeze(0).to(device)

    meta_emb = metadata_encoder(meta_tensor)  # (1,32)

    # Convert embeddings to tensors
    ct_tensor = torch.tensor(ct_emb).float().unsqueeze(0).to(device)
    audio_tensor = torch.tensor(audio_emb).float().unsqueeze(0).to(device)

    # Fusion prediction
    with torch.no_grad():
        prob = fusion_model(ct_tensor, audio_tensor, meta_emb).item()

    return float(prob)



if __name__ == "__main__":
    print("Testing fusion...")

    ct = np.random.randn(256)
    audio = np.random.randn(256)
    meta = {
        "age": 50,
        "sex": "female",
        "smoking_history": 1,
        "symptoms": ["cough", "short_breath"]
    }

    score = fusion_full_inference(ct, audio, meta)
    print("Fusion score:", score)
