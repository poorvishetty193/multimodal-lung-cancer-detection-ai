# ml_service/audio_pipeline/audio_inference.py
"""
Audio inference pipeline:
1. Load & preprocess audio
2. Compute log-mel spectrogram
3. Run CRNN model
4. Output:
    - audio_abnormality_score (0–1)
    - audio_embedding (vector for fusion model)
"""

import torch
import numpy as np
from pathlib import Path

from ml_service.audio_pipeline.preprocess_audio import preprocess_file
from ml_service.audio_pipeline.crnn_model import CRNN


# ---------------------------------------------------------
# Load CRNN Model
# ---------------------------------------------------------

def load_crnn_model(weight_path=None, device="cpu"):
    print("[INFO] Loading CRNN model...")

    model = CRNN(n_mels=64, num_classes=1).to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Loaded CRNN weights.")
    else:
        print("[WARNING] No weights provided — using untrained CRNN.")

    return model


# ---------------------------------------------------------
# Extract Embedding (before classification)
# ---------------------------------------------------------

def extract_embedding(model, log_mel, device="cpu"):
    """
    Extract embedding from CRNN:
    use the output of the LSTM BEFORE final classifier.
    """
    with torch.no_grad():
        x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,T)
        cnn_out = model.cnn(x)              # (1,C,freq,time)
        cnn_out = cnn_out.mean(dim=2)       # average over frequencies → (1,C,time)
        cnn_out = cnn_out.permute(0, 2, 1)  # (1,time,C)

        # RNN
        rnn_out, _ = model.rnn(cnn_out)     # (1,time,2*hidden)

        # Average pooling over time
        embedding = rnn_out.mean(dim=1)     # (1,2*hidden)
        embedding = embedding.squeeze(0).cpu().numpy()  # vector (256)
        return embedding


# ---------------------------------------------------------
# Master Inference Pipeline
# ---------------------------------------------------------

def audio_full_inference(path,
                         weight_path=None,
                         device="cpu"):
    """
    Input:
        - path to audio (.wav)
    Output:
        {
            "audio_score": float,
            "audio_embedding": np.ndarray,
            "log_mel": np.ndarray
        }
    """

    print("\n==============================")
    print("      AUDIO INFERENCE")
    print("==============================\n")

    # Step 1 — Preprocess audio
    audio = preprocess_file(path)
    log_mel = audio["log_mel"]     # (64, time)

    # Step 2 — Load model
    model = load_crnn_model(weight_path, device)

    # Prepare tensor
    mel_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)

    # Step 3 — Predict score
    with torch.no_grad():
        score = model(mel_tensor).item()    # probability (0–1)

    # Step 4 — Get embedding for fusion model
    embedding = extract_embedding(model, log_mel, device)

    return {
        "audio_score": float(score),
        "audio_embedding": embedding,
        "log_mel": log_mel
    }


# ---------------------------------------------------------
# Quick Test
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Testing audio inference...")

    test_path = "example.wav"  # replace with real path

    try:
        result = audio_full_inference(test_path)
        print("Audio Score:", result["audio_score"])
        print("Embedding shape:", result["audio_embedding"].shape)
    except Exception as e:
        print("Error:", e)
