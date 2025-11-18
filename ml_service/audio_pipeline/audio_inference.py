import torch
import numpy as np

from ml_service.audio_pipeline.preprocess_audio import preprocess_file
from ml_service.audio_pipeline.crnn_model import CRNN


class AudioInference:
    """
    Wrapper class for the audio inference pipeline.
    Use: model = AudioInference(); model.run("file.wav")
    """

    def __init__(self, weight_path=None, device="cpu"):
        self.device = device
        self.weight_path = weight_path
        self.model = self._load_model()

    # ------------------------------    
    def _load_model(self):
        print("[INFO] Loading CRNN model...")
        model = CRNN(n_mels=64, num_classes=1).to(self.device)
        model.eval()

        if self.weight_path:
            model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            print("[INFO] Loaded CRNN weights.")
        else:
            print("[WARNING] No weights provided — using untrained CRNN.")

        return model

    # ------------------------------
    def _extract_embedding(self, log_mel):
        with torch.no_grad():
            x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)

            cnn_out = self.model.cnn(x)
            cnn_out = cnn_out.mean(dim=2)
            cnn_out = cnn_out.permute(0, 2, 1)

            rnn_out, _ = self.model.rnn(cnn_out)

            embedding = rnn_out.mean(dim=1)
            return embedding.squeeze(0).cpu().numpy()

    # ------------------------------
    def run(self, audio_path):
        """
        Runs full audio inference.
        Returns:
            {
                "audio_score": float,
                "audio_embedding": np.ndarray,
                "log_mel": np.ndarray
            }
        """
        print("\n==========================")
        print("     AUDIO INFERENCE")
        print("==========================\n")

        # Step 1 – preprocess
        audio = preprocess_file(audio_path)
        log_mel = audio["log_mel"]

        # Step 2 – score
        mel_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model(mel_tensor).item()

        # Step 3 – embedding
        embedding = self._extract_embedding(log_mel)

        return {
            "audio_score": float(score),
            "audio_embedding": embedding,
            "log_mel": log_mel
        }
