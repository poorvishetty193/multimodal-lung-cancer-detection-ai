# infer_audio.py
import librosa
import numpy as np
import joblib

MODEL_PATH = "./models/audio_model.pkl"
ENC_PATH = "./models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENC_PATH)

def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).reshape(1, -1)

def predict_audio(audio_path):
    features = extract_mfcc(audio_path)
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    label = encoder.inverse_transform([pred])[0]

    return {
        "predicted_label": label,
        "probabilities": {
            lbl: float(prob) for lbl, prob in zip(encoder.classes_, probs)
        }
    }
