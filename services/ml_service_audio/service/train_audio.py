# train_audio.py
import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "./audio_training_data"  # your dataset folder
MODEL_PATH = "./models/audio_model.pkl"
ENC_PATH = "./models/label_encoder.pkl"

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)

def load_dataset():
    X, y = [], []
    for label in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        for f in os.listdir(class_dir):
            if not f.lower().endswith((".wav", ".mp3", ".m4a")):
                continue

            path = os.path.join(class_dir, f)
            features = extract_mfcc(path)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def train():
    print("Loading dataset...")
    X, y = load_dataset()

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y_enc)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(enc, ENC_PATH)

    print("Training complete → Model saved.")

if __name__ == "__main__":
    train()
