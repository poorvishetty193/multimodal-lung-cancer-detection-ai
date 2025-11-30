# utils_audio.py
import librosa
import numpy as np

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y
