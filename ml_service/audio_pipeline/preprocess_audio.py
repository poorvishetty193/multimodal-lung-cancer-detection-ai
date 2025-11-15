# ml-service/audio_pipeline/preprocess_audio.py
"""
Audio preprocessing utilities for cough & breath recordings.

Features:
- Load WAV/MP3
- Resample to target_sr (default 16000)
- Trim silence
- Basic denoising (spectral gating baseline)
- Compute log-mel spectrograms and MFCCs
- Data augmentation helpers (time shift, pitch shift, add noise)

Dependencies:
pip install numpy librosa soundfile scipy
"""

import os
from typing import Tuple, Any, Optional
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sps

def load_audio(path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray,int]:
    """Load audio file and resample to target_sr."""
    y, sr = sf.read(path)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def trim_silence(y: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
    """Trim leading/trailing silence using librosa.effects."""
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    return yt

def add_background_noise(y: np.ndarray, noise_level_db: float = -30.0) -> np.ndarray:
    """Add gaussian noise at specified dB level relative to signal RMS."""
    rms = np.sqrt(np.mean(y**2))
    desired_rms = rms * (10.0 ** (noise_level_db / 20.0))
    noise = np.random.normal(0, 1.0, size=y.shape).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise**2))
    noise = noise * (desired_rms / (noise_rms + 1e-9))
    return y + noise

def time_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Random time shift up to shift_max seconds (wrap-around)."""
    if y.size == 0:
        return y
    sr = 16000
    max_shift = int(sr * shift_max)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(y, shift)

def pitch_shift(y: np.ndarray, sr: int, n_steps: float):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def spectral_gating_denoise(y: np.ndarray, sr: int, prop_decrease=1.0) -> np.ndarray:
    """
    Simple spectral gating denoise (baseline). Not a replacement for robust denoisers.
    """
    # Compute STFT
    stft = librosa.stft(y, n_fft=1024, hop_length=512)
    mag, phase = np.abs(stft), np.angle(stft)
    # Estimate noise from first 0.5s
    noise_frames = int(max(1, (0.5 * sr) / 512))
    noise_spec = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    # Reduce magnitude where below noise estimate
    mask_gain = np.maximum(0.0, mag - prop_decrease * noise_spec)
    filtered = mask_gain * np.exp(1j * phase)
    y_denoised = librosa.istft(filtered, hop_length=512)
    return y_denoised

def compute_log_mel_spectrogram(y: np.ndarray, sr: int = 16000,
                                n_mels: int = 64, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    """
    Return log-mel spectrogram (shape: n_mels x time)
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    # Normalize to 0-1
    log_S_norm = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-9)
    return log_S_norm.astype(np.float32)

def compute_mfcc(y: np.ndarray, sr: int = 16000, n_mfcc: int = 40, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc.astype(np.float32)

def pad_or_trim(y: np.ndarray, sr: int, target_seconds: float = 3.0) -> np.ndarray:
    """Pad or trim audio to fixed duration for batch processing."""
    target_len = int(sr * target_seconds)
    if y.shape[0] > target_len:
        # center crop
        start = (y.shape[0] - target_len) // 2
        return y[start:start + target_len]
    elif y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        left = pad // 2
        right = pad - left
        return np.pad(y, (left, right), mode='constant')
    else:
        return y

# Convenience pipeline
def preprocess_file(path: str,
                    target_sr: int = 16000,
                    denoise: bool = True,
                    trim: bool = True,
                    target_seconds: float = 3.0) -> dict:
    """
    Full preprocess: load -> resample -> trim -> denoise -> pad -> features
    Returns dict with:
      - 'waveform': np.ndarray (target_seconds * sr)
      - 'sr': int
      - 'log_mel': np.ndarray (n_mels, time)
      - 'mfcc': np.ndarray
    """
    y, sr = load_audio(path, target_sr=target_sr)
    if trim:
        y = trim_silence(y, sr)
    if denoise:
        try:
            y = spectral_gating_denoise(y, sr)
        except Exception:
            # if denoising fails fallback to raw
            pass
    y = pad_or_trim(y, sr, target_seconds=target_seconds)
    log_mel = compute_log_mel_spectrogram(y, sr)
    mfcc = compute_mfcc(y, sr)
    return {
        "waveform": y,
        "sr": sr,
        "log_mel": log_mel,
        "mfcc": mfcc
    }

# Simple CLI test
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    args = p.parse_args()
    out = preprocess_file(args.file)
    print("Log-mel shape:", out["log_mel"].shape)
    print("MFCC shape:", out["mfcc"].shape)
