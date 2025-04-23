""" 
Advanced feature extractor with parallel processing for the WESAD dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from concurrent.futures import ThreadPoolExecutor


def extract_basic_stats(signal: np.ndarray) -> dict:
    if np.std(signal) < 1e-6:
        return {
            "mean": float(np.mean(signal)),
            "std": 0.0,
            "max": float(np.max(signal)),
            "min": float(np.min(signal)),
            "skew": 0.0,
            "kurtosis": 0.0
        }
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "max": float(np.max(signal)),
        "min": float(np.min(signal)),
        "skew": float(skew(signal)),
        "kurtosis": float(kurtosis(signal))
    }

def extract_freq_features(signal: np.ndarray, fs: int) -> dict:
    nperseg = min(256, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    return {
        "fft_mean": float(np.mean(Pxx)),
        "fft_peak": float(np.max(Pxx)),
        "fft_entropy": float(-np.sum(Pxx * np.log(Pxx + 1e-12)))
    }

def extract_peaks(signal: np.ndarray) -> dict:
    peaks, _ = find_peaks(signal)
    inter_peak = np.diff(peaks) if len(peaks) > 1 else np.array([0])
    return {
        "num_peaks": len(peaks),
        "avg_peak_dist": float(np.mean(inter_peak)) if inter_peak.size > 0 else 0
    }

def extract_signal_features(win: np.ndarray, signal_name: str, fs: int) -> dict:
    win = win if win.ndim > 1 else win.reshape(-1, 1)
    features = {}

    def extract_dim(dim_idx):
        sig = win[:, dim_idx]
        feats = {}
        feats.update(extract_basic_stats(sig))
        feats.update(extract_freq_features(sig, fs))
        feats.update(extract_peaks(sig))
        return {f"{signal_name}_dim{dim_idx}_{k}": v for k, v in feats.items()}

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_dim, range(win.shape[1])))
        for result in results:
            features.update(result)

    return features

def extract_features(signal: np.ndarray, signal_name: str, fs: int) -> dict:
    stats = extract_basic_stats(signal)
    freq = extract_freq_features(signal, fs)
    peaks = extract_peaks(signal)
    combined = {f"{signal_name}_{k}": v for d in [stats, freq, peaks] for k, v in d.items()}
    return combined