""" 
Advanced feature extractor with parallel processing for the WESAD dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from antropy import sample_entropy, spectral_entropy, higuchi_fd, detrended_fluctuation
from concurrent.futures import ThreadPoolExecutor

def extract_basic_stats(signal):
    if np.allclose(signal, signal[0]):
        return {
            "mean": float(signal[0]),
            "std": 0.0,
            "max": float(signal[0]),
            "min": float(signal[0]),
            "skew": 0.0,
            "kurtosis": 0.0
        }
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "max": float(np.max(signal)),
        "min": float(np.min(signal)),
        "skew": float(skew(signal, bias=False)),
        "kurtosis": float(kurtosis(signal, bias=False))
    }

def extract_freq_features(signal: np.ndarray, fs: int) -> dict:
    min_required_length = 64
    if len(signal) < min_required_length:
        return {}  # Skip frequency features for short signals

    nperseg = min(256, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)

    from antropy import spectral_entropy
    try:
        fft_entropy = spectral_entropy(signal, sf=fs, method='welch') if np.sum(Pxx) > 0 else 0.0
    except Exception:
        fft_entropy = 0.0

    band_powers = {
        "power_low": float(np.trapezoid(Pxx[(f >= 0.04) & (f < 0.15)], dx=np.mean(np.diff(f)))),
        "power_mid": float(np.trapezoid(Pxx[(f >= 0.15) & (f < 0.4)], dx=np.mean(np.diff(f)))),
        "power_high": float(np.trapezoid(Pxx[(f >= 0.4)], dx=np.mean(np.diff(f))))
    }

    return {
        **band_powers,
        "fft_mean": float(np.mean(Pxx)),
        "fft_entropy": fft_entropy
    }

def extract_entropy_features(signal):
    if np.std(signal) < 1e-6:
        return {"sample_entropy": 0.0, "higuchi_fd": 0.0, "dfa": 0.0}
    return {
        "sample_entropy": float(sample_entropy(signal)),
        "higuchi_fd": float(higuchi_fd(signal)),
        "dfa": float(detrended_fluctuation(signal))
    }

def extract_peaks(signal):
    peaks, _ = find_peaks(signal)
    inter_peak = np.diff(peaks) if len(peaks) > 1 else np.array([0])
    return {
        "num_peaks": len(peaks),
        "avg_peak_dist": float(np.mean(inter_peak)) if inter_peak.size > 0 else 0
    }

def extract_hrv(signal, fs):
    peaks, _ = find_peaks(signal)
    rr_intervals = np.diff(peaks) / fs
    if len(rr_intervals) < 2:
        return {"rmssd": 0, "sdnn": 0, "pnn50": 0}
    diff_rr = np.diff(rr_intervals)
    return {
        "rmssd": np.sqrt(np.mean(diff_rr**2)),
        "sdnn": np.std(rr_intervals),
        "pnn50": 100 * np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)
    }

def remove_outliers_iqr(signal: np.ndarray) -> np.ndarray:
    """Remove outliers from a 1D signal using IQR."""
    Q1 = np.percentile(signal, 25)
    Q3 = np.percentile(signal, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered = signal[(signal >= lower_bound) & (signal <= upper_bound)]
    
    # Ensure minimum viable length for feature extraction
    if len(filtered) < 10:  # safeguard for very short segments
        return signal
    return filtered

def extract_signal_features(win: np.ndarray, signal_name: str, fs: int) -> dict:
    win = win if win.ndim > 1 else win.reshape(-1, 1)
    features = {}

    def extract_dim(dim_idx):
        sig = win[:, dim_idx]
        sig = remove_outliers_iqr(sig)  # Remove outliers

        feats = {}
        feats.update(extract_basic_stats(sig))
        freq_feats = extract_freq_features(sig, fs)
        if freq_feats:
            feats.update(freq_feats)
        feats.update(extract_peaks(sig))
        feats.update(extract_entropy_features(sig))
        if "ECG" in signal_name.upper():
            feats.update(extract_hrv(sig, fs))
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