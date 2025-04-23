"""
Data loader for WESAD with signal-specific window sizes and advanced feature engineering.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor
from features.feature_extractor import extract_features
from utils.logger import logger

# Signal configuration: sampling frequencies and window lengths (in seconds)
SIGNAL_CONFIG = {
    ('chest', 'ECG'): {'fs': 700, 'window_s': 1.0},
    ('chest', 'EMG'): {'fs': 700, 'window_s': 0.5},
    ('chest', 'EDA'): {'fs': 700, 'window_s': 2.0},
    ('chest', 'Resp'): {'fs': 700, 'window_s': 2.0},
    ('chest', 'Temp'): {'fs': 700, 'window_s': 5.0},
    ('chest', 'ACC'): {'fs': 700, 'window_s': 0.5},
    ('wrist', 'ACC'): {'fs': 32, 'window_s': 0.5},
    ('wrist', 'BVP'): {'fs': 64, 'window_s': 1.0},
    ('wrist', 'EDA'): {'fs': 4, 'window_s': 2.0},
    ('wrist', 'TEMP'): {'fs': 4, 'window_s': 5.0},
}

def generate_time_windows(total_duration_s: float, step_s: float):
    return [(i * step_s, (i + 1) * step_s) for i in range(int(total_duration_s // step_s))]

def extract_window_slice(signal_data: np.ndarray, fs: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = int(start_s * fs)
    end_idx = int(end_s * fs)
    if end_idx > len(signal_data):
        return None
    return signal_data[start_idx:end_idx]

def get_window_label(start_idx: int, end_idx: int, label_series: pd.Series) -> int:
    segment = label_series.iloc[start_idx:end_idx]
    return segment.mode().iloc[0] if not segment.empty else 0


def process_time_window(start_s, end_s, data, labels, step_s):
    feature_row = {}
    window_valid = True

    for (device, signal), conf in SIGNAL_CONFIG.items():
        if device not in data or signal not in data[device]:
            continue

        fs = conf['fs']
        signal_data = np.array(data[device][signal])
        start_idx = int(start_s * fs)
        end_idx = int(end_s * fs)
        if end_idx > len(signal_data):
            window_valid = False
            break

        win = signal_data[start_idx:end_idx]
        win = win if win.ndim > 1 else win.reshape(-1, 1)

        for dim in range(win.shape[1]):
            feats = extract_features(win[:, dim], f"{device}_{signal}_dim{dim}", fs)
            feature_row.update(feats)

    if not window_valid:
        return None, None

    label_start_idx = int(start_s * 700)
    label_end_idx = int(end_s * 700)
    segment = labels[label_start_idx:label_end_idx]
    label = int(pd.Series(segment).mode().iloc[0]) if len(segment) > 0 else 0

    return feature_row, label

def load_subject_timeframe(filepath: str, drop_non_study: bool, imputer_strategy: str, shorten_non_study: bool) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading subject data from {filepath}")
    with open(filepath, 'rb') as file:
        subject = pickle.load(file, encoding='latin1')

    labels = np.array(subject['label'])  # Use NumPy for fast slicing
    data = subject['signal']

    max_duration_s = len(labels) / 700.0
    step_s = 0.5
    time_windows = [(i * step_s, (i + 1) * step_s) for i in range(int(max_duration_s // step_s))]

    all_features = []
    all_labels = []

    logger.info("Processing all windows in parallel...")

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda t: process_time_window(t[0], t[1], data, labels, step_s),
            time_windows
        ))

    for feats, label in results:
        if feats is None:
            continue
        all_features.append(feats)
        all_labels.append(label)

    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)

    if drop_non_study:
        logger.info("Dropping non-study labels")
        mask = ~y.isin([0, 5, 6, 7])
        X, y = X[mask], y[mask]

    if shorten_non_study:
        logger.info("Shortening non-study labels to 0")
        y = y.replace({1: 0, 2: 0, 3: 0, 4: 0})
        y = y.astype(int)

    logger.info(f"Imputing missing values using strategy: {imputer_strategy}")
    imputer = SimpleImputer(strategy=imputer_strategy)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    logger.info(f"Finished processing subject data from {filepath}")
    return X, y

def preload_data(data_dir: str, drop_non_study: bool, imputer_strategy: str, shorten_non_study: bool) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    logger.info(f"Preloading data from directory: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    data = {}
    for f in files:
        try:
            logger.info(f"Processing file: {f}")
            X, y = load_subject_timeframe(os.path.join(data_dir, f), drop_non_study, imputer_strategy, shorten_non_study)
            data[f] = (X, y)
        except Exception as e:
            logger.warning(f"Skipping {f} due to an error: {e}")
    logger.info("Finished preloading data")
    return data