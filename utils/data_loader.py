"""
Data loader for WESAD with signal-specific window sizes and advanced feature engineering.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.impute import SimpleImputer
from features.feature_extractor import extract_features
from utils.logger import logger

# Define signal-specific frequencies and window sizes in seconds
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

def window_signal(signal: np.ndarray, fs: int, window_s: float) -> list:
    logger.debug(f"Windowing signal with fs={fs}, window_s={window_s}")
    window_len = int(fs * window_s)
    return [signal[i:i + window_len] for i in range(0, len(signal) - window_len, window_len)]

def load_subject_timeframe(filepath: str, drop_non_study: bool, imputer_strategy: str, shorten_non_study: bool) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading subject data from {filepath}")
    with open(filepath, 'rb') as file:
        subject = pickle.load(file, encoding='latin1')

    labels = pd.Series(subject['label'])
    data = subject['signal']

    all_features = []
    all_labels = []

    for signal_key, conf in SIGNAL_CONFIG.items():
        device, signal = signal_key
        if device not in data or signal not in data[device]:
            logger.warning(f"Signal {signal} from device {device} not found in data.")
            continue

        logger.debug(f"Processing signal {signal} from device {device}")
        signal_data = np.array(data[device][signal])
        fs, window_s = conf['fs'], conf['window_s']
        windows = window_signal(signal_data, fs, window_s)
        label_windows = labels.groupby(labels.index // int(fs * window_s)).apply(lambda x: x.mode()[0])
        label_windows = label_windows.iloc[:len(windows)]

        for i, win in enumerate(windows):
            features = {}
            win = win if win.ndim > 1 else win.reshape(-1, 1)
            for dim in range(win.shape[1]):
                dim_features = extract_features(win[:, dim], f"{device}_{signal}_dim{dim}", fs)
                features.update(dim_features)

            all_features.append(features)
            all_labels.append(label_windows.iloc[i])

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

def preload_data(data_dir: str, drop_non_study: bool, imputer_strategy: str, shorten_non_study) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
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