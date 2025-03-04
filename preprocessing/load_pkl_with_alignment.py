import pickle

import numpy as np
import pandas as pd

from utils.synced_logger import logger

from preprocessing.align_signal import align_signal


def loadPklWithAlignment(pkl_path):
    # Load data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    signals = data['signal']
    labels = data['label']

    # Calculate timing parameters
    base_freq = 700  # Label frequency (Hz)
    collection_seconds = len(labels) / base_freq
    num_base_samples = len(labels)
    
    # Create numerical index (time in seconds)
    base_index = np.arange(num_base_samples)
    
    # Initialize DataFrames
    X = pd.DataFrame(index=base_index)
    y = pd.Series(labels, index=base_index, name='label')
    
    logger.print_with_timestamp("Data Loaded...")

    # Process each signal source
    for device, device_signals in signals.items():
        for signal_type, signal_data in device_signals.items():
            # Process each column in the signal
            for col_idx in range(signal_data.shape[1]):
                col_data = signal_data[:, col_idx]
                aligned_values = align_signal(
                    base_index=base_index,
                    signal_data=col_data,
                    collection_seconds=collection_seconds
                )
                col_name = f'{device}_{signal_type}_{col_idx}'
                X[col_name] = aligned_values
                
    logger.print_with_timestamp("Signals Processed...")
    
    return X, y