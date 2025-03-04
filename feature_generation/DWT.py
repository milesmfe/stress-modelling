import numpy as np
import pandas as pd
import pywt

def _shannon_entropy(signal):
    """Calculate Shannon entropy of a signal"""
    energy = np.sum(signal**2)
    if energy == 0:
        return 0.0
    prob = (signal**2) / energy
    prob = prob[prob > 0]  # Remove zero probabilities
    return -np.sum(prob * np.log2(prob))

def add_dwt_features(X, signal_names, wavelet='db4', level=5):
    """
    Adds DWT frequency-domain features to a DataFrame for specified signals.
    
    Parameters:
    X (pd.DataFrame): Input dataframe with time series data
    signal_names (list): List of column names containing signals to process
    wavelet (str): Wavelet type (default: 'db4')
    level (int): Maximum decomposition level (default: 5)
    
    Returns:
    pd.DataFrame: DataFrame with new DWT features added
    """
    df = X.copy()
    
    for signal in signal_names:
        if signal not in df.columns:
            raise ValueError(f"Column {signal} not found in DataFrame")
            
        # Process each row independently
        for idx, row in df.iterrows():
            ts = row[signal]
            
            # Convert to numpy array if needed
            if not isinstance(ts, np.ndarray):
                ts = np.array(ts)
                
            # Skip invalid time series
            if ts.size < 2:
                continue
                
            # Calculate actual decomposition level
            max_possible_level = pywt.dwt_max_level(len(ts), wavelet)
            actual_level = min(level, max_possible_level)
            
            if actual_level < 1:
                continue
                
            try:
                coeffs = pywt.wavedec(ts, wavelet=wavelet, level=actual_level)
            except ValueError:
                continue

            # Extract features from detail coefficients
            for lvl in range(1, actual_level + 1):
                cD = coeffs[-lvl]  # Detail coefficients for current level
                
                # Calculate features
                features = {
                    f'{signal}_dwt_energy_l{lvl}': np.sum(cD**2),
                    f'{signal}_dwt_mean_l{lvl}': np.mean(cD),
                    f'{signal}_dwt_std_l{lvl}': np.std(cD),
                    f'{signal}_dwt_min_l{lvl}': np.min(cD),
                    f'{signal}_dwt_max_l{lvl}': np.max(cD),
                    f'{signal}_dwt_entropy_l{lvl}': _shannon_entropy(cD)
                }
                
                # Add features to dataframe
                for feat, val in features.items():
                    df.at[idx, feat] = val
                    
    return df
