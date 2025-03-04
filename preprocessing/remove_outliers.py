import pandas as pd
import numpy as np

def remove_outliers(X, y, window_size=100, threshold=3.0):
    """
    Remove local outliers from time series data using sliding window Z-score analysis.
    
    Parameters:
    X (pd.DataFrame): Feature matrix with time series data
    y (pd.Series): Labels corresponding to X
    window_size (int): Size of the sliding window
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame, pd.Series: Cleaned X and y with outliers removed
    """

    # Create outlier mask
    outlier_mask = pd.Series(False, index=X.index)
    half_window = window_size // 2

    for col in X.columns:
        # Convert to numpy for efficient window operations
        values = X[col].values
        n = len(values)
        z_scores = np.zeros(n)
        
        for i in range(n):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window = values[start:end]
            
            # Skip if window is too small
            if len(window) < 3:
                z_scores[i] = 0
                continue
                
            # Calculate robust statistics
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            if mad == 0:
                z_scores[i] = 0
            else:
                # Modified Z-score using MAD
                z_scores[i] = 0.6745 * (values[i] - median) / mad

        # Update outlier mask
        outlier_mask |= np.abs(z_scores) > threshold

    # Filter outliers and maintain temporal order
    clean_X = X.loc[~outlier_mask].copy()
    clean_y = y.loc[~outlier_mask].copy()
    
    return clean_X, clean_y
