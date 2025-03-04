import pandas as pd


def remove_outliers(X, y, window_size=20, threshold=3):
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
    X = X.copy()
    y = y.copy()
    mask = pd.Series(False, index=X.index)
    
    for col in X.columns:
        # Calculate rolling statistics for previous window (excl. current row)
        rolling_mean = X[col].rolling(
            window=window_size,
            min_periods=window_size
        ).mean().shift(1)
        
        rolling_std = X[col].rolling(
            window=window_size,
            min_periods=window_size
        ).std(ddof=0).shift(1)  # Population standard deviation
        
        # Compute Z-scores and identify outliers
        z_scores = (X[col] - rolling_mean) / rolling_std
        mask_col = z_scores.abs() > threshold
        mask |= mask_col.fillna(False)
    
    # Filter out outliers while preserving original indices
    valid_indices = mask[~mask].index
    return X.loc[valid_indices], y.loc[valid_indices]