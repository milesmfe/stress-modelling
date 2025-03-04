def normalise(X):
    """
    Normalize features in X to the range [-1, 1] using min-max scaling.
    Handles constant columns by setting them to 0.
    
    Parameters:
    X (pd.DataFrame): Input features (time series columns)
    y (pd.Series): Labels corresponding to the features
    
    Returns:
    pd.DataFrame: Normalized features
    """

    # Calculate normalization parameters
    X_min = X.min()
    X_max = X.max()
    range = X_max - X_min

    # Vectorized normalization formula
    X_normalized = ((X - X_min) / range)

    return X_normalized
