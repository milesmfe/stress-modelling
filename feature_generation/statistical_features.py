from utils.synced_logger import logger


class StatisticalFeatureGenerator:
    """Generates statistical features for time series data using rolling windows.
    
    Parameters:
    features (list): Statistical features to calculate (e.g., ['mean', 'std', 'min', 'max'])
    window_size (int): Size of rolling window for calculations (default=3)
    """
    
    def __init__(self, features, window_size=3):
        self.features = features
        self.window_size = window_size
        
    def generate_features(self, df, columns):
        """Generate specified features for given columns using rolling windows.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            columns (list): List of column names to process
            
        Returns:
            pd.DataFrame: DataFrame with original columns removed and new features added
        """
        for col in columns:
            logger.info(f"Generating features for {col}...")
            rolling = df[col].rolling(window=self.window_size)
            for feature in self.features:
                if hasattr(rolling, feature):
                    func = getattr(rolling, feature)
                    df[f'{col}_{feature}'] = func()
                else:
                    logger.error(f"Unsupported feature: {feature}")
                    raise
        df.drop(columns=columns, inplace=True)
        return df
