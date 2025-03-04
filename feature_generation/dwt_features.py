import numpy as np
import pandas as pd
import pywt
from typing import List, Union

from utils.synced_logger import logger

class WaveletFeatureGenerator:
    """
    DWT feature generator
    
    Parameters: s
    wavelet : str
        Wavelet type (default: 'db1')
    mode : str
        Signal extension mode (default: 'sym')
    max_level : int
        Maximum decomposition level (default: None for auto-calculate)
    features : List[str]
        Features to extract: ['energy', 'entropy', 'mean', 'std', 'abs_mean', 'abs_std']
    skip_nans : bool
        Remove features with NaN values (default: True)
    """

    def __init__(self, 
                 wavelet: str = 'db1',
                 mode: str = 'sym',
                 max_level: int = None,
                 features: List[str] = None,
                 skip_nans: bool = True):
        
        self.wavelet = wavelet
        self.mode = mode
        self.max_level = max_level
        self.skip_nans = skip_nans
        
        # Initialize wavelet object with validation
        try:
            self.wavelet_obj = pywt.Wavelet(self.wavelet)
        except ValueError as e:
            logger.error(f"Invalid wavelet {self.wavelet}: {str(e)}")
            raise
            
        # Configure feature extraction functions
        self.feature_map = {
            'energy': self._calc_energy,
            'entropy': self._calc_entropy,
            'mean': self._calc_mean,
            'std': self._calc_std,
            'abs_mean': self._calc_abs_mean,
            'abs_std': self._calc_abs_std
        }
        
        self.features = features or list(self.feature_map.keys())
        self.feature_names_ = []

    def transform_columns(self, 
                         df: pd.DataFrame, 
                         columns: List[str]) -> pd.DataFrame:
        """
        Transform specified DataFrame columns in-place with wavelet features
        
        Parameters:
        df : pd.DataFrame
            Input DataFrame with time series columns
        columns : List[str]
            List of columns to transform
        
        Returns:
        pd.DataFrame with original columns replaced by wavelet features
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            logger.info(f"Processing column: {col}")
            
            # Convert to numpy array and validate
            X = self._validate_input(df[col])
            
            # Generate wavelet features
            features = self._process_batch(X)
            
            # Create feature DataFrame
            feature_names = self._generate_feature_names(prefix=f"{col}_")
            features_df = pd.DataFrame(features, 
                                     columns=feature_names,
                                     index=df.index)
            
            # Replace original column
            df = df.drop(columns=[col])
            df = pd.concat([df, features_df], axis=1)
            
        return df

    def _process_batch(self, X: np.ndarray) -> np.ndarray:
        """Process batch of time series data"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        features = []
        for sample in X:
            coeffs = self._decompose(sample)
            sample_features = self._extract_features(coeffs)
            features.append(sample_features)
            
        features = np.array(features)
        
        if self.skip_nans:
            features = np.nan_to_num(features)
            
        return features

    def _decompose(self, sample: np.ndarray) -> List[np.ndarray]:
        """Perform multilevel wavelet decomposition"""
        if self.max_level is None:
            self.max_level = pywt.dwt_max_level(
                len(sample), 
                self.wavelet_obj.dec_len
            )
            
        return pywt.wavedec(
            sample, 
            self.wavelet_obj, 
            mode=self.mode, 
            level=self.max_level
        )

    def _extract_features(self, coeffs: List[np.ndarray]) -> List[float]:
        """Extract features from wavelet coefficients"""
        features = []
        for level, c in enumerate(coeffs):
            for feat_name in self.features:
                feat_value = self.feature_map[feat_name](c)
                features.append(feat_value)
                # Store feature name template
                self.feature_names_.append(f"level{level}_{feat_name}")
                
        return features

    def _generate_feature_names(self, prefix: str = "") -> List[str]:
        """Generate formatted feature names"""
        return [f"{prefix}{name}" for name in set(self.feature_names_)]

    def _validate_input(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input data"""
        if isinstance(data, pd.Series):
            if data.apply(lambda x: isinstance(x, (list, np.ndarray))).all():
                return np.stack(data.values)
            return data.to_numpy().reshape(-1, 1)
            
        if isinstance(data, np.ndarray):
            return data
            
        raise TypeError("Input must be pandas Series or numpy array")

    # Feature calculation methods
    def _calc_energy(self, c: np.ndarray) -> float:
        return np.sum(c**2)

    def _calc_entropy(self, c: np.ndarray) -> float:
        energy = np.sum(c**2)
        if energy == 0:
            return 0.0
        prob = (c**2) / energy
        return -np.sum(prob * np.log2(prob + 1e-12))

    def _calc_mean(self, c: np.ndarray) -> float:
        return np.mean(c)

    def _calc_std(self, c: np.ndarray) -> float:
        return np.std(c)

    def _calc_abs_mean(self, c: np.ndarray) -> float:
        return np.mean(np.abs(c))

    def _calc_abs_std(self, c: np.ndarray) -> float:
        return np.std(np.abs(c))
