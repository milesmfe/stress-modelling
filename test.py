import numpy as np
import pandas as pd
import time

from utils.synced_logger import logger

from preprocessing.load_pkl_with_alignment import load_pkl_with_alignment
from preprocessing.normalise import normalise
from preprocessing.remove_outliers import remove_outliers
from preprocessing.balance_classes import balance_classes

from feature_generation.dwt_features import WaveletFeatureGenerator
from feature_generation.statistical_features import StatisticalFeatureGenerator
    
    
def test(subject):
    # Load data
    X, y = load_pkl_with_alignment(f'.data/{subject}.pkl')
    
    # Clean
    X, y = remove_outliers(X, y)
    
    logger.info("Outliers Removed...")   
        
    # Balance classes
    X, y = balance_classes(X, y)
    
    logger.info("Classes Balanced...")
    
    sfg = StatisticalFeatureGenerator(
        window_size=20,
        features = ['mean', 'std']
    )
    
    wfg = WaveletFeatureGenerator(
        wavelet='db4',
        max_level=3,
        features=['energy', 'entropy']
    )
    
    # Generate features
    X = sfg.generate_features(X, [
        'chest_ACC_0', 
        'chest_ACC_1', 
        'chest_ACC_2', 
        'chest_ECG_0',
        'chest_Temp_0',
        'wrist_ACC_0',
        'wrist_ACC_1',
        'wrist_ACC_2',
        'wrist_TEMP_0'
    ])
    
    X = wfg.transform_columns(X, [
        'chest_EMG_0',
        'chest_EDA_0',
        'chest_Resp_0',
        'wrist_BVP_0',
        'wrist_EDA_0',
    ])
    
    logger.info("Features Generated...")
    
    # Normalise
    # TODO: Organise folds first, calculate min/max from train and apply to train/test
    X = normalise(X)
    
    logger.info("Data Normalised...")
    
    # Save
    out = pd.concat([X, y], axis=1)
    out.sort_index(inplace=True)
    out.to_csv(f'{subject}.csv', index=False)
    
    logger.info("Output Saved...")


if __name__ == '__main__':
    start_time = time.time()
    logger.info("Starting...")
    test('S2')
    logger.info("Done...")