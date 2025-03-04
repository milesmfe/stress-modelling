import pandas as pd
import time

from utils.synced_logger import logger

from preprocessing.load_pkl_with_alignment import loadPklWithAlignment
from preprocessing.normalise import normalise
from preprocessing.remove_outliers import remove_outliers
from preprocessing.balance_classes import underSample
from preprocessing.align_signal import align_signal

from feature_generation.DWT import add_dwt_features
    
    
def test(subject):
    # Load data
    X, y = loadPklWithAlignment(f'.data/{subject}.pkl')
    
    # Clean
    X, y = remove_outliers(
        X, 
        y,
        threshold=3,
        window_size=100
    )
    
    logger.print_with_timestamp("Outliers Removed...", start_time)   
        
    # Balance classes
    X, y = underSample(X, y)
    
    logger.print_with_timestamp("Classes Balanced...", start_time)
    
    # TODO: Add DWT features
    
    # Normalise
    # TODO: Organise folds first, calculate min/max from train and apply to train/test
    X = normalise(X)
    
    logger.print_with_timestamp("Data Normalised...", start_time)
    
    # Save
    out = pd.concat([X, y], axis=1)
    out.sort_index(inplace=True)
    out.to_csv(f'{subject}.csv', index=False)
    
    logger.print_with_timestamp("Output Saved...", start_time)


if __name__ == '__main__':
    start_time = time.time()
    logger.print_with_timestamp("Starting...", start_time)
    test('S2')