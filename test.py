import pandas as pd
import time

from utils.synced_logger import logger

from preprocessing.load_pkl_with_alignment import load_pkl_with_alignment
from preprocessing.normalise import normalise
from preprocessing.remove_outliers import remove_outliers
from preprocessing.balance_classes import balance_classes

# from feature_generation.DWT import add_dwt_features
    
    
def test(subject):
    # Load data
    X, y = load_pkl_with_alignment(f'.data/{subject}.pkl')
    
    # Clean
    X, y = remove_outliers(X, y)
    
    logger.print_with_timestamp("Outliers Removed...")   
        
    # Balance classes
    X, y = balance_classes(X, y)
    
    logger.print_with_timestamp("Classes Balanced...")
    
    # TODO: Add DWT features
    
    # Normalise
    # TODO: Organise folds first, calculate min/max from train and apply to train/test
    X = normalise(X)
    
    logger.print_with_timestamp("Data Normalised...")
    
    # Save
    out = pd.concat([X, y], axis=1)
    out.sort_index(inplace=True)
    out.to_csv(f'{subject}.csv', index=False)
    
    logger.print_with_timestamp("Output Saved...")


if __name__ == '__main__':
    start_time = time.time()
    logger.print_with_timestamp("Starting...")
    test('S2')
    logger.print_with_timestamp("Done...")