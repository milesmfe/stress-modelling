"""
Command-line argument parsing module.
"""

import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="WESAD Pipeline CLI")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .pkl files.")
    parser.add_argument("--results_dir", type=str, default=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help="Directory to save results.")
    parser.add_argument("--window_size", type=float, default=1.0, help="Window size in seconds.")
    parser.add_argument("--group_func", type=str, default='mean', help="Aggregation function.")
    parser.add_argument("--drop_non_study", action="store_true", help="Drop labels 0, 5, 6, 7.")
    parser.add_argument("--shorten_non_study", action="store_true", help="Shorten non-study labels to 0.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--save_datasets", action="store_true", help="Save train/test datasets per fold.")
    parser.add_argument("--model_name", type=str, default="RandomForest", help="Model name in registry.")
    parser.add_argument("--feature_selection", action="store_true", help="Enable feature selection.")
    parser.add_argument("--imputer", type=str, default="mean", help="Imputation strategy: mean, median, most_frequent.")
    return parser.parse_args()