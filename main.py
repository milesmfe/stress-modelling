"""
Main controller for WESAD advanced pipeline.
"""

import os
import pickle
from utils.arguments import parse_args
from utils.data_loader import preload_data
from utils.cross_validation import run_cross_validation
from utils.logger import logger

def main():
    logger.info("Parsing arguments...")
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    logger.info(f"Results directory created or already exists: {args.results_dir}")

    data = None
    if args.use_cache and os.path.exists(args.use_cache):
        logger.info(f"Cache file found at {args.use_cache}. Loading data from cache...")
        with open(args.use_cache, 'rb') as cache_file:
            data = pickle.load(cache_file)
        logger.info("Data loaded from cache successfully.")
    else:
        logger.info("Preloading data...")
        data = preload_data(
            data_dir=args.data_dir,
            drop_non_study=args.drop_non_study,
            imputer_strategy=args.imputer,
            shorten_non_study=args.shorten_non_study,
        )
        logger.info("Data preloaded successfully.")
        if args.use_cache:
            logger.info(f"Saving data to cache at {args.use_cache}...")
            with open(args.use_cache, 'wb') as cache_file:
                pickle.dump(data, cache_file)
            logger.info("Data saved to cache successfully.")

    logger.info("Starting cross-validation...")
    run_cross_validation(
        data=data,
        n_splits=args.n_splits,
        results_dir=args.results_dir,
        save_datasets=args.save_datasets,
        model_name=args.model_name,
        use_feature_selection=args.feature_selection
    )
    logger.info("Cross-validation completed.")

if __name__ == "__main__":
    logger.info("Starting main pipeline...")
    main()
    logger.info("Pipeline execution finished.")