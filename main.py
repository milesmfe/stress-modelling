"""
Main controller for WESAD advanced pipeline.
"""

import os
from utils.arguments import parse_args
from utils.data_loader import preload_data
from utils.cross_validation import run_cross_validation
from utils.logger import logger

def main():
    logger.info("Parsing arguments...")
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    logger.info(f"Results directory created or already exists: {args.results_dir}")

    logger.info("Preloading data...")
    data = preload_data(
        data_dir=args.data_dir,
        drop_non_study=args.drop_non_study,
        imputer_strategy=args.imputer,
        shorten_non_study=args.shorten_non_study,
    )
    logger.info("Data preloaded successfully.")

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