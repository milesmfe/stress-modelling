"""
Cross-validation using extracted features with optional feature selection.
"""

import os
import pickle
import pandas as pd
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from utils.logger import logger
from utils.evaluation import save_results
from utils.feature_selector import select_features
from models.model_registry import get_model

RANDOM_STATE = 42

def run_cross_validation(data, n_splits, results_dir, save_datasets, model_name, use_feature_selection):
    logger.info("Starting cross-validation process.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    file_keys = list(data.keys())
    logger.info(f"Data contains {len(file_keys)} files. Performing {n_splits}-fold cross-validation.")

    for fold, (train_idx, test_idx) in enumerate(kf.split(file_keys), start=1):
        logger.info(f"Processing fold {fold}...")
        train_files = [file_keys[i] for i in train_idx]
        test_files = [file_keys[i] for i in test_idx]

        logger.debug(f"Train files: {train_files}")
        logger.debug(f"Test files: {test_files}")

        X_train = pd.concat([data[f][0] for f in train_files]).reset_index(drop=True)
        y_train = pd.concat([data[f][1] for f in train_files]).reset_index(drop=True)
        X_test = pd.concat([data[f][0] for f in test_files]).reset_index(drop=True)
        y_test = pd.concat([data[f][1] for f in test_files]).reset_index(drop=True)

        if save_datasets:
            train_path = f"{results_dir}/fold{fold}_train.csv"
            test_path = f"{results_dir}/fold{fold}_test.csv"
            logger.info(f"Saving training dataset to {train_path}")
            logger.info(f"Saving testing dataset to {test_path}")
            X_train.assign(label=y_train).to_csv(train_path, index=False)
            X_test.assign(label=y_test).to_csv(test_path, index=False)

        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if use_feature_selection:
            logger.info("Performing feature selection...")
            X_train, X_test = select_features(X_train, y_train, X_test)

        # Balance classes with RUS
        logger.info("Balancing classes using Random Under Sampling...")
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

        logger.info(f"Initializing model: {model_name}")
        clf = get_model(model_name)
        logger.info("Training the model...")
        clf.fit(X_train, y_train)
        logger.info("Model training complete. Making predictions...")
        y_pred = clf.predict(X_test)

        logger.info(f"Fold {fold} - {model_name} Report:\n{classification_report(y_test, y_pred)}")
        save_results(y_test, y_pred, model_name, fold, results_dir)

        model_path = os.path.join(results_dir, f"fold{fold}_{model_name.lower()}_model.pkl")
        logger.info(f"Saving model for fold {fold} to {model_path}")
        with open(model_path, 'wb') as model_file:
            pickle.dump(clf, model_file)

    logger.info("Cross-validation process completed.")