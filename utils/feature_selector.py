"""
Feature selection utility based on feature importance from RandomForest.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, y_train, X_test, threshold=0.01):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    mask = importance >= threshold
    return X_train[:, mask], X_test[:, mask]