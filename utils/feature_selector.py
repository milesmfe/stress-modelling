"""
Feature selection utility based on feature importance from RandomForest.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, y_train, X_test, columns, threshold=0.01):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    mask = importance >= threshold

    print(f"Selected features: {np.sum(mask)} out of {len(mask)}")
    feature_names = columns if not columns.empty else np.arange(X_train.shape[1])
    selected_features = [(name, importance[i]) for i, name in enumerate(feature_names) if mask[i]]
    print("Selected feature importances:")
    print("Feature Name\tImportance")
    for name, imp in selected_features:
        print(f"{name}\t{imp}")

    return X_train[:, mask], X_test[:, mask]