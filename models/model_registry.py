"""
Model registry for easy plug-and-play support.
Add new models here and use by name.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.interval_based import CanonicalIntervalForest, TimeSeriesForestClassifier
from xgboost import XGBClassifier
from aeon.classification.interval_based import QUANTClassifier

MODEL_REGISTRY = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": lambda: SVC(kernel='rbf', C=1.0, probability=True),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "XGBoost": lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "CanonicalIntervalForest": lambda: CanonicalIntervalForest(random_state=42),
    "TimeSeriesForest": lambda: TimeSeriesForestClassifier(random_state=42),
    "QUANT": lambda: QUANTClassifier(random_state=42),
    # Add more models here
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()