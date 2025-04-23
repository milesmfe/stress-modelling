"""
Evaluation and metrics saving module.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from utils.logger import logger

def save_results(y_true, y_pred, model_name, fold_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(out_dir, f"fold{fold_idx}_{model_name}_report_{timestamp}.csv"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Fold {fold_idx} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(out_dir, f"fold{fold_idx}_{model_name}_cm_{timestamp}.png"))
    plt.close()

    logger.info(f"Saved fold {fold_idx} results to {out_dir}")