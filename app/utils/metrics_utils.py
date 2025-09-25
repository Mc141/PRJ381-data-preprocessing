#!/usr/bin/env python
"""
Standardized metrics calculation for all Pyracantha invasion risk prediction models.
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)


def calculate_standard_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate a standard set of metrics for binary classification models."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "average_precision": avg_precision,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_optimal_threshold(y_true, y_pred_proba):
    """Find the optimal classification threshold with balance between sensitivity and specificity."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    best_balance = float("inf")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        current_f1 = f1_score(y_true, y_pred)
        balance = np.sqrt((1 - sensitivity) ** 2 + (1 - specificity) ** 2)
        if current_f1 > best_f1 and balance < best_balance * 1.1:
            best_f1 = current_f1
            best_threshold = threshold
            best_balance = balance
    optimal_metrics = calculate_standard_metrics(y_true, y_pred_proba, float(best_threshold))
    return float(best_threshold), optimal_metrics


def report_metrics_markdown(metrics, model_name, threshold=None, file_path=None):
    """Generate a markdown formatted report of metrics and append to file if provided."""
    from datetime import datetime

    report = f"\n\n## {model_name} Model Results\n\n"
    report += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report += f"### Performance Metrics\n"
    report += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
    report += f"- **ROC AUC**: {metrics['auc']:.4f}\n"
    report += f"- **F1 Score**: {metrics['f1_score']:.4f}\n"
    report += f"- **Sensitivity**: {metrics['sensitivity']:.4f}\n"
    report += f"- **Specificity**: {metrics['specificity']:.4f}\n"
    report += f"- **Average Precision**: {metrics['average_precision']:.4f}\n"
    if threshold is not None:
        report += f"- **Optimal Threshold**: {threshold:.4f}\n"
    cm = metrics["confusion_matrix"]
    report += f"\n### Confusion Matrix\n"
    report += f"```\n[[ {cm['tn']} {cm['fp']} ]]\n[ [ {cm['fn']} {cm['tp']} ]]\n```\n"
    if file_path:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Model Results\n")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(report)
        return f"Metrics appended to {file_path}"
    return report
