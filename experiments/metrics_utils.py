#!/usr/bin/env python
"""
Standardized metrics calculation for all Pyracantha invasion risk prediction models.
This ensures consistent metric reporting across different model implementations.
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    f1_score, 
    precision_recall_curve, 
    average_precision_score
)

def calculate_standard_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate a standard set of metrics for binary classification models.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    threshold : float, default=0.5
        Classification threshold for converting probabilities to binary predictions
        
    Returns:
    --------
    dict : Dictionary containing all standard metrics
    """
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Calculate sensitivity and specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # True positive rate, recall
    specificity = tn / (tn + fp)  # True negative rate
    
    # Calculate precision (positive predictive value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Return all metrics as a dictionary
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity, 
        'precision': precision,
        'average_precision': avg_precision,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the optimal classification threshold that balances sensitivity and specificity.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
        
    Returns:
    --------
    float : Optimal threshold value
    dict : Metrics at optimal threshold
    """
    # Try different thresholds and track metrics
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    best_balance = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate F1 score
        current_f1 = f1_score(y_true, y_pred)
        
        # Balance is the distance from the perfect point (sensitivity=1, specificity=1)
        balance = np.sqrt((1 - sensitivity)**2 + (1 - specificity)**2)
        
        # We want to maximize F1 score and minimize the balance distance
        if current_f1 > best_f1 and balance < best_balance * 1.1:  # Allow small trade-off
            best_f1 = current_f1
            best_threshold = threshold
            best_balance = balance
    
    # Calculate metrics at the optimal threshold
    optimal_metrics = calculate_standard_metrics(y_true, y_pred_proba, float(best_threshold))
    
    return float(best_threshold), optimal_metrics

def report_metrics_markdown(metrics, model_name, threshold=None, file_path=None):
    """
    Generate a markdown formatted report of metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_standard_metrics()
    model_name : str
        Name of the model
    threshold : float, optional
        Classification threshold used
    file_path : str, optional
        If provided, append to this file instead of returning string
    
    Returns:
    --------
    str : Markdown formatted metrics report
    """
    from datetime import datetime
    
    # Format metrics report
    report = f"\n\n## {model_name} Model Results\n\n"
    report += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    report += f"### Performance Metrics\n"
    report += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
    report += f"- **ROC AUC**: {metrics['auc']:.4f}\n"
    report += f"- **F1 Score**: {metrics['f1_score']:.4f}\n"
    report += f"- **Sensitivity**: {metrics['sensitivity']:.4f}\n"
    report += f"- **Specificity**: {metrics['specificity']:.4f}\n"
    report += f"- **Average Precision**: {metrics['average_precision']:.4f}\n"
    
    if threshold:
        report += f"- **Optimal Threshold**: {threshold:.4f}\n"
    
    cm = metrics['confusion_matrix']
    report += f"\n### Confusion Matrix\n"
    report += f"```\n[[ {cm['tn']} {cm['fp']} ]\n [ {cm['fn']} {cm['tp']} ]]\n```\n"
    
    # Either append to file or return string
    if file_path:
        # Check if file exists, create it if not
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Model Results\n")
        
        with open(file_path, 'a') as f:
            f.write(report)
        return f"Metrics appended to {file_path}"
    else:
        return report
