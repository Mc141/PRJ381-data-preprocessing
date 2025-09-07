#!/usr/bin/env python
"""
Train and evaluate a simplified Ensemble model for predicting Pyracantha invasion risk.
This script creates a voting ensemble that combines:

1. XGBoost with optimized parameters
2. Random Forest with optimized parameters
3. Optimal threshold selection for balanced predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
from datetime import datetime

# Import standardized metrics calculation utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics_utils import calculate_standard_metrics, find_optimal_threshold, report_metrics_markdown

# Add the root project directory to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
GLOBAL_DATA_PATH = os.path.join(DATA_DIR, 'global_training_ml_ready.csv')
LOCAL_DATA_PATH = os.path.join(DATA_DIR, 'local_validation_ml_ready.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'MODEL_RESULTS.md')

def load_data():
    """
    Load and prepare the datasets, creating stratified background points.
    """
    print("Loading datasets...")
    
    # Load datasets
    global_data = pd.read_csv(GLOBAL_DATA_PATH)
    local_data = pd.read_csv(LOCAL_DATA_PATH)
    
    print(f"Global dataset shape: {global_data.shape}")
    print(f"Local validation dataset shape: {local_data.shape}")
    
    # Ensure 'presence' column exists
    if 'presence' not in global_data.columns:
        print("'presence' column not found in global data, creating it")
        global_data['presence'] = 1
        
    if 'presence' not in local_data.columns:
        print("'presence' column not found in local data, creating it")
        local_data['presence'] = 1
    
    # Check if we have any absence data (class 0)
    if 1 not in global_data['presence'].unique() or 0 not in global_data['presence'].unique():
        print("Warning: Binary classification requires both presence (1) and absence (0) classes.")
        print("Creating background points using stratified environmental sampling...")
        
        # Create stratified background points from environmental gradients
        background_data1 = global_data.sort_values('bio1').iloc[::3, :].copy()  # Temperature
        background_data2 = global_data.sort_values('bio12').iloc[::3, :].copy() # Precipitation
        background_data3 = global_data.sort_values('elevation').iloc[::3, :].copy() # Elevation
        
        # Combine these stratified samples
        background_data = pd.concat([background_data1, background_data2, background_data3], 
                                  ignore_index=True).drop_duplicates()
        
        # Limit to a reasonable number
        if len(background_data) > 500:
            background_data = background_data.sample(n=500, random_state=42)
        
        # Add small random offset to coordinates
        rng = np.random.RandomState(42)
        background_data['latitude'] = background_data['latitude'] + rng.uniform(-0.3, 0.3, size=len(background_data))
        background_data['longitude'] = background_data['longitude'] + rng.uniform(-0.3, 0.3, size=len(background_data))
        
        # Clip to realistic bounds
        background_data['latitude'] = np.clip(background_data['latitude'], -90, 90)
        background_data['longitude'] = np.clip(background_data['longitude'], -180, 180)
        background_data['presence'] = 0
        
        print(f"Created {len(background_data)} background comparison points")
        
        # Combine datasets
        train_data = pd.concat([global_data, background_data], ignore_index=True)
    else:
        print("Using provided presence/absence data without modification")
        train_data = global_data
    
    # Do the same for local validation data
    if 1 not in local_data['presence'].unique() or 0 not in local_data['presence'].unique():
        print("Creating background points for local validation")
        
        # Same stratified approach for local data
        background_local1 = local_data.sort_values('bio1').iloc[::3, :].copy()
        background_local2 = local_data.sort_values('bio12').iloc[::3, :].copy()
        background_local3 = local_data.sort_values('elevation').iloc[::3, :].copy()
        
        background_local = pd.concat([background_local1, background_local2, background_local3], 
                                    ignore_index=True).drop_duplicates()
        
        if len(background_local) > 100:
            background_local = background_local.sample(n=100, random_state=42)
        
        # Add small random offset
        rng = np.random.RandomState(42)
        background_local['latitude'] = background_local['latitude'] + rng.uniform(-0.3, 0.3, size=len(background_local))
        background_local['longitude'] = background_local['longitude'] + rng.uniform(-0.3, 0.3, size=len(background_local))
        
        # Clip to South Africa bounds
        background_local['latitude'] = np.clip(background_local['latitude'], -35, -22)
        background_local['longitude'] = np.clip(background_local['longitude'], 16, 33)
        background_local['presence'] = 0
        
        # Combine datasets
        local_validation = pd.concat([local_data, background_local], ignore_index=True)
    else:
        local_validation = local_data
        
    print(f"Final training data shape: {train_data.shape} with {train_data['presence'].sum()} presence and {len(train_data) - train_data['presence'].sum()} background points")
    print(f"Final validation data shape: {local_validation.shape} with {local_validation['presence'].sum()} presence and {len(local_validation) - local_validation['presence'].sum()} background points")
    
    return train_data, local_validation

def engineer_features(train_data, local_validation):
    """
    Create advanced features for both training and validation data
    """
    print("\nPerforming feature engineering...")
    
    # Create copies to avoid modifying the original data
    train_features = train_data.copy()
    val_features = local_validation.copy()
    
    # 1. Climate indices and ecological features
    for df in [train_features, val_features]:
        # Temperature range
        df['temp_range'] = df['bio5'] - df['bio6']
        
        # Aridity index
        df['aridity_index'] = df['bio12'] / (df['bio1'] + 10)
        
        # Precipitation seasonality
        df['precip_seasonality_ratio'] = df['bio13'] / (df['bio14'] + 1)
        
        # Growing degree days approximation
        df['growing_degree_approx'] = np.maximum(0, df['bio1'] - 5) * (1 - 0.1 * df['bio4'] / 100)
    
    # 2. Distance-based features
    # Calculate median lat/long of presence points
    presence_points = train_data[train_data['presence'] == 1]
    median_lat = presence_points['latitude'].median()
    median_lon = presence_points['longitude'].median()
    
    # Distance from median presence point
    train_features['dist_from_median'] = np.sqrt(
        (train_features['latitude'] - median_lat)**2 + 
        (train_features['longitude'] - median_lon)**2
    )
    
    val_features['dist_from_median'] = np.sqrt(
        (val_features['latitude'] - median_lat)**2 + 
        (val_features['longitude'] - median_lon)**2
    )
    
    # 3. Interaction terms
    for df in [train_features, val_features]:
        # Temperature-precipitation interactions
        df['temp_precip'] = df['bio1'] * df['bio12'] / 1000
        df['elev_temp'] = df['elevation'] * df['bio1'] / 1000
        
        # Selected key interactions (using only a subset to avoid too many features)
        df['bio1_x_bio12'] = df['bio1'] * df['bio12'] / 100
        df['bio1_x_bio4'] = df['bio1'] * df['bio4'] / 100
        df['bio4_x_bio12'] = df['bio4'] * df['bio12'] / 100
        df['bio5_x_bio13'] = df['bio5'] * df['bio13'] / 100
        df['bio6_x_bio14'] = df['bio6'] * df['bio14'] / 100
        df['bio13_x_bio15'] = df['bio13'] * df['bio15'] / 100
    
    print(f"Extended feature set: {len(train_features.columns) - 1} features (from original {len(train_data.columns) - 1})")
    print(f"New features added: {len(train_features.columns) - len(train_data.columns)}")
    
    return train_features, val_features

# Using the imported standardized function from metrics_utils.py instead
# def find_optimal_threshold(y_true, y_pred_proba):
#     """
#     Find the optimal classification threshold using Youden's J statistic.
#     J = sensitivity + specificity - 1
#     """
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
#     j_scores = tpr - fpr
#     optimal_idx = np.argmax(j_scores)
#     optimal_threshold = thresholds[optimal_idx]
#     return optimal_threshold

def train_and_evaluate_ensemble():
    """
    Main function to train and evaluate a voting ensemble.
    """
    print("=== Ensemble Model Training Pipeline ===")
    
    # 1. Load and prepare data
    train_data, local_validation = load_data()
    
    # 2. Apply feature engineering
    train_features, val_features = engineer_features(train_data, local_validation)
    
    # 3. Prepare training and validation sets
    X_train = train_features.drop(['presence'], axis=1)
    y_train = train_features['presence']
    X_val = val_features.drop(['presence'], axis=1)
    y_val = val_features['presence']
    
    # 4. Create optimized base models
    print("\nCreating and training ensemble model...")
    
    # XGBoost with optimized parameters
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.01,
        n_estimators=200,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False
    )
    
    # Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Create and train voting ensemble
    voting_ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model)
        ],
        voting='soft',  # Use probability estimates
        n_jobs=-1
    )
    
    # Train the ensemble
    voting_ensemble.fit(X_train, y_train)
    
    # 6. Evaluate the ensemble
    print("\nEvaluating ensemble model...")
    y_pred = voting_ensemble.predict(X_val)
    
    # Get class 1 probabilities 
    y_prob = voting_ensemble.predict_proba(X_val)
    y_prob_class1 = y_prob[:, 1]
    
    # Calculate initial metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob_class1)
    
    print(f"Initial accuracy: {accuracy:.4f}")
    print(f"Initial ROC AUC: {auc:.4f}")
    
    # 7. Find optimal threshold and calculate standard metrics using our utility function
    optimal_threshold, metrics = find_optimal_threshold(y_val, y_prob_class1)
    print(f"Optimal classification threshold: {optimal_threshold:.4f}")
    
    # Extract metrics for ease of use
    accuracy = metrics['accuracy']
    auc = metrics['auc']
    avg_precision = metrics['average_precision']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    f1 = metrics['f1_score']
    tn = metrics['confusion_matrix']['tn']
    fp = metrics['confusion_matrix']['fp']
    fn = metrics['confusion_matrix']['fn']
    tp = metrics['confusion_matrix']['tp']
    
    # Create y_pred_optimal for consistency with rest of code
    y_pred_optimal = (y_prob_class1 >= optimal_threshold).astype(int)
    
    print("\nFinal model performance with optimal threshold:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred_optimal))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_optimal))
    
    # 9. Generate ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_val, y_prob_class1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Voting Ensemble')
    plt.legend(loc='lower right')
    roc_path = os.path.join(os.path.dirname(__file__), 'roc_curve.png')
    plt.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")
    plt.close()
    
    # 10. Generate precision-recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_val, y_prob_class1)
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Voting Ensemble')
    plt.legend(loc='upper right')
    pr_path = os.path.join(os.path.dirname(__file__), 'precision_recall_curve.png')
    plt.savefig(pr_path)
    print(f"Precision-Recall curve saved to {pr_path}")
    plt.close()
    
    # 11. Save the ensemble model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(voting_ensemble, f)
    print("Model saved successfully.")
    
    # 12. Save the optimal threshold separately
    threshold_path = os.path.join(os.path.dirname(__file__), 'optimal_threshold.pkl')
    with open(threshold_path, 'wb') as f:
        pickle.dump(optimal_threshold, f)
    print(f"Optimal threshold saved to {threshold_path}")
    
    # 13. Get feature importances from base models
    if hasattr(voting_ensemble.estimators_[1], 'feature_importances_'):
        # Use RF feature importances as it's more interpretable
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': voting_ensemble.estimators_[1].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
    else:
        feature_importance = pd.DataFrame()
    
    # 14. Add results to MODEL_RESULTS.md using our standardized reporting function
    print(f"\nAppending results to {RESULTS_PATH}...")
    
    # Create metrics dictionary matching our standardized format
    standard_metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'average_precision': avg_precision,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }
    
    # Use our standardized reporting function
    report_metrics_markdown(standard_metrics, "Ensemble", optimal_threshold, RESULTS_PATH)
    
    # Manually append feature importance and model details to the results file
    with open(RESULTS_PATH, 'a') as f:
        if not feature_importance.empty:
            f.write(f"### Top Features by Importance\n")
            f.write("| Feature | Importance |\n")
            f.write("| ------- | ---------- |\n")
            for _, row in feature_importance.iterrows():
                f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
            f.write("\n")
        
        f.write(f"### Model Details\n")
        f.write(f"- **Model**: Voting Ensemble of XGBoost and Random Forest\n")
        f.write(f"- **Optimal Threshold**: {optimal_threshold:.4f}\n")
        f.write(f"- **Trained on**: Global dataset with environmentally stratified background points\n")
        f.write(f"- **Validated on**: South African dataset\n")
        f.write(f"- **Model file**: `experiments/ensemble/model.pkl`\n\n")
        
        f.write(f"### Visualization\n")
        f.write(f"- ROC Curve: `experiments/ensemble/roc_curve.png`\n")
        f.write(f"- Precision-Recall: `experiments/ensemble/precision_recall_curve.png`\n")
    
    print("Results appended successfully.")
    print("\n=== Ensemble Model Training Complete ===")
    
    return voting_ensemble, optimal_threshold

if __name__ == "__main__":
    train_and_evaluate_ensemble()
