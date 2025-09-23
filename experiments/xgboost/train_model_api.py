#!/usr/bin/env python
"""
Train and evaluate an XGBoost model for predicting Pyracantha invasion risk.
This script trains a model using real global data and validates it on real South African data.

XGBoost advantages over Random Forest:
1. Typically better performance on structured data
2. Built-in regularization to prevent overfitting
3. Gradient boosting handles imbalanced datasets better
4. More efficient for large datasets

Note on presence/absence data:
- The script primarily uses real environmental data from occurrence records
- Since binary classification requires both presence (1) and absence (0) classes,
  if the dataset only contains presence points, the script creates a minimal set
  of background points by offsetting a subset of the real data locations
- These background points aren't claimed to be true absences, just comparison points
  with real environmental conditions from nearby areas
- If your dataset already contains both presence and absence classes, those will
  be used without modification
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

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
    Load and prepare the datasets, creating minimal background points if necessary.
    XGBoost requires both presence and absence data for binary classification.
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
        print("Creating minimal background points by using environmental locations at a distance from occurrences.")
        
        # Create a minimal set of background points by using a subset of the data with offset coordinates
        # This is more realistic than completely random points as we're using real environmental conditions
        # But we're not claiming these are true absences - just background comparison points
        background_data = global_data.sample(n=min(len(global_data), 500), random_state=42).copy()
        
        # Add offset to coordinates so they're not at the exact same locations as presence points
        # but still in similar environmental conditions (not completely random)
        rng = np.random.RandomState(42)
        background_data['latitude'] = background_data['latitude'] + rng.uniform(-0.5, 0.5, size=len(background_data))
        background_data['longitude'] = background_data['longitude'] + rng.uniform(-0.5, 0.5, size=len(background_data))
        
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
    
    # Do the same check for local validation data
    if 1 not in local_data['presence'].unique() or 0 not in local_data['presence'].unique():
        print("Creating minimal background points for local validation")
        
        background_local = local_data.sample(n=min(len(local_data), 100), random_state=42).copy()
        
        rng = np.random.RandomState(42)
        background_local['latitude'] = background_local['latitude'] + rng.uniform(-0.5, 0.5, size=len(background_local))
        background_local['longitude'] = background_local['longitude'] + rng.uniform(-0.5, 0.5, size=len(background_local))
        
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

def prepare_features(train_data, local_validation=None):
    """Prepare features for model training and evaluation."""
    # Define features for model
    feature_columns = ['latitude', 'longitude', 'elevation', 
                       'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15',
                       'sin_month', 'cos_month']
    
    # Split into features and target
    X_train = train_data[feature_columns]
    y_train = train_data['presence']
    
    if local_validation is not None:
        X_local = local_validation[feature_columns]
        y_local = local_validation['presence']
        return X_train, y_train, X_local, y_local
    
    return X_train, y_train

def train_xgboost_model(X_train, y_train):
    """Train an XGBoost model with hyperparameter tuning."""
    print("Training XGBoost model...")
    
    # Create train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    # For faster execution, we'll use a smaller grid
    small_param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    print("Performing hyperparameter tuning...")
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False,
                              eval_metric='auc')
    
    # Use smaller grid for quicker tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=small_param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1
    )
    
    grid_search.fit(X_train_split, y_train_split)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters on full training data
    best_params = grid_search.best_params_
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    
    print("Training final model with best parameters...")
    final_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return final_model

def evaluate_model(model, X_local, y_local):
    """Evaluate the model on local validation data using standardized metrics."""
    print("\nEvaluating model on local validation data...")
    
    # Get raw predictions and probabilities
    y_pred = model.predict(X_local)
    y_prob = model.predict_proba(X_local)[:, 1]
    
    # Find optimal threshold using our standardized function
    try:
        optimal_threshold, metrics = find_optimal_threshold(y_local, y_prob)
        print(f"Optimal classification threshold: {optimal_threshold:.4f}")
        
        # Extract key metrics for easy access
        accuracy = metrics['accuracy']
        auc = metrics['auc'] 
        avg_precision = metrics['average_precision']
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        f1 = metrics['f1_score']
        
        print("\nModel performance with optimal threshold:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Re-create predictions with optimal threshold
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        
        # Print classification report with optimal threshold
        print("\nClassification Report (with optimal threshold):")
        print(classification_report(y_local, y_pred_optimal))
        
    except Exception as e:
        print(f"Could not calculate optimal threshold: {e}")
        # Fallback to standard metrics with default threshold
        optimal_threshold = 0.5
        metrics = calculate_standard_metrics(y_local, y_prob)
        accuracy = metrics['accuracy']
        auc = metrics['auc']
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_local, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve on Local Validation Data')
    plt.legend(loc='lower right')
    
    # Save the plot
    roc_plot_path = os.path.join(os.path.dirname(__file__), 'roc_curve.png')
    plt.savefig(roc_plot_path)
    print(f"ROC curve saved to {roc_plot_path}")
    
    # Save the optimal threshold
    threshold_path = os.path.join(os.path.dirname(__file__), 'optimal_threshold.pkl')
    with open(threshold_path, 'wb') as f:
        pickle.dump(optimal_threshold, f)
    print(f"Optimal threshold saved to {threshold_path}")
    
    return metrics, optimal_threshold

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'])
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    
    # Save the plot
    imp_plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.savefig(imp_plot_path)
    print(f"Feature importance plot saved to {imp_plot_path}")
    
    return feat_imp

def save_model(model):
    """Save the model to disk."""
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

def append_results_to_markdown(metrics, optimal_threshold, feature_importance):
    """Append model results to the markdown file using standardized format."""
    print(f"\nAppending results to {RESULTS_PATH}...")
    
    # Use our standardized reporting function for the main metrics
    report_metrics_markdown(metrics, "XGBoost", optimal_threshold, RESULTS_PATH)
    
    # Append model-specific details like feature importance
    with open(RESULTS_PATH, 'a') as f:
        f.write("### Top Features by Importance\n")
        f.write("| Feature | Importance |\n")
        f.write("| ------- | ---------- |\n")
        for _, row in feature_importance.head(10).iterrows():
            f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
        
        f.write("\n### Model Details\n")
        f.write("- **Model**: XGBoost Classifier\n")
        f.write("- **Hyperparameters**: Tuned via GridSearchCV (best params shown in console)\n")
        f.write("- **Optimal Threshold**: {optimal_threshold:.4f}\n")
        f.write("- **Trained on**: Global dataset with background comparison points\n")
        f.write("- **Validated on**: South African dataset\n")
        f.write("- **Model file**: `experiments/xgboost/model.pkl`\n\n")
        
        f.write("### Visualization\n")
        f.write("- ROC Curve: `experiments/xgboost/roc_curve.png`\n")
        f.write("- Feature Importance: `experiments/xgboost/feature_importance.png`\n")
    
    print("Results appended successfully.")

def main():
    """Run the entire model training pipeline using standardized metrics."""
    print("=== XGBoost Model Training Pipeline ===")
    
    # Load and prepare data
    train_data, local_validation = load_data()
    
    # Use the prepare_features function but handle the return values correctly
    result = prepare_features(train_data, local_validation)
    if len(result) == 4:
        X_train, y_train, X_local, y_local = result
    else:
        X_train, y_train = result
        X_local, y_local = None, None
    
    # Train model
    model = train_xgboost_model(X_train, y_train)
    
    # Evaluate model (if we have local validation data)
    if X_local is not None and y_local is not None:
        metrics, optimal_threshold = evaluate_model(model, X_local, y_local)
    else:
        # If no local validation data, use training data with a warning
        print("WARNING: No local validation data available for evaluation.")
        print("Using training data for metrics calculation - results may be overly optimistic.")
        y_train_prob = model.predict_proba(X_train)[:, 1]
        metrics, optimal_threshold = find_optimal_threshold(y_train, y_train_prob)
    
    # Plot feature importance
    feature_importance = plot_feature_importance(model, X_train.columns)
    
    # Save model
    save_model(model)
    
    # Append results to markdown using standardized format
    append_results_to_markdown(metrics, optimal_threshold, feature_importance)
    
    print("\n=== XGBoost Model Training Complete ===")

if __name__ == "__main__":
    main()
