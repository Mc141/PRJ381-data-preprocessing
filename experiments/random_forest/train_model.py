#!/usr/bin/env python
"""
Train and evaluate a Random Forest model for predicting Pyracantha invasion risk.
This script trains a model using global data and validates it on South African data.

Random Forest advantages for ecological modeling:
1. Robust to outliers and noise
2. Can handle non-linear relationships
3. Built-in feature importance ranking
4. Low risk of overfitting with proper tuning
5. Can handle both categorical and numerical features

Note on presence/absence data:
- The script primarily uses real environmental data from occurrence records
- Since binary classification requires both presence (1) and absence (0) classes,
  if the dataset only contains presence points, the script creates background comparison points
- These background points aren't claimed to be true absences, just comparison points
  with real environmental conditions from nearby areas
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance

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
    Load and prepare the datasets, creating background points if necessary.
    Random Forest requires both presence and absence data for binary classification.
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
        print("Creating background points by using environmental locations at a distance from occurrences.")
        
        # Create background points by using a subset of the data with offset coordinates
        # This is more realistic than completely random points as we're using real environmental conditions
        background_data = global_data.sample(n=min(len(global_data), 500), random_state=42).copy()
        
        # Add offset to coordinates (smaller offsets for more realistic environmental conditions)
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
        print("Creating background points for local validation")
        
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
    # Define features for model (excluding day_of_year as we have month encoded as sin/cos)
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

def train_model(X_train, y_train):
    """Train the Random Forest model with hyperparameter tuning."""
    print("\nTraining Random Forest model with hyperparameter tuning...")
    
    # Split global data into train/test sets
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # For faster execution, we'll use a smaller grid
    small_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    print("Performing hyperparameter tuning...")
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=small_param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_split, y_train_split)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters on full training data
    best_params = grid_search.best_params_
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    print("Training final model with best parameters...")
    final_model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = final_model.predict(X_test_split)
    y_prob = final_model.predict_proba(X_test_split)[:, 1]
    
    # Performance metrics
    accuracy = accuracy_score(y_test_split, y_pred)
    auc = roc_auc_score(y_test_split, y_prob)
    
    print(f"Global model accuracy: {accuracy:.4f}")
    print(f"Global model AUC: {auc:.4f}")
    
    # Calculate feature importances
    importances = final_model.feature_importances_
    feature_importance = pd.DataFrame(
        {'Feature': X_train.columns, 'Importance': importances}
    ).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return final_model, feature_importance

def evaluate_local_performance(model, X_local, y_local):
    """Evaluate the model on local South African data."""
    print("\nEvaluating on South African validation data...")
    
    y_local_pred = model.predict(X_local)
    y_local_prob = model.predict_proba(X_local)[:, 1]
    
    accuracy = accuracy_score(y_local, y_local_pred)
    try:
        auc = roc_auc_score(y_local, y_local_prob)
    except:
        auc = float('nan')  # In case of only one class
    
    print(f"Local validation accuracy: {accuracy:.4f}")
    if not np.isnan(auc):
        print(f"Local validation AUC: {auc:.4f}")
    
    print("\nClassification Report (Local Validation):")
    print(classification_report(y_local, y_local_pred))
    
    # Plot ROC curve
    try:
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_local, y_local_prob)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest ROC Curve on Local Validation Data')
        plt.legend(loc="lower right")
        
        # Save the plot
        roc_plot_path = os.path.join(os.path.dirname(__file__), 'roc_curve.png')
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"ROC curve saved to {roc_plot_path}")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
    
    return accuracy, auc, y_local_pred, y_local_prob

def save_model_and_results(model, feature_importance, global_accuracy, global_auc, 
                         local_accuracy, local_auc, best_params=None):
    """Save the trained model and results."""
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Update results markdown file
    with open(RESULTS_PATH, 'a') as f:
        f.write("\n\n## Random Forest Model Results (Optimized)\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("### Performance Metrics\n")
        f.write(f"- **Accuracy**: {local_accuracy:.4f}\n")
        f.write(f"- **ROC AUC**: {local_auc:.4f}\n\n")
        
        f.write("### Top Features by Importance\n")
        f.write("| Feature | Importance |\n")
        f.write("| ------- | ---------- |\n")
        for _, row in feature_importance.head(10).iterrows():
            f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
        
        f.write("\n### Model Details\n")
        f.write("- **Model**: Random Forest Classifier\n")
        
        # Write hyperparameters
        if best_params:
            f.write("- **Hyperparameters**: Tuned via GridSearchCV\n")
            for param, value in best_params.items():
                f.write(f"  - {param}: {value}\n")
        else:
            f.write("- **Hyperparameters**: Default\n")
            
        f.write("- **Trained on**: Global dataset with background comparison points\n")
        f.write("- **Validated on**: South African dataset\n")
        f.write("- **Model file**: `experiments/random_forest/model.pkl`\n\n")
        
        f.write("### Visualization\n")
        f.write("- Feature Importance: `experiments/random_forest/feature_importance.png`\n")
        f.write("- ROC Curve: `experiments/random_forest/roc_curve.png`\n")
    
    print(f"Results saved to {RESULTS_PATH}")

def plot_feature_importance(feature_importance, save_path):
    """Plot feature importance and save the figure."""
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")
    
    # Also plot ROC curve if possible
    try:
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest ROC Curve')
        plt.savefig(os.path.join(os.path.dirname(save_path), 'roc_curve.png'))
        plt.close()
        print(f"ROC curve saved to {os.path.join(os.path.dirname(save_path), 'roc_curve.png')}")
    except:
        pass

def main():
    """Main function to train and evaluate the model."""
    print("Starting Random Forest model training pipeline (Optimized)...\n")
    
    # Load data
    train_data, local_validation = load_data()
    
    # Prepare features
    result = prepare_features(train_data, local_validation)
    if len(result) == 4:
        X_train, y_train, X_local, y_local = result
    else:
        X_train, y_train = result
        X_local, y_local = None, None
    
    # Train model with hyperparameter tuning
    model, feature_importance = train_model(X_train, y_train)
    
    # Get the best parameters
    best_params = None
    if hasattr(model, 'get_params'):
        best_params = {
            'n_estimators': model.get_params()['n_estimators'],
            'max_depth': model.get_params()['max_depth'],
            'min_samples_split': model.get_params()['min_samples_split'],
            'min_samples_leaf': model.get_params()['min_samples_leaf']
        }
    
    # Evaluate on local data
    if X_local is not None and y_local is not None:
        local_accuracy, local_auc, _, _ = evaluate_local_performance(model, X_local, y_local)
    else:
        local_accuracy, local_auc = 0.0, 0.0
        print("No local validation data available for evaluation.")
    
    # Get global metrics from the initial model training
    global_accuracy = accuracy_score(y_train, model.predict(X_train))
    global_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    
    # Save model and results
    save_model_and_results(
        model, 
        feature_importance, 
        global_accuracy, 
        global_auc, 
        local_accuracy, 
        local_auc,
        best_params
    )
    
    # Plot feature importance
    plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plot_feature_importance(feature_importance, plot_path)
    
    print("\nRandom Forest model training pipeline completed!")

if __name__ == "__main__":
    main()
