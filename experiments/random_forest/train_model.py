#!/usr/bin/env python
"""
Train and evaluate a Random Forest model for predicting Pyracantha invasion risk.
This script trains a model using global data and validates it on South African data.
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
from sklearn.model_selection import train_test_split
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
    """Load and prepare the datasets."""
    print("Loading datasets...")
    
    # Load datasets
    global_data = pd.read_csv(GLOBAL_DATA_PATH)
    local_data = pd.read_csv(LOCAL_DATA_PATH)
    
    print(f"Global dataset shape: {global_data.shape}")
    print(f"Local validation dataset shape: {local_data.shape}")
    
    # Since we only have presence data, we'll generate pseudo-absence data
    # using geographic and climate constraints for global training
    print("Generating pseudo-absence data for global training...")
    
    # Create a copy of global data and randomly adjust coordinates
    # to create "absence" points
    absence_data = global_data.copy()
    
    # Add random offset to coordinates (between 1-3 degrees)
    rng = np.random.RandomState(42)
    absence_data['latitude'] = absence_data['latitude'] + rng.uniform(-3, 3, size=len(absence_data))
    absence_data['longitude'] = absence_data['longitude'] + rng.uniform(-3, 3, size=len(absence_data))
    
    # Clip to realistic bounds
    absence_data['latitude'] = np.clip(absence_data['latitude'], -90, 90)
    absence_data['longitude'] = np.clip(absence_data['longitude'], -180, 180)
    
    # Create target variable (1 for presence, 0 for absence)
    global_data['presence'] = 1
    absence_data['presence'] = 0
    
    # Combine datasets
    train_data = pd.concat([global_data, absence_data], ignore_index=True)
    
    # For local validation data, we'll create pseudo-absence points specific to South Africa
    print("Generating pseudo-absence data for local validation...")
    
    local_absence = local_data.copy()
    local_absence['latitude'] = local_absence['latitude'] + rng.uniform(-1, 1, size=len(local_absence))
    local_absence['longitude'] = local_absence['longitude'] + rng.uniform(-1, 1, size=len(local_absence))
    
    # Clip to South Africa approximate bounds
    local_absence['latitude'] = np.clip(local_absence['latitude'], -35, -22)
    local_absence['longitude'] = np.clip(local_absence['longitude'], 16, 33)
    
    local_data['presence'] = 1
    local_absence['presence'] = 0
    
    # Combine local datasets
    local_validation = pd.concat([local_data, local_absence], ignore_index=True)
    
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
    """Train the Random Forest model."""
    print("\nTraining Random Forest model...")
    
    # Split global data into train/test sets
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train_split, y_train_split)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test_split)
    y_prob = model.predict_proba(X_test_split)[:, 1]
    
    # Performance metrics
    accuracy = accuracy_score(y_test_split, y_pred)
    auc = roc_auc_score(y_test_split, y_prob)
    
    print(f"Global model accuracy: {accuracy:.4f}")
    print(f"Global model AUC: {auc:.4f}")
    
    # Calculate feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame(
        {'Feature': X_train.columns, 'Importance': importances}
    ).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, feature_importance

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
    
    return accuracy, auc, y_local_pred, y_local_prob

def save_model_and_results(model, feature_importance, global_accuracy, global_auc, 
                         local_accuracy, local_auc):
    """Save the trained model and results."""
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Update results markdown file
    with open(RESULTS_PATH, 'a') as f:
        f.write("\n\n## Random Forest Model Results\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### Model Parameters\n")
        f.write("- Model Type: Random Forest\n")
        f.write("- Number of Trees: 100\n")
        f.write("- Max Depth: 15\n")
        f.write("- Min Samples Split: 10\n")
        f.write("- Min Samples Leaf: 5\n\n")
        
        f.write("### Performance Metrics\n")
        f.write(f"- Global Training Accuracy: {global_accuracy:.4f}\n")
        f.write(f"- Global Training AUC: {global_auc:.4f}\n")
        f.write(f"- South Africa Validation Accuracy: {local_accuracy:.4f}\n")
        if not np.isnan(local_auc):
            f.write(f"- South Africa Validation AUC: {local_auc:.4f}\n")
        
        f.write("\n### Feature Importance\n")
        f.write("| Feature | Importance |\n")
        f.write("| ------- | ---------- |\n")
        for _, row in feature_importance.iterrows():
            f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
    
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

def main():
    """Main function to train and evaluate the model."""
    print("Starting Random Forest model training pipeline...\n")
    
    # Load data
    train_data, local_validation = load_data()
    
    # Prepare features
    X_train, y_train, X_local, y_local = prepare_features(train_data, local_validation)
    
    # Train model
    model, feature_importance = train_model(X_train, y_train)
    
    # Evaluate on local data
    local_accuracy, local_auc, _, _ = evaluate_local_performance(model, X_local, y_local)
    
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
        local_auc
    )
    
    # Plot feature importance
    plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plot_feature_importance(feature_importance, plot_path)
    
    print("\nRandom Forest model training pipeline completed!")

if __name__ == "__main__":
    main()
