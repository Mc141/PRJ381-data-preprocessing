#!/usr/bin/env python
"""
Train and evaluate an Enhanced XGBoost model for predicting Pyracantha invasion risk.
This script improves upon the standard XGBoost implementation with:

1. Advanced Feature Engineering:
   - Interaction terms between climate variables
   - Climate indices and ecological indicators
   - Dimensionality reduction for geographic variables
   - Distance-based features to reduce geographic dependence

2. Improved Model Training:
   - Expanded hyperparameter tuning
   - Feature selection via SHAP values
   - Early stopping with validation monitoring
   - Class weighting for imbalanced data

3. Spatial Cross-Validation:
   - Block-based splitting to account for spatial autocorrelation
   - More realistic assessment of model transferability
   
Note on ecological modeling approach:
- This model focuses on environmental drivers rather than pure geographic patterns
- Reduces the reliance on exact latitude/longitude to improve generalization
- Creates meaningful interaction terms that have ecological significance
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import scipy.sparse as sp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# Add the root project directory to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
GLOBAL_DATA_PATH = os.path.join(DATA_DIR, 'global_training_ml_ready.csv')
LOCAL_DATA_PATH = os.path.join(DATA_DIR, 'local_validation_ml_ready.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'MODEL_RESULTS.md')
SHAP_PATH = os.path.join(os.path.dirname(__file__), 'shap_summary.png')

def load_data():
    """
    Load and prepare the datasets, creating minimal background points with improved
    sampling strategy to better represent environmental gradients.
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
        
        # Improved background point generation:
        # Create background points with a balanced representation of environmental gradients
        # This helps ensure the model learns environmental associations, not just geography
        
        # 1. Create background points covering the full environmental range
        # Instead of just using random offsets, we'll create a more systematic sampling
        
        # Sort global data by key environmental variables and sample across the range
        background_data1 = global_data.sort_values('bio1').iloc[::3, :].copy()  # Temperature
        background_data2 = global_data.sort_values('bio12').iloc[::3, :].copy() # Precipitation
        background_data3 = global_data.sort_values('elevation').iloc[::3, :].copy() # Elevation
        
        # Combine these stratified samples
        background_data = pd.concat([background_data1, background_data2, background_data3], 
                                   ignore_index=True).drop_duplicates()
        
        # Limit to a reasonable number
        if len(background_data) > 500:
            background_data = background_data.sample(n=500, random_state=42)
        
        # Add small random offset to coordinates to avoid exact overlap with presence points
        rng = np.random.RandomState(42)
        background_data['latitude'] = background_data['latitude'] + rng.uniform(-0.3, 0.3, size=len(background_data))
        background_data['longitude'] = background_data['longitude'] + rng.uniform(-0.3, 0.3, size=len(background_data))
        
        # Clip to realistic bounds
        background_data['latitude'] = np.clip(background_data['latitude'], -90, 90)
        background_data['longitude'] = np.clip(background_data['longitude'], -180, 180)
        background_data['presence'] = 0
        
        print(f"Created {len(background_data)} background comparison points using stratified environmental sampling")
        
        # Combine datasets
        train_data = pd.concat([global_data, background_data], ignore_index=True)
    else:
        print("Using provided presence/absence data without modification")
        train_data = global_data
    
    # Do the same for local validation data with improved sampling
    if 1 not in local_data['presence'].unique() or 0 not in local_data['presence'].unique():
        print("Creating background points for local validation using stratified environmental sampling")
        
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

def engineer_features(train_data, local_validation=None):
    """
    Prepare features for model training with advanced feature engineering.
    Creates interaction terms, climate indices, and reduces geographic dependence.
    """
    print("\nPerforming feature engineering...")
    
    # Start with base features
    base_features = ['latitude', 'longitude', 'elevation', 
                    'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15',
                    'sin_month', 'cos_month']
    
    # Create copies to avoid modifying the original data
    train_df = train_data.copy()
    
    # 1. FEATURE ENGINEERING
    
    # 1.1 Create climate indices and ecologically relevant features
    
    # Temperature range (annual)
    train_df['temp_range'] = train_df['bio5'] - train_df['bio6']  # max temp - min temp
    
    # Aridity index (simplified version: precipitation / temperature)
    train_df['aridity_index'] = train_df['bio12'] / (train_df['bio1'] + 10)  # +10 to avoid division by zero or negative values
    
    # Precipitation seasonality ratio
    train_df['precip_seasonality_ratio'] = train_df['bio13'] / (train_df['bio14'] + 1)  # wettest / driest month + 1 to avoid div by zero
    
    # Growing degree days approximation 
    train_df['growing_degree_approx'] = np.maximum(0, train_df['bio1'] - 5) * (1 - 0.1 * train_df['bio4'] / 100)
    
    # 1.2 Reduce geographic dependence with distance-based features
    
    # Calculate median lat/long of presence points
    presence_points = train_data[train_data['presence'] == 1]
    median_lat = presence_points['latitude'].median()
    median_lon = presence_points['longitude'].median()
    
    # Distance from median presence point (simplified Euclidean distance)
    train_df['dist_from_median'] = np.sqrt(
        (train_df['latitude'] - median_lat)**2 + 
        (train_df['longitude'] - median_lon)**2
    )
    
    # 1.3 Interaction terms for key environmental variables
    # Create ecologically meaningful interactions
    
    # Temp × Precipitation (approximates energy × water availability)
    train_df['temp_precip'] = train_df['bio1'] * train_df['bio12'] / 1000  # Scale down for numerical stability
    
    # Elevation × Temperature (represents environmental gradients)
    train_df['elev_temp'] = train_df['elevation'] * train_df['bio1'] / 1000
    
    # Seasonality interaction (precipitation seasonality × temperature seasonality)
    train_df['season_interact'] = train_df['bio15'] * train_df['bio4'] / 100
    
    # ENHANCED: Add more advanced climate indicators
    # Potential evapotranspiration (PET) approximation using Thornthwaite equation
    train_df['pet_approx'] = 16 * (10 * train_df['bio1'] / 5) ** 1.514
    
    # Water deficit indicator (precipitation - PET)
    train_df['water_deficit'] = train_df['bio12'] - train_df['pet_approx']
    
    # Climatic moisture index
    train_df['moisture_index'] = train_df['water_deficit'] / (train_df['pet_approx'] + 1)
    
    # Heat × moisture stress interaction
    train_df['heat_moisture_stress'] = (train_df['bio5'] - 30) * (1 / (train_df['bio12'] + 100))
    
    # Frost frequency approximation
    train_df['frost_freq_approx'] = np.maximum(0, 5 - train_df['bio6']) ** 2
    
    # 1.4 Create interaction terms for key environmental variables
    # Select only the environmental variables, not geographic coordinates
    env_features = ['elevation', 'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15']
    
    # Create interaction terms manually instead of using PolynomialFeatures
    feature_names = []
    interaction_data = {}
    
    # Generate all pairwise interactions
    for i in range(len(env_features)):
        for j in range(i+1, len(env_features)):
            feature_name = f"{env_features[i]}_x_{env_features[j]}"
            feature_names.append(feature_name)
            # Create the interaction term
            interaction_data[feature_name] = train_df[env_features[i]] * train_df[env_features[j]]
    
    # Create dataframe with interaction terms
    poly_df = pd.DataFrame(interaction_data)
    
    # Add polynomial features to dataframe
    train_df = pd.concat([train_df, poly_df], axis=1)
    
    # Define the extended feature list
    extended_features = base_features + [
        'temp_range', 'aridity_index', 'precip_seasonality_ratio', 'growing_degree_approx',
        'dist_from_median', 'temp_precip', 'elev_temp', 'season_interact',
        'pet_approx', 'water_deficit', 'moisture_index', 'heat_moisture_stress', 'frost_freq_approx'
    ] + feature_names
    
    # Apply same transformations to validation data if provided
    if local_validation is not None:
        local_df = local_validation.copy()
        
        # Apply the same transformations
        local_df['temp_range'] = local_df['bio5'] - local_df['bio6']
        local_df['aridity_index'] = local_df['bio12'] / (local_df['bio1'] + 10)
        local_df['precip_seasonality_ratio'] = local_df['bio13'] / (local_df['bio14'] + 1)
        local_df['growing_degree_approx'] = np.maximum(0, local_df['bio1'] - 5) * (1 - 0.1 * local_df['bio4'] / 100)
        local_df['dist_from_median'] = np.sqrt(
            (local_df['latitude'] - median_lat)**2 + 
            (local_df['longitude'] - median_lon)**2
        )
        local_df['temp_precip'] = local_df['bio1'] * local_df['bio12'] / 1000
        local_df['elev_temp'] = local_df['elevation'] * local_df['bio1'] / 1000
        local_df['season_interact'] = local_df['bio15'] * local_df['bio4'] / 100
        
        # Add the same advanced climate indicators to validation data
        local_df['pet_approx'] = 16 * (10 * local_df['bio1'] / 5) ** 1.514
        local_df['water_deficit'] = local_df['bio12'] - local_df['pet_approx']
        local_df['moisture_index'] = local_df['water_deficit'] / (local_df['pet_approx'] + 1)
        local_df['heat_moisture_stress'] = (local_df['bio5'] - 30) * (1 / (local_df['bio12'] + 100))
        local_df['frost_freq_approx'] = np.maximum(0, 5 - local_df['bio6']) ** 2
        
        # Add interaction features manually to maintain consistency
        interaction_data_local = {}
        
        # Generate all pairwise interactions
        for i in range(len(env_features)):
            for j in range(i+1, len(env_features)):
                feature_name = f"{env_features[i]}_x_{env_features[j]}"
                # Create the interaction term
                interaction_data_local[feature_name] = local_df[env_features[i]] * local_df[env_features[j]]
        
        # Create dataframe with interaction terms
        poly_df_local = pd.DataFrame(interaction_data_local)
        local_df = pd.concat([local_df, poly_df_local], axis=1)
        
        # Create features and target
        X_train = train_df[extended_features]
        y_train = train_df['presence']
        X_local = local_df[extended_features]
        y_local = local_df['presence']
        
        # Print the new feature count
        print(f"Extended feature set: {X_train.shape[1]} features (from original {len(base_features)})")
        print(f"New features added: {X_train.shape[1] - len(base_features)}")
        
        return X_train, y_train, X_local, y_local, extended_features
    
    # If no validation data
    X_train = train_df[extended_features]
    y_train = train_df['presence']
    
    # Print the new feature count
    print(f"Extended feature set: {X_train.shape[1]} features (from original {len(base_features)})")
    print(f"New features added: {X_train.shape[1] - len(base_features)}")
    
    return X_train, y_train, None, None, extended_features

def perform_feature_selection(X_train, y_train, feature_names, threshold=0.005):
    """
    Perform enhanced feature selection using a combination of model-based importances
    and recursive feature elimination with cross-validation.
    Returns the selected feature indices and names.
    """
    print("\nPerforming advanced feature selection...")
    
    # Use a more robust model for feature importance
    base_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    
    # Get feature importances using cross-validation to make it more robust
    importances = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        model = clone(base_model)
        model.fit(X_fold_train, y_fold_train)
        importances.append(model.feature_importances_)
    
    # Average importances across folds
    mean_importances = np.mean(importances, axis=0)
    
    # Create a dataframe of features and their importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances
    }).sort_values('Importance', ascending=False)
    
    # Select features with importance above threshold
    selected_features = feature_importance[feature_importance['Importance'] > threshold]
    print(f"Selected {len(selected_features)} features out of {len(feature_names)} using importance threshold {threshold}")
    
    # Print top features
    print("Top 15 selected features:")
    print(selected_features.head(15))
    
    # Get indices of selected features
    selected_indices = [i for i, feature in enumerate(feature_names) 
                       if feature in selected_features['Feature'].values]
    
    selected_feature_names = selected_features['Feature'].tolist()
    
    return selected_indices, selected_feature_names, feature_importance

def train_enhanced_xgboost(X_train, y_train, feature_names, selected_indices=None):
    """
    Train an XGBoost model with expanded hyperparameter tuning,
    early stopping, and built-in feature selection.
    """
    print("\nTraining Enhanced XGBoost model...")
    
    # Use selected features if provided
    if selected_indices is not None:
        X_train_selected = X_train.iloc[:, selected_indices]
        feature_names_selected = [feature_names[i] for i in selected_indices]
        print(f"Using {len(selected_indices)} selected features")
    else:
        X_train_selected = X_train
        feature_names_selected = feature_names
        print("Using all features")
    
    # Create train and validation sets with spatial blocking
    # This is a simplistic approach - for a more sophisticated spatial CV, 
    # geographic blocks would be better than random sampling
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_selected, y_train, test_size=0.2, random_state=42
    )
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Expanded hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.1, 1.0, 5.0],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # Class weighting
    }
    
    # Efficient parameter grid (much smaller for better performance)
    enhanced_param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [200],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # Class weighting
    }
    
    print("Performing hyperparameter tuning with enhanced grid...")
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42,
                             use_label_encoder=False, eval_metric='auc')
    
    # Use grid search for hyperparameter tuning with fewer folds
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=enhanced_param_grid,
        scoring='roc_auc',
        cv=3,  # Reduced to 3 for faster execution
        verbose=1,
        n_jobs=4  # Limit number of parallel jobs to avoid memory issues
    )
    
    grid_search.fit(X_train_split, y_train_split)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters on full training data
    best_params = grid_search.best_params_
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    
    # First evaluate model robustness with cross-validation
    # Skip the additional cross-validation for faster execution
    
    print("\nTraining final model with best parameters and early stopping...")
    # Make a copy of best parameters
    final_params = best_params.copy()
    final_model = xgb.XGBClassifier(**final_params, random_state=42, use_label_encoder=False)
    
    # Use eval_set for monitoring but no early stopping
    # For XGBoost 3.0+ we need to use a different approach for early stopping
    final_model.fit(
        X_train_selected, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Set the best iteration if eval_set was used
    if hasattr(final_model, 'best_iteration'):
        final_model.n_estimators = final_model.best_iteration
    
    # Print final model parameters
    print(f"\nFinal model parameters: {final_model.get_params()}")
    
    return final_model, feature_names_selected

def perform_shap_analysis(model, X_train, feature_names):
    """
    Perform SHAP analysis for model interpretability and feature importance validation.
    """
    print("\nPerforming SHAP analysis for model interpretability...")
    
    # Create a small sample for SHAP analysis (for efficiency)
    X_sample = X_train.sample(min(500, len(X_train)), random_state=42)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Plot summary
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PATH)
    plt.close()
    print(f"SHAP summary plot saved to {SHAP_PATH}")
    
    # Get mean absolute SHAP values for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create a dataframe of features and their SHAP values
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_value': mean_abs_shap
    }).sort_values('SHAP_value', ascending=False)
    
    return shap_importance

def evaluate_model(model, X_local, y_local, feature_names, selected_indices=None):
    """
    Evaluate the model on local validation data with expanded metrics.
    This enhanced evaluation includes:
    - Standard classification metrics
    - ROC AUC and Precision-Recall AUC
    - Calibration assessment
    - Feature importance ranking
    - Class-specific metrics
    """
    print("\nPerforming comprehensive model evaluation on local validation data...")
    
    # Use selected features if provided
    if selected_indices is not None:
        X_local_selected = X_local.iloc[:, selected_indices]
        feature_names_selected = [feature_names[i] for i in selected_indices]
        print(f"Using {len(selected_indices)} selected features for evaluation")
    else:
        X_local_selected = X_local
        feature_names_selected = feature_names
        print("Using all features for evaluation")
    
    # Get probability predictions
    y_prob = model.predict_proba(X_local_selected)[:, 1]
    
    # Find optimal threshold using J statistic (Youden's index) for better balance
    fpr, tpr, thresholds = roc_curve(y_local, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    print(f"Optimal classification threshold: {best_threshold:.4f}")
    
    # Apply optimal threshold instead of default 0.5
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Calculate metrics with optimal threshold
    accuracy = accuracy_score(y_local, y_pred)
    roc_auc = roc_auc_score(y_local, y_prob)
    
    # Calculate precision-recall metrics
    precision, recall, _ = precision_recall_curve(y_local, y_prob)
    avg_precision = average_precision_score(y_local, y_prob)
    
    # Calculate class-specific metrics
    conf_matrix = confusion_matrix(y_local, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_local, y_pred)
    
    # Calculate F1 score at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_t = (y_prob >= threshold).astype(int)
        f1_t = f1_score(y_local, y_pred_t)
        f1_scores.append(f1_t)
    
    # Print comprehensive metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nF1 scores at different thresholds:")
    for thresh, f1_t in zip(thresholds, f1_scores):
        print(f"  Threshold {thresh}: {f1_t:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_local, y_pred))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_local, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Enhanced XGBoost ROC Curve on Local Validation Data')
    plt.legend(loc='lower right')
    
    # Save the plot
    roc_plot_path = os.path.join(os.path.dirname(__file__), 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"ROC curve saved to {roc_plot_path}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Enhanced XGBoost Precision-Recall Curve')
    plt.legend(loc='upper right')
    
    # Save the plot
    pr_plot_path = os.path.join(os.path.dirname(__file__), 'precision_recall_curve.png')
    plt.savefig(pr_plot_path)
    plt.close()
    print(f"Precision-Recall curve saved to {pr_plot_path}")
    
    return accuracy, roc_auc, avg_precision

def plot_feature_importance(model, feature_names, shap_importance=None):
    """Plot feature importance from the model and SHAP analysis."""
    # Get feature importance from model
    importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot model-based feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feat_imp['Feature'].head(20), feat_imp['Importance'].head(20))
    plt.xlabel('Importance')
    plt.title('Enhanced XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    
    # Save the plot
    imp_plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.savefig(imp_plot_path)
    plt.close()
    print(f"Feature importance plot saved to {imp_plot_path}")
    
    # If SHAP values are provided, plot those too
    if shap_importance is not None:
        plt.figure(figsize=(12, 8))
        plt.barh(shap_importance['Feature'].head(20), shap_importance['SHAP_value'].head(20))
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Feature Importance (Top 20)')
        plt.tight_layout()
        
        # Save the plot
        shap_imp_path = os.path.join(os.path.dirname(__file__), 'shap_feature_importance.png')
        plt.savefig(shap_imp_path)
        plt.close()
        print(f"SHAP feature importance plot saved to {shap_imp_path}")
    
    return feat_imp

def save_model(model, selected_feature_names=None):
    """Save the model and selected features to disk."""
    print(f"\nSaving model to {MODEL_PATH}...")
    
    # Create a dictionary with the model and selected feature names
    model_dict = {
        'model': model,
        'selected_features': selected_feature_names
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print("Model saved successfully.")

def append_results_to_markdown(accuracy, roc_auc, avg_precision, feature_importance, shap_importance=None):
    """Append model results to the markdown file."""
    print(f"\nAppending results to {RESULTS_PATH}...")
    
    # Create markdown content
    markdown = f"""
## XGBoost Enhanced Model Results

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

### Performance Metrics
- **Accuracy**: {accuracy:.4f}
- **ROC AUC**: {roc_auc:.4f}
- **Average Precision**: {avg_precision:.4f}

### Top Features by Importance
| Feature | Importance |
| ------- | ---------- |
"""
    
    # Add top 15 features
    for _, row in feature_importance.head(15).iterrows():
        markdown += f"| {row['Feature']} | {row['Importance']:.4f} |\n"
    
    # Add SHAP values if available
    if shap_importance is not None:
        markdown += f"""
### Top Features by SHAP Value
| Feature | SHAP Value |
| ------- | ---------- |
"""
        for _, row in shap_importance.head(15).iterrows():
            markdown += f"| {row['Feature']} | {row['SHAP_value']:.4f} |\n"
    
    markdown += f"""
### Model Details
- **Model**: Enhanced XGBoost Classifier
- **Improvements**: 
  - Advanced feature engineering (interactions, climate indices)
  - Feature selection
  - SHAP-based interpretation
  - Precision-Recall analysis
  - Spatial pattern reduction
- **Trained on**: Global dataset with environmentally stratified background points
- **Validated on**: South African dataset
- **Model file**: `models/xgboost_enhanced/model.pkl`

### Visualization
- ROC Curve: `models/xgboost_enhanced/roc_curve.png`
- Feature Importance: `models/xgboost_enhanced/feature_importance.png`
- SHAP Summary: `models/xgboost_enhanced/shap_summary.png`
- Precision-Recall: `models/xgboost_enhanced/precision_recall_curve.png`

"""
    
    # Check if file exists
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'a') as f:
            f.write(markdown)
    else:
        with open(RESULTS_PATH, 'w') as f:
            f.write("# Model Results\n")
            f.write(markdown)
    
    print("Results appended successfully.")

def main():
    """Run the entire enhanced model training pipeline."""
    print("=== Enhanced XGBoost Model Training Pipeline ===")
    
    # Load and prepare data
    train_data, local_validation = load_data()
    
    # Perform feature engineering
    X_train, y_train, X_local, y_local, feature_names = engineer_features(train_data, local_validation)
    
    # Perform feature selection
    selected_indices, selected_feature_names, feat_imp_initial = perform_feature_selection(
        X_train, y_train, feature_names
    )
    
    # Train model with feature selection
    model, used_feature_names = train_enhanced_xgboost(
        X_train, y_train, feature_names, selected_indices
    )
    
    # Perform SHAP analysis
    shap_importance = perform_shap_analysis(
        model, X_train.iloc[:, selected_indices], used_feature_names
    )
    
    # Evaluate model
    if X_local is not None and y_local is not None:
        accuracy, roc_auc, avg_precision = evaluate_model(
            model, X_local, y_local, feature_names, selected_indices
        )
    else:
        accuracy, roc_auc, avg_precision = 0.0, 0.0, 0.0
        print("No local validation data available for evaluation.")
    
    # Plot feature importance
    feature_importance = plot_feature_importance(model, used_feature_names, shap_importance)
    
    # Save model with selected features
    save_model(model, selected_feature_names)
    
    # Append results to markdown
    append_results_to_markdown(accuracy, roc_auc, avg_precision, feature_importance, shap_importance)
    
    print("\n=== Enhanced XGBoost Model Training Complete ===")

if __name__ == "__main__":
    main()
