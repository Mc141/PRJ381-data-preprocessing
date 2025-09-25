# Enhanced XGBoost for Pyracantha Invasion Risk Prediction

This folder contains an enhanced XGBoost model implementation that builds upon the standard XGBoost model with several improvements aimed at better ecological modeling and reduced geographic dependence.

## Key Improvements

### 1. Advanced Feature Engineering

- **Climate Indices**: Created ecologically relevant variables like aridity index, temperature range, and precipitation seasonality ratio
- **Interaction Terms**: Generated meaningful interactions between climate variables that have ecological significance
- **Distance Features**: Reduced geographic dependence by using distance-based features instead of raw coordinates
- **Polynomial Features**: Generated interaction terms between environmental variables to capture non-linear relationships

### 2. Better Background Point Generation

- **Environmentally Stratified Sampling**: Generated background points that better represent the full range of environmental conditions
- **Reduced Geographic Bias**: Used smaller coordinate offsets to create more realistic environmental conditions
- **Multiple Sampling Strategies**: Combined samples across different environmental gradients (temperature, precipitation, elevation)

### 3. Feature Selection

- **Importance-based Selection**: Used initial model feature importances to select most relevant predictors
- **SHAP Analysis**: Used SHAP values to understand and validate feature importance
- **Dimensionality Reduction**: Reduced the feature set to the most important variables to prevent overfitting

### 4. Improved Model Training

- **Expanded Hyperparameter Tuning**: Explored a wider range of XGBoost parameters
- **Early Stopping**: Prevented overfitting through early stopping based on validation performance
- **Class Weighting**: Adjusted for imbalance between presence and background points
- **Spatial Cross-Validation**: More realistic assessment of model performance across space

### 5. Enhanced Evaluation

- **Precision-Recall Analysis**: Added precision-recall curves better suited for imbalanced classification
- **SHAP Interpretation**: Used SHAP values to interpret model predictions
- **Visual Diagnostics**: Added more comprehensive visualizations of model performance and feature importance

## Usage

Run the enhanced model with:

```bash
python -m experiments.xgboost_enhanced.train_model
```

## Files

- `train_model.py`: The main script containing the enhanced XGBoost implementation
- `model.pkl`: The trained model (created after running the script)
- `feature_importance.png`: Plot of feature importance (created after running)
- `shap_summary.png`: SHAP value summary plot (created after running)
- `roc_curve.png`: ROC curve visualization (created after running)
- `precision_recall_curve.png`: Precision-recall curve (created after running)

## Results

Results will be appended to the main `MODEL_RESULTS.md` file in the experiments directory.
