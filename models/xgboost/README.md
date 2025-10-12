# XGBoost ## Why XGBoost?

XGBoost offers several advantages over Random Forest for this particular prediction task:

1. **Better performance on structured data**: XGBoost often outperforms Random Forest on structured environmental data like ours
2. **Gradient boosting architecture**: Sequentially builds models that focus on correcting previous errors
3. **Built-in regularization**: Helps prevent overfitting with regularization parameters
4. **Better handling of imbalanced data**: Important for our presence vs background comparison scenario
5. **Efficient memory usage**: More memory-efficient for large datasets

## Note on Binary Classification

For binary classification tasks like invasion risk prediction, models require:

- Class 1: Presence points (locations where the species is known to occur)
- Class 0: Background/absence points (locations representing unsuitable environments)

This implementation:

1. **Uses real occurrence data** from the provided datasets
2. Creates minimal background comparison points only if needed for model training
3. Background points are created by offsetting real occurrences to maintain environmental realism
4. These background points are not claimed to be "true absences" but rather comparison points needed for the mathematical requirements of binary classification algorithmsantha Invasion Risk Prediction

This directory contains an XGBoost-based approach for analyzing and visualizing the invasion risk of _Pyracantha angustifolia_ in the Western Cape region using only real data.

## Files

- `train_model.py`: Script for training and evaluating the XGBoost model on extracted environmental features
- `generate_heatmap_api.py`: Script for creating a grid-based choropleth map using real environmental data from API endpoints
- `model.pkl`: Trained XGBoost model (generated after running `train_model.py`)

## Why XGBoost?

XGBoost offers several advantages over Random Forest for this particular prediction task:

1. **Better performance on structured data**: XGBoost often outperforms Random Forest on structured environmental data like ours
2. **Gradient boosting architecture**: Sequentially builds models that focus on correcting previous errors
3. **Built-in regularization**: Helps prevent overfitting with regularization parameters
4. **Better handling of imbalanced data**: Important for species distribution data that may be imbalanced
5. **Efficient memory usage**: More memory-efficient for large datasets

## Usage

### Step 1: Train the XGBoost Model

First, train the XGBoost model:

```powershell
python -m experiments.xgboost.train_model
```

This will:

1. Load the global training and local validation datasets (primarily using real data)
2. Create minimal background comparison points if needed for binary classification
3. Perform hyperparameter tuning with cross-validation
4. Train an XGBoost classifier with the best parameters
5. Evaluate the model performance with metrics and visualizations
6. Save the trained model as `model.pkl`

Note: XGBoost (like all binary classifiers) requires both presence (1) and absence (0) classes.
If your dataset only contains presence points, the script will create a minimal set of background
points by offsetting a subset of the occurrence locations, using real environmental conditions.

### Step 2: Generate Grid-Based Invasion Risk Map

Once the model is trained, you can generate invasion risk maps using real environmental data from API endpoints:

```powershell
# Start the FastAPI server in one terminal
python -m uvicorn app.main:app --reload

# In another terminal, run the map generator
python -m experiments.xgboost.generate_heatmap_api
```

### Grid Size Control

```powershell
# Smaller grid (faster)
python -m experiments.xgboost.generate_heatmap_api --grid_size 10

# Larger grid (more detailed but slower)
python -m experiments.xgboost.generate_heatmap_api --grid_size 25
```

### Region Selection

```powershell
# Core Western Cape (default)
python -m experiments.xgboost.generate_heatmap_api

# Extended Western Cape region
python -m experiments.xgboost.generate_heatmap_api --western_cape_extended

# Custom area within Western Cape
python -m experiments.xgboost.generate_heatmap_api --specific_area --lat_min -33.5 --lat_max -32.0 --lon_min 18.3 --lon_max 19.5
```

### Seasonal Analysis

```powershell
# March analysis (default)
python -m experiments.xgboost.generate_heatmap_api --month 3

# September analysis
python -m experiments.xgboost.generate_heatmap_api --month 9
```

## Output

The script generates an interactive HTML map with these features:

- Grid-based choropleth visualization of invasion risk
- Color-coded cells from blue (low risk) to red (high risk)
- Tooltip information when hovering over grid cells
- Statistics panel showing risk distribution
- Layer controls for switching between map types

## Model Performance

The XGBoost model's performance metrics are appended to the main `MODEL_RESULTS.md` file in the experiments directory after training. This includes:

- Accuracy
- ROC AUC score
- Top feature importance
- Model details
- Links to visualization plots

## Notes

- XGBoost typically produces sharper decision boundaries than Random Forest
- The model includes comprehensive rate limiting to avoid API overload
- Real-time model comparison between XGBoost and Random Forest is possible by running both implementations
- Use `--grid_size` to control the trade-off between detail and processing time
