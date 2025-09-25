# Experiments - Pyracantha Invasion Risk Modeling

This directory contains experimental models for Pyracantha invasion risk prediction using the preprocessed data from the main application.

## Models

### Random Forest

A supervised machine learning model trained on global Pyracantha occurrence data and validated on South African data.

- **Directory**: `random_forest/`
- **Features**: 13 features including geographic coordinates, climate variables, and temporal features
- **Usage**: Run `python run_pipeline.py` to train the model and generate invasion risk maps

### Rule-based

A simple rule-based approach using climate thresholds for comparison with machine learning models.

- **Directory**: `rule_based/`
- **Features**: Primarily uses climate thresholds without machine learning
- **Usage**: See documentation in the directory

## Results

See `MODEL_RESULTS.md` for detailed performance metrics and comparisons between different modeling approaches.

## Running Experiments

Each experiment folder contains its own README with specific instructions. The general workflow is:

1. First run the data preprocessing pipeline in the main application to generate the necessary CSV files
2. Navigate to the experiment directory of interest
3. Follow the specific instructions to run the model
4. Results are saved in the experiment directory and summarized in `MODEL_RESULTS.md`

## Visualization

Each model includes code to generate interactive maps showing invasion risk across geographical areas. These visualizations help identify high-risk regions for monitoring and management.
