# Outputs Directory

This directory contains generated files from the PRJ381 Data Preprocessing and ML pipeline.

## Generated Files

- **ML Models**: Trained machine learning models (Random Forest)
- **Prediction Maps**: Interactive HTML maps showing seasonal invasion risk
- **Data Exports**: CSV files from dataset export operations  
- **Visualizations**: Performance charts, feature importance plots
- **Model Reports**: JSON summaries of model performance and patterns

## File Naming Convention

- **`peak_season_invasion_map.html`** - Peak season (April) invasion risk map
- **`off_season_invasion_map.html`** - Off-season (July) invasion risk map
- **`seasonal_pyracantha_model.pkl`** - Trained seasonal Random Forest model
- **`seasonal_model_summary.json`** - Model performance and seasonal patterns
- **`dataset_export_YYYYMMDD_HHMMSS.csv`** - Dataset exports with timestamp

## Machine Learning Outputs

The ML experiments generate:
- **Peak Season Maps** (April): High invasion risk during flowering
- **Off-Season Maps** (July): Lower invasion risk during winter
- **Model Performance**: Accuracy metrics and feature importance
- **Seasonal Analysis**: Biological pattern recognition results

## Note

This directory is included in `.gitignore` to prevent large generated files from being committed to version control.
