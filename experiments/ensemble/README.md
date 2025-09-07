# Ensemble Model for Pyracantha Invasion Risk Prediction

This folder contains an ensemble model that combines the strengths of different modeling approaches to achieve better and more reliable predictions for Pyracantha invasion risk.

## Model Description

The ensemble approach combines:

1. **Multiple Base Learners**:

   - XGBoost (optimized parameters from enhanced XGBoost)
   - Random Forest (optimized parameters from Random Forest model)

2. **Ensemble Methods**:

   - Stacking: Uses a meta-learner (Logistic Regression) to learn how to best combine base model predictions
   - Voting: Averages probabilistic predictions from base models
   - The better-performing ensemble is automatically selected

3. **Advanced Feature Engineering**:

   - Uses all the enhanced feature engineering from the XGBoost enhanced model
   - Includes climate indices, interaction terms, and distance-based features

4. **Optimal Classification Threshold**:
   - Uses Youden's Index (J-statistic) to find optimal decision boundary
   - Maximizes balanced performance between sensitivity and specificity

## Files

- `train_model.py`: Main script to train and evaluate ensemble models
- `model.pkl`: Saved ensemble model (after running the script)
- `optimal_threshold.pkl`: Optimal classification threshold for predictions
- `roc_curve.png`: ROC curve visualization
- `precision_recall_curve.png`: Precision-recall curve visualization

## Usage

To train and evaluate the ensemble model:

```bash
python -m experiments.ensemble.train_model
```

## Key Advantages

1. **Improved Accuracy**: Ensemble models generally provide better predictive performance than individual models
2. **Reduced Overfitting**: Different models overfit in different ways, so combining them reduces overall overfitting
3. **Better Generalization**: Combined predictions are more reliable across different environments
4. **More Robust**: Less sensitive to data issues or parameter choices in any single algorithm
5. **Enhanced Balance**: Optimized threshold selection provides better balance between sensitivity and specificity

## Performance Metrics

Performance metrics are automatically added to the main MODEL_RESULTS.md file after training.
