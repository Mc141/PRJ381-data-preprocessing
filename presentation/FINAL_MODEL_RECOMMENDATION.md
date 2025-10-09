# Final Model Recommendation for Pyracantha Invasion Risk Prediction

**Date:** 2025-09-21

## Model Comparison

![Model Comparison](./model_comparison.png)

### Performance Metrics

| Model            | Accuracy | AUC    | F1 Score | Sensitivity | Specificity |
| ---------------- | -------- | ------ | -------- | ----------- | ----------- |
| Random Forest    | 0.7548   | 0.8284 | 0.8553   | 1.0000      | 0.1100      |
| XGBoost          | 0.7769   | 0.7791 | 0.8657   | 0.9924      | 0.2100      |
| XGBoost Enhanced | 0.6970   | 0.6928 | 0.7835   | 0.7567      | 0.5400      |

## Recommended Model

**XGBoost**

### Key Performance Metrics

- AUC: 0.7791
- Accuracy: 0.7769
- F1 Score: 0.8657
- Specificity: 0.2100
- Sensitivity: 0.9924
- Average Precision: 0.8915

### Reasons for Recommendation

- AUC is 0.05 lower than Random Forest
- Accuracy is 0.02 higher than Random Forest
- F1 Score is 0.01 higher than Random Forest
- Specificity is 0.10 higher than Random Forest
- Average Precision is 0.04 lower than Random Forest
- Overall weighted score (0.8389) is highest among all 3 models

## Implementation Notes

To use the recommended model in the production API:

1. Load the model from `models/xgboost/model.pkl`
2. Use the same feature engineering steps as in the training script
3. Apply the optimal classification threshold for balanced predictions
4. Consider updating the model periodically as new data becomes available

## Future Improvements

1. Incorporate additional environmental data sources
2. Consider temporal dynamics with time-series analysis
3. Add uncertainty quantification to predictions
4. Expand validation with additional field observations
