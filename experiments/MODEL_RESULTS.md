# Model Results - Pyracantha Invasion Risk Prediction

This file tracks the performance metrics and results for different modeling approaches used to predict Pyracantha angustifolia invasion risk.

## Model Comparison

| Model         | Global Accuracy | Global AUC | South Africa Accuracy | South Africa AUC | Key Features                    |
| ------------- | --------------- | ---------- | --------------------- | ---------------- | ------------------------------- |
| Random Forest | TBD             | TBD        | TBD                   | TBD              | Climate + Geographic + Temporal |
| Rule-based    | N/A             | N/A        | N/A                   | N/A              | Climate thresholds              |

_Note: This table will be updated automatically as models are trained_

## Transfer Learning Effectiveness

This section will be updated with metrics showing how well models trained on global data transfer to the South African context.

## Random Forest Model Results

Date: 2025-09-04 15:06:16

### Model Parameters

- Model Type: Random Forest
- Number of Trees: 100
- Max Depth: 15
- Min Samples Split: 10
- Min Samples Leaf: 5

### Performance Metrics

- Global Training Accuracy: 0.9271
- Global Training AUC: 0.9841
- South Africa Validation Accuracy: 0.7034
- South Africa Validation AUC: 0.8601

### Feature Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.3959     |
| latitude  | 0.3149     |
| bio4      | 0.0374     |
| elevation | 0.0372     |
| bio15     | 0.0338     |
| bio13     | 0.0285     |
| bio12     | 0.0273     |
| bio14     | 0.0269     |
| bio1      | 0.0268     |
| bio5      | 0.0266     |
| bio6      | 0.0248     |
| cos_month | 0.0102     |
| sin_month | 0.0096     |
