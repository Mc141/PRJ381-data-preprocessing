# Model Results - Pyracantha Invasion Risk Prediction

This file tracks the performance metrics and results for different modeling approaches used to predict Pyracantha angustifolia invasion risk.

## Model Comparison

| Model            | Accuracy | ROC AUC | F1 Score | Sensitivity | Specificity | Avg Precision | Key Features                         |
| ---------------- | -------- | ------- | -------- | ----------- | ----------- | ------------- | ------------------------------------ |
| Random Forest    | 0.7548   | 0.8284  | 0.8553   | 1.0000      | 0.1100      | 0.9306        | Climate + Geographic + Temporal      |
| XGBoost          | 0.7769   | 0.7791  | 0.8657   | 0.9924      | 0.2100      | 0.8915        | Climate + Geographic + Temporal      |
| XGBoost Enhanced | 0.6970   | 0.6928  | 0.7835   | 0.7567      | 0.5400      | 0.8422        | Enhanced features + SHAP + Selection |
| Rule-based       | N/A      | N/A     | N/A      | N/A         | N/A         | N/A           | Climate thresholds                   |

_Note: All metrics evaluated on South African validation dataset. XGBoost Enhanced shows more balanced precision/recall._

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

## XGBoost Model Results

**Date**: 2025-09-07 15:12

### Performance Metrics

- **Accuracy**: 0.7686
- **ROC AUC**: 0.7791

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.1678     |
| bio6      | 0.1000     |
| bio4      | 0.0940     |
| elevation | 0.0936     |
| latitude  | 0.0934     |
| bio14     | 0.0690     |
| bio1      | 0.0626     |
| bio15     | 0.0609     |
| bio13     | 0.0587     |
| bio12     | 0.0558     |

### Model Details

- **Model**: XGBoost Classifier
- **Trained on**: Global dataset with pseudo-absence points
- **Validated on**: South African dataset
- **Model file**: `experiments/xgboost/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost/roc_curve.png`
- Feature Importance: `experiments/xgboost/feature_importance.png`

## XGBoost Model Results

**Date**: 2025-09-07 15:16

### Performance Metrics

- **Accuracy**: 0.7686
- **ROC AUC**: 0.7791

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.1678     |
| bio6      | 0.1000     |
| bio4      | 0.0940     |
| elevation | 0.0936     |
| latitude  | 0.0934     |
| bio14     | 0.0690     |
| bio1      | 0.0626     |
| bio15     | 0.0609     |
| bio13     | 0.0587     |
| bio12     | 0.0558     |

### Model Details

- **Model**: XGBoost Classifier
- **Trained on**: Global dataset with real presence data and background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/xgboost/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost/roc_curve.png`
- Feature Importance: `experiments/xgboost/feature_importance.png`

## XGBoost Model Results

**Date**: 2025-09-07 15:20

### Performance Metrics

- **Accuracy**: 0.7686
- **ROC AUC**: 0.7791

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.1678     |
| bio6      | 0.1000     |
| bio4      | 0.0940     |
| elevation | 0.0936     |
| latitude  | 0.0934     |
| bio14     | 0.0690     |
| bio1      | 0.0626     |
| bio15     | 0.0609     |
| bio13     | 0.0587     |
| bio12     | 0.0558     |

### Model Details

- **Model**: XGBoost Classifier
- **Trained on**: Global dataset with real presence data and background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/xgboost/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost/roc_curve.png`
- Feature Importance: `experiments/xgboost/feature_importance.png`

## Random Forest Model Results

Date: 2025-09-07 15:21:37

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

## Random Forest Model Results

**Date**: 2025-09-07 15:26

### Performance Metrics

- **Accuracy**: 0.7034
- **ROC AUC**: 0.8601

### Top Features by Importance

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

### Model Details

- **Model**: Random Forest Classifier
- **Hyperparameters**: Trees=100, Max Depth=15, Min Samples Split=10, Min Samples Leaf=5
- **Trained on**: Global dataset with pseudo-absence points
- **Validated on**: South African dataset
- **Model file**: `experiments/random_forest/model.pkl`

### Visualization

- Feature Importance: `experiments/random_forest/feature_importance.png`

## XGBoost Model Results

**Date**: 2025-09-07 15:27

### Performance Metrics

- **Accuracy**: 0.7686
- **ROC AUC**: 0.7791

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.1678     |
| bio6      | 0.1000     |
| bio4      | 0.0940     |
| elevation | 0.0936     |
| latitude  | 0.0934     |
| bio14     | 0.0690     |
| bio1      | 0.0626     |
| bio15     | 0.0609     |
| bio13     | 0.0587     |
| bio12     | 0.0558     |

### Model Details

- **Model**: XGBoost Classifier
- **Hyperparameters**: Tuned via GridSearchCV (best params shown in console)
- **Trained on**: Global dataset with background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/xgboost/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost/roc_curve.png`
- Feature Importance: `experiments/xgboost/feature_importance.png`

## Random Forest Model Results (Optimized)

**Date**: 2025-09-07 15:38

### Performance Metrics

- **Accuracy**: 0.7410
- **ROC AUC**: 0.8284

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.2769     |
| latitude  | 0.2278     |
| elevation | 0.0738     |
| bio14     | 0.0516     |
| bio6      | 0.0511     |
| bio4      | 0.0505     |
| bio1      | 0.0476     |
| bio12     | 0.0474     |
| bio15     | 0.0467     |
| bio5      | 0.0438     |

### Model Details

- **Model**: Random Forest Classifier
- **Hyperparameters**: Tuned via GridSearchCV
  - n_estimators: 200
  - max_depth: 12
  - min_samples_split: 5
  - min_samples_leaf: 2
- **Trained on**: Global dataset with background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/random_forest/model.pkl`

### Visualization

- Feature Importance: `experiments/random_forest/feature_importance.png`
- ROC Curve: `experiments/random_forest/roc_curve.png`

## XGBoost Enhanced Model Results

**Date**: 2025-09-07 16:37

### Performance Metrics

- **Accuracy**: 0.7273
- **ROC AUC**: 0.6838
- **Average Precision**: 0.8356

### Top Features by Importance

| Feature          | Importance |
| ---------------- | ---------- |
| dist_from_median | 0.0528     |
| bio6_x_bio13     | 0.0492     |
| bio1_x_bio13     | 0.0489     |
| bio6_x_bio14     | 0.0417     |
| longitude        | 0.0347     |
| bio1_x_bio14     | 0.0340     |
| bio1             | 0.0309     |
| sin_month        | 0.0253     |
| bio14_x_bio15    | 0.0252     |
| latitude         | 0.0232     |
| bio5_x_bio12     | 0.0220     |
| bio6             | 0.0215     |
| bio13_x_bio15    | 0.0213     |
| elev_temp        | 0.0208     |
| bio5_x_bio6      | 0.0204     |

### Top Features by SHAP Value

| Feature           | SHAP Value |
| ----------------- | ---------- |
| dist_from_median  | 0.4742     |
| longitude         | 0.3403     |
| latitude          | 0.1947     |
| elevation         | 0.0308     |
| bio1              | 0.0293     |
| bio14_x_bio15     | 0.0280     |
| sin_month         | 0.0267     |
| bio1_x_bio13      | 0.0267     |
| elevation_x_bio6  | 0.0252     |
| elevation_x_bio13 | 0.0214     |
| elev_temp         | 0.0210     |
| cos_month         | 0.0209     |
| bio13             | 0.0165     |
| bio6_x_bio14      | 0.0162     |
| bio4_x_bio5       | 0.0147     |

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
- **Model file**: `experiments/xgboost_enhanced/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost_enhanced/roc_curve.png`
- Feature Importance: `experiments/xgboost_enhanced/feature_importance.png`
- SHAP Summary: `experiments/xgboost_enhanced/shap_summary.png`
- Precision-Recall: `experiments/xgboost_enhanced/precision_recall_curve.png`

## XGBoost Enhanced Model Results

**Date**: 2025-09-07 17:10

### Performance Metrics

- **Accuracy**: 0.7245
- **ROC AUC**: 0.6928
- **Average Precision**: 0.8422

### Top Features by Importance

| Feature          | Importance |
| ---------------- | ---------- |
| bio6_x_bio14     | 0.0445     |
| dist_from_median | 0.0429     |
| bio1_x_bio13     | 0.0394     |
| longitude        | 0.0289     |
| bio5_x_bio6      | 0.0276     |
| bio6_x_bio13     | 0.0258     |
| bio1_x_bio14     | 0.0227     |
| sin_month        | 0.0217     |
| bio1             | 0.0212     |
| bio6_x_bio15     | 0.0204     |
| latitude         | 0.0192     |
| bio13_x_bio15    | 0.0186     |
| bio15            | 0.0185     |
| bio1_x_bio12     | 0.0180     |
| temp_precip      | 0.0178     |

### Top Features by SHAP Value

| Feature           | SHAP Value |
| ----------------- | ---------- |
| dist_from_median  | 0.4519     |
| longitude         | 0.3501     |
| latitude          | 0.1958     |
| bio1_x_bio13      | 0.0335     |
| elevation_x_bio6  | 0.0304     |
| bio1              | 0.0277     |
| sin_month         | 0.0275     |
| elevation         | 0.0231     |
| bio14_x_bio15     | 0.0223     |
| elev_temp         | 0.0218     |
| bio13             | 0.0203     |
| cos_month         | 0.0191     |
| elevation_x_bio13 | 0.0190     |
| bio6_x_bio14      | 0.0167     |
| bio13_x_bio15     | 0.0142     |

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
- **Model file**: `experiments/xgboost_enhanced/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost_enhanced/roc_curve.png`
- Feature Importance: `experiments/xgboost_enhanced/feature_importance.png`
- SHAP Summary: `experiments/xgboost_enhanced/shap_summary.png`
- Precision-Recall: `experiments/xgboost_enhanced/precision_recall_curve.png`

## XGBoost Enhanced Model Results

**Date**: 2025-09-07 17:12

### Performance Metrics

- **Accuracy**: 0.6970
- **ROC AUC**: 0.6928
- **Average Precision**: 0.8422

### Top Features by Importance

| Feature          | Importance |
| ---------------- | ---------- |
| bio6_x_bio14     | 0.0445     |
| dist_from_median | 0.0429     |
| bio1_x_bio13     | 0.0394     |
| longitude        | 0.0289     |
| bio5_x_bio6      | 0.0276     |
| bio6_x_bio13     | 0.0258     |
| bio1_x_bio14     | 0.0227     |
| sin_month        | 0.0217     |
| bio1             | 0.0212     |
| bio6_x_bio15     | 0.0204     |
| latitude         | 0.0192     |
| bio13_x_bio15    | 0.0186     |
| bio15            | 0.0185     |
| bio1_x_bio12     | 0.0180     |
| temp_precip      | 0.0178     |

### Top Features by SHAP Value

| Feature           | SHAP Value |
| ----------------- | ---------- |
| dist_from_median  | 0.4519     |
| longitude         | 0.3501     |
| latitude          | 0.1958     |
| bio1_x_bio13      | 0.0335     |
| elevation_x_bio6  | 0.0304     |
| bio1              | 0.0277     |
| sin_month         | 0.0275     |
| elevation         | 0.0231     |
| bio14_x_bio15     | 0.0223     |
| elev_temp         | 0.0218     |
| bio13             | 0.0203     |
| cos_month         | 0.0191     |
| elevation_x_bio13 | 0.0190     |
| bio6_x_bio14      | 0.0167     |
| bio13_x_bio15     | 0.0142     |

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
- **Model file**: `experiments/xgboost_enhanced/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost_enhanced/roc_curve.png`
- Feature Importance: `experiments/xgboost_enhanced/feature_importance.png`
- SHAP Summary: `experiments/xgboost_enhanced/shap_summary.png`
- Precision-Recall: `experiments/xgboost_enhanced/precision_recall_curve.png`

## Random Forest Model Results

**Date**: 2025-09-07 19:07

### Performance Metrics

- **Accuracy**: 0.7548
- **ROC AUC**: 0.8284
- **F1 Score**: 0.8553
- **Sensitivity**: 1.0000
- **Specificity**: 0.1100
- **Average Precision**: 0.9306
- **Optimal Threshold**: 0.5900

### Confusion Matrix

```
[[ 11 89 ]
 [ 0 263 ]]
```

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.2769     |
| latitude  | 0.2278     |
| elevation | 0.0738     |
| bio14     | 0.0516     |
| bio6      | 0.0511     |
| bio4      | 0.0505     |
| bio1      | 0.0476     |
| bio12     | 0.0474     |
| bio15     | 0.0467     |
| bio5      | 0.0438     |

### Model Details

- **Model**: Random Forest Classifier
- **Hyperparameters**: Tuned via GridSearchCV
  - n_estimators: 200
  - max_depth: 12
  - min_samples_split: 5
  - min_samples_leaf: 2
- **Trained on**: Global dataset with background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/random_forest/model.pkl`

### Visualization

- Feature Importance: `experiments/random_forest/feature_importance.png`
- ROC Curve: `experiments/random_forest/roc_curve.png`

## XGBoost Model Results

**Date**: 2025-09-07 19:08

### Performance Metrics

- **Accuracy**: 0.7769
- **ROC AUC**: 0.7791
- **F1 Score**: 0.8657
- **Sensitivity**: 0.9924
- **Specificity**: 0.2100
- **Average Precision**: 0.8915
- **Optimal Threshold**: 0.5700

### Confusion Matrix

```
[[ 21 79 ]
 [ 2 261 ]]
```

### Top Features by Importance

| Feature   | Importance |
| --------- | ---------- |
| longitude | 0.1678     |
| bio6      | 0.1000     |
| bio4      | 0.0940     |
| elevation | 0.0936     |
| latitude  | 0.0934     |
| bio14     | 0.0690     |
| bio1      | 0.0626     |
| bio15     | 0.0609     |
| bio13     | 0.0587     |
| bio12     | 0.0558     |

### Model Details

- **Model**: XGBoost Classifier
- **Hyperparameters**: Tuned via GridSearchCV (best params shown in console)
- **Optimal Threshold**: 0.5700
- **Trained on**: Global dataset with background comparison points
- **Validated on**: South African dataset
- **Model file**: `experiments/xgboost/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost/roc_curve.png`
- Feature Importance: `experiments/xgboost/feature_importance.png`

## XGBoost Enhanced Model Results

**Date**: 2025-09-21 12:48

### Performance Metrics

- **Accuracy**: 0.6970
- **ROC AUC**: 0.6928
- **Average Precision**: 0.8422
- **F1 Score**: 0.7835
- **Sensitivity (Recall)**: 0.7567
- **Specificity**: 0.5400

### Top Features by Importance

| Feature          | Importance |
| ---------------- | ---------- |
| bio6_x_bio14     | 0.0445     |
| dist_from_median | 0.0429     |
| bio1_x_bio13     | 0.0394     |
| longitude        | 0.0289     |
| bio5_x_bio6      | 0.0276     |
| bio6_x_bio13     | 0.0258     |
| bio1_x_bio14     | 0.0227     |
| sin_month        | 0.0217     |
| bio1             | 0.0212     |
| bio6_x_bio15     | 0.0204     |
| latitude         | 0.0192     |
| bio13_x_bio15    | 0.0186     |
| bio15            | 0.0185     |
| bio1_x_bio12     | 0.0180     |
| temp_precip      | 0.0178     |

### Top Features by SHAP Value

| Feature           | SHAP Value |
| ----------------- | ---------- |
| dist_from_median  | 0.4519     |
| longitude         | 0.3501     |
| latitude          | 0.1958     |
| bio1_x_bio13      | 0.0335     |
| elevation_x_bio6  | 0.0304     |
| bio1              | 0.0277     |
| sin_month         | 0.0275     |
| elevation         | 0.0231     |
| bio14_x_bio15     | 0.0223     |
| elev_temp         | 0.0218     |
| bio13             | 0.0203     |
| cos_month         | 0.0191     |
| elevation_x_bio13 | 0.0190     |
| bio6_x_bio14      | 0.0167     |
| bio13_x_bio15     | 0.0142     |

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
- **Model file**: `experiments/xgboost_enhanced/model.pkl`

### Visualization

- ROC Curve: `experiments/xgboost_enhanced/roc_curve.png`
- Feature Importance: `experiments/xgboost_enhanced/feature_importance.png`
- SHAP Summary: `experiments/xgboost_enhanced/shap_summary.png`
- Precision-Recall: `experiments/xgboost_enhanced/precision_recall_curve.png`
