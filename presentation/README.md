# Presentation Materials - PRJ381 Species Distribution Modeling

This folder contains all materials needed to create a comprehensive presentation explaining the model selection logic for the PRJ381 project.

## 📁 Folder Contents

### 📄 Main Documentation

- **`MODELS_COMPREHENSIVE_COMPARISON.md`** (40 KB) - Complete technical comparison of all 3 models with 15 sections
- **`PRESENTATION_GUIDE.md`** (14 KB) - Step-by-step guide to create your presentation
- **`MODEL_RESULTS.md`** (25 KB) - Historical record of all model training runs
- **`FINAL_MODEL_RECOMMENDATION.md`** (2 KB) - Final decision: XGBoost selected

### 🎨 Existing Visualizations

#### Root Folder

- `model_comparison.png` - Overall model performance comparison (from root)

#### Random Forest Images (`random_forest_images/`)

- `feature_importance.png` - Feature importance for Random Forest
- `roc_curve.png` - ROC curve for Random Forest

#### XGBoost Images (`xgboost_images/`)

- `feature_importance.png` - Feature importance for XGBoost
- `roc_curve.png` - ROC curve for XGBoost

#### XGBoost Enhanced Images (`xgboost_enhanced_images/`)

- `feature_importance.png` - Feature importance for Enhanced model
- `roc_curve.png` - ROC curve for Enhanced model
- `precision_recall_curve.png` - Precision-Recall curve
- `shap_summary.png` - SHAP summary plot
- `shap_feature_importance.png` - SHAP-based feature importance

**Total: 10 existing images**

### 🐍 Generation Script

- **`generate_presentation_images.py`** - Python script to generate 6 additional visualizations

## 🚀 Quick Start

### Step 1: Generate Additional Images

Run the Python script to create 6 more professional visualizations:

```powershell
python generate_presentation_images.py
```

This will create a `presentation_images/` subdirectory with:

1. `confusion_matrices_comparison.png` - 3-panel confusion matrix comparison
2. `transfer_learning_comparison.png` - Global vs South Africa performance
3. `feature_comparison_across_models.png` - Feature importance comparison
4. `performance_radar_chart.png` - Multi-metric radar chart
5. `sensitivity_specificity_tradeoff.png` - Trade-off scatter plot
6. `model_selection_flowchart.png` - Decision flow diagram

### Step 2: Create Manual Diagrams

You'll need to create 4 additional diagrams manually:

- Data sources map (GBIF, WorldClim, SRTM locations)
- Pipeline architecture diagram (ingestion → enrichment → ML)
- API architecture diagram (FastAPI structure)
- Heatmap screenshot (run `python -m models.xgboost.generate_heatmap_api`)

### Step 3: Build Your Presentation

Follow the **24-slide structure** in `PRESENTATION_GUIDE.md`:

1. Read `MODELS_COMPREHENSIVE_COMPARISON.md` for complete analysis
2. Use the 20 image placeholders marked throughout the document
3. Insert images at the suggested locations
4. Follow the narrative flow: Problem → Solution → Evaluation → Decision

## 📊 Key Metrics Summary

| Model            | Accuracy   | AUC    | F1 Score   | Sensitivity | Specificity | Weighted Score |
| ---------------- | ---------- | ------ | ---------- | ----------- | ----------- | -------------- |
| **XGBoost** ⭐   | **81.27%** | 0.7921 | **0.8836** | 98.1%       | 37%         | **0.8389**     |
| Random Forest    | 75.48%     | 0.8284 | 0.8553     | 100%        | 11%         | 0.7792         |
| XGBoost Enhanced | 69.70%     | 0.6928 | 0.7835     | 75.7%       | 54%         | 0.7364         |

## 🎯 Final Recommendation

**Selected Model: XGBoost (Standard)**

**Why?**

- Best overall balance of accuracy (81.27%) and sensitivity (98.1%)
- Highest weighted score (0.8389) using the scoring formula:
  ```
  Score = 0.25×AUC + 0.25×F1 + 0.15×Accuracy + 0.15×Specificity + 0.10×Sensitivity + 0.10×AvgPrecision
  ```
- Strong transfer learning performance (global → South Africa)
- Production-ready with reasonable false positive rate

## 📋 Presentation Checklist

- [ ] Run `generate_presentation_images.py`
- [ ] Create 4 manual diagrams
- [ ] Review `MODELS_COMPREHENSIVE_COMPARISON.md`
- [ ] Follow 24-slide structure from `PRESENTATION_GUIDE.md`
- [ ] Insert 20 images at placeholder locations
- [ ] Practice presentation flow
- [ ] After approval: Delete unused models (Random Forest & XGBoost Enhanced)

## 📂 File Organization

```
presentation/
├── README.md (this file)
├── MODELS_COMPREHENSIVE_COMPARISON.md (main analysis)
├── PRESENTATION_GUIDE.md (step-by-step instructions)
├── MODEL_RESULTS.md (historical data)
├── FINAL_MODEL_RECOMMENDATION.md (final decision)
├── generate_presentation_images.py (image generator)
├── model_comparison.png (existing comparison)
├── random_forest_images/ (2 images)
├── xgboost_images/ (2 images)
├── xgboost_enhanced_images/ (5 images)
└── presentation_images/ (6 images - to be generated)
```

## 🔗 Related Files (in main project)

- `/models/xgboost/` - Selected model implementation
- `/models/random_forest/` - Alternative model (to be deleted)
- `/models/xgboost_enhanced/` - Alternative model (to be deleted)
- `/data/global_training_ml_ready.csv` - Training dataset
- `/data/local_validation_ml_ready.csv` - Validation dataset

## 💡 Tips for Presentation

1. **Start with the problem**: Why do we need invasive species prediction?
2. **Explain the data pipeline**: How we got from GBIF to ML-ready datasets
3. **Show all 3 models fairly**: Acknowledge strengths and weaknesses
4. **Use visuals heavily**: The 20 images tell the story
5. **Emphasize the decision logic**: Why XGBoost won (not just accuracy)
6. **Discuss trade-offs**: Sensitivity vs specificity, complexity vs performance
7. **End with next steps**: Deployment, monitoring, future improvements

## 📞 Support

For questions or clarifications, refer to:

- `MODELS_COMPREHENSIVE_COMPARISON.md` - Technical deep dive
- `PRESENTATION_GUIDE.md` - Presentation structure
- `/models/xgboost/README.md` - Implementation details

---

**Last Updated**: October 8, 2025
**Status**: Ready for presentation creation
**Next Action**: Run `generate_presentation_images.py`
