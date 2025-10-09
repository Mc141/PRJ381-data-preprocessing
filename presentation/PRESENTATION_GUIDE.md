# Presentation Guide - Model Comparison Analysis

## Quick Reference for Your Presentation

### What I Created

I've compiled all model analysis into: **`MODELS_COMPREHENSIVE_COMPARISON.md`**

This document contains:
- ✅ Complete comparison of all 3 models (Random Forest, XGBoost, XGBoost Enhanced)
- ✅ Feature importance analysis with ecological interpretation
- ✅ Performance metrics with business context
- ✅ Transfer learning effectiveness analysis
- ✅ 20 placeholder markers for where to insert images in your presentation
- ✅ Detailed methodology for each model
- ✅ Final decision logic and rationale

---

## Available Images (Already Generated)

### Root Directory
- `model_comparison.png` - Bar chart comparing all 3 models across metrics

### Random Forest Model
- `models/random_forest/feature_importance.png`
- `models/random_forest/roc_curve.png`

### XGBoost (Standard)
- `models/xgboost/feature_importance.png`
- `models/xgboost/roc_curve.png`

### XGBoost Enhanced
- `models/xgboost_enhanced/feature_importance.png`
- `models/xgboost_enhanced/roc_curve.png`
- `models/xgboost_enhanced/shap_summary.png`
- `models/xgboost_enhanced/shap_feature_importance.png`
- `models/xgboost_enhanced/precision_recall_curve.png`

---

## Images You Need to Create (7 total)

### Placeholder 1: Data Sources Diagram
**Content:** Map showing:
- Global GBIF occurrence points
- South African validation region
- WorldClim raster coverage
- SRTM elevation tiles

**Tool:** Use PowerPoint, draw.io, or Python (matplotlib + cartopy)

### Placeholder 2: Pipeline Architecture
**Content:** Flowchart showing:
```
GBIF API → WorldClim → SRTM → ML Dataset → Model Training → Validation
```
**Tool:** draw.io, Lucidchart, or PowerPoint

### Placeholder 4: Confusion Matrices Side-by-Side
**Content:** 3 confusion matrices as heatmaps
```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Random Forest
cm_rf = [[11, 89], [0, 263]]
sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Random Forest')

# XGBoost
cm_xgb = [[37, 63], [5, 258]]
sns.heatmap(cm_xgb, annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('XGBoost')

# Enhanced
cm_enh = [[54, 46], [64, 199]]
sns.heatmap(cm_enh, annot=True, fmt='d', ax=axes[2], cmap='Oranges')
axes[2].set_title('XGBoost Enhanced')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300)
```

### Placeholder 14: Transfer Learning Performance
**Content:** Chart showing training vs validation accuracy for each model
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'XGBoost', 'XGBoost Enhanced']
training = [0.93, 0.85, 0.78]
validation = [0.7548, 0.8127, 0.6970]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, training, width, label='Training', color='skyblue')
ax.bar(x + width/2, validation, width, label='Validation', color='coral')

ax.set_ylabel('Accuracy')
ax.set_title('Transfer Learning: Training vs Validation Performance')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.savefig('transfer_learning_comparison.png', dpi=300)
```

### Placeholder 15: Feature Comparison Across Models
**Content:** Grouped bar chart showing top 10 features importance across models
```python
import pandas as pd
import matplotlib.pyplot as plt

# Top features data
data = {
    'Feature': ['longitude', 'latitude', 'elevation', 'bio4', 'bio6', 'bio12', 'bio1', 'bio14', 'bio15', 'bio13'],
    'Random Forest': [0.2769, 0.2278, 0.0738, 0.0505, 0.0511, 0.0474, 0.0476, 0.0516, 0.0467, 0.0285],
    'XGBoost': [0.1565, 0.0932, 0.0678, 0.1126, 0.0834, 0.0748, 0.0738, 0.0690, 0.0700, 0.0649],
    'XGBoost Enhanced': [0.0289, 0.0192, 0.0, 0.0, 0.0, 0.0, 0.0212, 0.0, 0.0185, 0.0]
}

df = pd.DataFrame(data)
df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance Comparison Across Models')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.legend(title='Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_comparison_across_models.png', dpi=300)
```

### Placeholder 16: ROC Curves Overlay
**Content:** All 3 ROC curves on same plot
```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated ROC curves (you'll need actual data from model outputs)
plt.figure(figsize=(8, 6))

# Random Forest
fpr_rf = np.array([0, 0.11, 0.5, 0.89, 1.0])
tpr_rf = np.array([0, 0.3, 0.7, 0.95, 1.0])
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC=0.8284)', linewidth=2)

# XGBoost
fpr_xgb = np.array([0, 0.21, 0.4, 0.63, 1.0])
tpr_xgb = np.array([0, 0.4, 0.75, 0.98, 1.0])
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC=0.7921)', linewidth=2)

# Enhanced
fpr_enh = np.array([0, 0.46, 0.54, 0.76, 1.0])
tpr_enh = np.array([0, 0.5, 0.76, 0.90, 1.0])
plt.plot(fpr_enh, tpr_enh, label='XGBoost Enhanced (AUC=0.6928)', linewidth=2)

# Diagonal reference
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - All Models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curves_overlay.png', dpi=300)
```

### Placeholder 18: Confusion Matrix Heatmaps
(Same as Placeholder 4, but styled as heatmap)

### Placeholder 19: API Architecture Diagram
**Content:** Show:
- FastAPI endpoints
- Data flow: Coordinates → Environmental Extraction → Model Prediction → Heatmap
- Include rate limiting, async processing

**Tool:** draw.io or PowerPoint

### Placeholder 20: Example Heatmap Output
**Action:** Run the heatmap generator and take screenshot
```bash
# Make sure API is running
uvicorn app.main:app --reload

# In another terminal
python -m models.xgboost.generate_heatmap_api --grid_size 20
```
Then screenshot the HTML output

---

## Presentation Structure Suggestion

### Slide 1: Title
- Project name
- Your name
- Date

### Slide 2: Problem Statement
- What: Pyracantha invasion prediction
- Why: Conservation & resource management
- How: Transfer learning with ML

### Slide 3: Data Sources (Placeholder 1)
- GBIF occurrences
- WorldClim climate
- SRTM elevation

### Slide 4: Data Pipeline (Placeholder 2)
- Show architecture
- Mention async processing, rate limiting

### Slide 5: Feature Engineering
- 13 base features
- Bioclimate variables explanation
- Temporal encoding (sin/cos month)

### Slide 6: Models Overview
- Random Forest
- XGBoost
- XGBoost Enhanced

### Slide 7: Model Comparison (Placeholder 3)
- Insert `model_comparison.png`
- Highlight XGBoost wins

### Slide 8-9: Random Forest Deep Dive
- Feature importance (Placeholder 5)
- ROC curve (Placeholder 6)
- Problems: geographic bias, poor specificity

### Slide 10-11: XGBoost Deep Dive
- Feature importance (Placeholder 7)
- ROC curve (Placeholder 8)
- Strengths: balanced, fast, production-ready

### Slide 12-14: Enhanced Model Deep Dive
- Feature engineering details
- Feature importance (Placeholder 9)
- SHAP analysis (Placeholder 10, 11)
- Precision-Recall (Placeholder 12)
- Why it wasn't selected

### Slide 15: Confusion Matrices (Placeholder 4)
- Show all 3 side-by-side
- Explain sensitivity/specificity trade-off

### Slide 16: Transfer Learning (Placeholder 14)
- Training vs validation performance
- Show XGBoost generalizes best

### Slide 17: Feature Analysis (Placeholder 15)
- Compare feature importance across models
- Ecological interpretation

### Slide 18: ROC Comparison (Placeholder 16)
- All curves overlaid
- Annotate trade-offs

### Slide 19: Final Decision Logic
- Weighted scoring system
- XGBoost wins: 0.8389 score
- Rationale: balance, speed, simplicity

### Slide 20: Implementation (Placeholder 19)
- API architecture
- How to use the model
- Example prediction

### Slide 21: Live Demo (Placeholder 20)
- Show heatmap output
- Explain risk visualization
- Interactive features

### Slide 22: Conclusions
- 81% accuracy achieved
- 98% sensitivity (catches invasions)
- Production-ready system

### Slide 23: Future Work
- Temporal dynamics
- Additional features (soil, land use)
- Active learning with field data

### Slide 24: Q&A
- Thank you

---

## Python Script to Generate Missing Images

Create this file: `generate_presentation_images.py`

```python
"""Generate all missing images for presentation"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 1. Confusion Matrices Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cm_rf = np.array([[11, 89], [0, 263]])
cm_xgb = np.array([[37, 63], [5, 258]])
cm_enh = np.array([[54, 46], [64, 199]])

for ax, cm, title, cmap in zip(axes, [cm_rf, cm_xgb, cm_enh], 
                                ['Random Forest', 'XGBoost', 'XGBoost Enhanced'],
                                ['Blues', 'Greens', 'Oranges']):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap, 
                xticklabels=['Absent', 'Present'],
                yticklabels=['Absent', 'Present'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', bbox_inches='tight')
print("✅ Created: confusion_matrices_comparison.png")

# 2. Transfer Learning Comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Random Forest', 'XGBoost', 'XGBoost Enhanced']
training = [0.93, 0.85, 0.78]
validation = [0.7548, 0.8127, 0.6970]
transfer_gap = [t - v for t, v in zip(training, validation)]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, training, width, label='Training Accuracy', 
               color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, validation, width, label='Validation Accuracy', 
               color='coral', edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Transfer Learning: Global Training vs South Africa Validation', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', bbox_inches='tight')
print("✅ Created: transfer_learning_comparison.png")

# 3. Feature Importance Comparison
data = {
    'Feature': ['longitude', 'latitude', 'elevation', 'bio4', 'bio6', 'bio12', 'bio1', 'bio14', 'bio15', 'bio13'],
    'Random Forest': [0.2769, 0.2278, 0.0738, 0.0505, 0.0511, 0.0474, 0.0476, 0.0516, 0.0467, 0.0285],
    'XGBoost': [0.1565, 0.0932, 0.0678, 0.1126, 0.0834, 0.0748, 0.0738, 0.0690, 0.0700, 0.0649],
}

df = pd.DataFrame(data)
ax = df.set_index('Feature').plot(kind='bar', figsize=(12, 6), 
                                   color=['skyblue', 'coral'],
                                   edgecolor='black', width=0.8)
plt.title('Feature Importance Comparison: Random Forest vs XGBoost', 
          fontsize=14, fontweight='bold')
plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
plt.xlabel('Feature', fontsize=12, fontweight='bold')
plt.legend(title='Model', fontsize=11, title_fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_comparison_across_models.png', bbox_inches='tight')
print("✅ Created: feature_comparison_across_models.png")

# 4. ROC Curves Overlay (Simplified - you'll need actual FPR/TPR data)
print("\n⚠️  For ROC curves overlay, you need actual model predictions.")
print("   Run this to get data:")
print("   - Load each model")
print("   - Get predictions on validation set")
print("   - Calculate FPR/TPR with sklearn.metrics.roc_curve()")
print("   - Plot all three curves together")

print("\n✅ Generated 3 out of 7 missing images!")
print("\nStill need to create:")
print("  • Data Sources Map (use mapping library or PowerPoint)")
print("  • Pipeline Architecture Diagram (use draw.io)")
print("  • ROC Curves Overlay (need actual model predictions)")
print("  • API Architecture Diagram (use draw.io)")
print("  • Screenshot of heatmap output (run generate_heatmap_api.py)")
```

---

## Next Steps

1. **Read the comprehensive comparison**: `MODELS_COMPREHENSIVE_COMPARISON.md`
2. **Run the image generator**: `python generate_presentation_images.py`
3. **Create remaining diagrams** using draw.io or PowerPoint
4. **Generate heatmap screenshot** by running the API
5. **Build your presentation** using the structure above
6. **Insert images** at the 20 placeholder locations

---

## Quick Decision Summary

**Why XGBoost Wins:**
- ✅ Highest accuracy (81.27%)
- ✅ Best F1 score (0.8836)
- ✅ Excellent sensitivity (98.1%)
- ✅ Acceptable specificity (37%)
- ✅ Fast training (~3 min)
- ✅ Simple features (13 vars)
- ✅ Production-ready

**Why Not Random Forest:**
- ❌ Poor specificity (11%)
- ❌ Geographic bias (50% lat/lon)
- ❌ Overfits to training region

**Why Not Enhanced:**
- ❌ Lowest accuracy (69.7%)
- ❌ Complex (46+ features)
- ❌ Slow training (~8 min)
- ❌ No performance gain

---

## Contact & Support

If you need help understanding any section of the comprehensive comparison document, let me know!

Good luck with your presentation! 🚀
