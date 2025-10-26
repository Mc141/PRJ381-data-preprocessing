"""
Generate presentation images for model comparison
Run this script to create missing visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Create output directory
output_dir = 'presentation_images'
os.makedirs(output_dir, exist_ok=True)

print("Generating presentation images...")
print("=" * 60)

# ============================================================================
# 1. Confusion Matrices Comparison (3-panel)
# ============================================================================
print("\nCreating confusion matrices comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cm_rf = np.array([[11, 89], [0, 263]])
cm_xgb = np.array([[37, 63], [5, 258]])
cm_enh = np.array([[54, 46], [64, 199]])

labels = ['Absent', 'Present']

for ax, cm, title, cmap in zip(
    axes, 
    [cm_rf, cm_xgb, cm_enh], 
    ['Random Forest\n(Specificity: 11%)', 'XGBoost\n(Specificity: 37%)', 'XGBoost Enhanced\n(Specificity: 54%)'],
    ['Blues', 'Greens', 'Oranges']
):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap, 
                xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison - All Models', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: confusion_matrices_comparison.png")

# ============================================================================
# 2. Transfer Learning Performance
# ============================================================================
print("\nCreating transfer learning comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Random Forest', 'XGBoost', 'XGBoost Enhanced']
training = [0.93, 0.85, 0.78]
validation = [0.7548, 0.8127, 0.6970]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, training, width, label='Global Training', 
               color='#4A90E2', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, validation, width, label='South Africa Validation', 
               color='#E85D75', edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add transfer gap annotations
for i, (t, v) in enumerate(zip(training, validation)):
    gap = t - v
    ax.annotate(f'Gap: {gap:.3f}', 
                xy=(i, (t + v) / 2), 
                xytext=(i + 0.4, (t + v) / 2),
                fontsize=9, color='gray', style='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Transfer Learning Effectiveness\nGlobal Training vs South African Validation', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
ax.text(2.5, 0.81, 'Good Performance', fontsize=9, color='green', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'transfer_learning_comparison.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: transfer_learning_comparison.png")

# ============================================================================
# 3. Feature Importance Comparison
# ============================================================================
print("\nCreating feature importance comparison...")

data = {
    'Feature': ['longitude', 'latitude', 'bio4', 'elevation', 'bio6', 
                'bio12', 'bio1', 'bio14', 'bio15', 'bio13'],
    'Random Forest': [0.2769, 0.2278, 0.0505, 0.0738, 0.0511, 
                      0.0474, 0.0476, 0.0516, 0.0467, 0.0285],
    'XGBoost': [0.1565, 0.0932, 0.1126, 0.0678, 0.0834, 
                0.0748, 0.0738, 0.0690, 0.0700, 0.0649],
}

df = pd.DataFrame(data)
ax = df.set_index('Feature').plot(
    kind='bar', 
    figsize=(14, 7), 
    color=['#4A90E2', '#E85D75'],
    edgecolor='black', 
    width=0.8,
    linewidth=1.5
)

plt.title('Feature Importance Comparison\nRandom Forest vs XGBoost', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Importance Score', fontsize=13, fontweight='bold')
plt.xlabel('Feature', fontsize=13, fontweight='bold')
plt.legend(title='Model', fontsize=12, title_fontsize=13, 
           loc='upper right', frameon=True, shadow=True)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.ylim(0, 0.3)

# Add annotations for key differences
ax.annotate('Geographic\nDominance', xy=(0, 0.28), xytext=(0.5, 0.32),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.annotate('Climate\nFocus', xy=(2, 0.113), xytext=(3.5, 0.18),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_comparison_across_models.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: feature_comparison_across_models.png")

# ============================================================================
# 4. Model Performance Radar Chart
# ============================================================================
print("\nCreating performance radar chart...")

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

categories = ['Accuracy', 'AUC', 'F1 Score', 'Sensitivity', 'Specificity']
N = len(categories)

# Model scores
rf_scores = [0.7548, 0.8284, 0.8553, 1.0000, 0.1100]
xgb_scores = [0.8127, 0.7921, 0.8836, 0.9810, 0.3700]
enh_scores = [0.6970, 0.6928, 0.7835, 0.7567, 0.5400]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
rf_scores += rf_scores[:1]
xgb_scores += xgb_scores[:1]
enh_scores += enh_scores[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, rf_scores, 'o-', linewidth=2, label='Random Forest', color='#4A90E2')
ax.fill(angles, rf_scores, alpha=0.15, color='#4A90E2')

ax.plot(angles, xgb_scores, 'o-', linewidth=2, label='XGBoost', color='#50C878')
ax.fill(angles, xgb_scores, alpha=0.15, color='#50C878')

ax.plot(angles, enh_scores, 'o-', linewidth=2, label='XGBoost Enhanced', color='#E85D75')
ax.fill(angles, enh_scores, alpha=0.15, color='#E85D75')

# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

# Set y-axis limits
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True, shadow=True)
plt.title('Model Performance Comparison\nRadar Chart', 
          fontsize=16, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_radar_chart.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: performance_radar_chart.png")

# ============================================================================
# 5. Sensitivity vs Specificity Trade-off
# ============================================================================
print("\nCreating sensitivity-specificity trade-off...")

fig, ax = plt.subplots(figsize=(10, 8))

models = ['Random Forest', 'XGBoost', 'XGBoost Enhanced']
sensitivity = [1.0000, 0.9810, 0.7567]
specificity = [0.1100, 0.3700, 0.5400]
colors = ['#4A90E2', '#50C878', '#E85D75']
sizes = [300, 400, 300]  # XGBoost larger (recommended)

for i, (model, sens, spec, color, size) in enumerate(zip(models, sensitivity, specificity, colors, sizes)):
    ax.scatter(spec, sens, s=size, alpha=0.6, c=color, edgecolors='black', linewidth=2)
    ax.annotate(model, (spec, sens), xytext=(10, 10 if i != 1 else -20), 
                textcoords='offset points', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))

# Add diagonal line (balanced point)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Balance')

# Add ideal region
ideal_region = plt.Rectangle((0.7, 0.7), 0.3, 0.3, alpha=0.1, color='green')
ax.add_patch(ideal_region)
ax.text(0.85, 0.85, 'Ideal\nRegion', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='green', alpha=0.5)

ax.set_xlabel('Specificity (True Negative Rate)', fontsize=13, fontweight='bold')
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=13, fontweight='bold')
ax.set_title('Sensitivity vs Specificity Trade-off\n"Perfect" model would be top-right corner', 
             fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11)

# Add annotations
ax.annotate('High False\nPositives', xy=(0.11, 1.0), xytext=(0.2, 0.85),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', style='italic')

ax.annotate('Best\nBalance', xy=(0.37, 0.981), xytext=(0.5, 0.75),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sensitivity_specificity_tradeoff.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: sensitivity_specificity_tradeoff.png")

# ============================================================================
# 6. Model Selection Decision Tree
# ============================================================================
print("\nCreating model selection flowchart...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Helper function to draw boxes
def draw_box(x, y, width, height, text, color, is_decision=False):
    if is_decision:
        # Diamond shape for decisions
        points = np.array([
            [x, y + height/2],
            [x + width/2, y + height],
            [x + width, y + height/2],
            [x + width/2, y]
        ])
        polygon = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(polygon)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', wrap=True)
    else:
        rect = plt.Rectangle((x, y), width, height, facecolor=color, 
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', wrap=True)

# Draw flowchart
draw_box(3.5, 8.5, 3, 1, 'Start:\nModel Selection', '#E8F4F8', False)

# Decision 1
draw_box(3.75, 6.5, 2.5, 1.5, 'Need highest\naccuracy?', '#FFE6CC', True)
ax.arrow(5, 8.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')

# Yes branch
draw_box(0.5, 4.5, 2, 1, 'XGBoost\n81% Accuracy', '#C6E0B4', False)
ax.arrow(4.5, 6.8, -2.5, -1, head_width=0.2, head_length=0.1, fc='green', ec='green')
ax.text(2.5, 6, 'Yes', fontsize=10, color='green', fontweight='bold')

# No branch (continue)
ax.arrow(5.8, 7, 1, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
draw_box(6.5, 5, 2.5, 1.5, 'Need best\nspecificity?', '#FFE6CC', True)

# Yes branch
draw_box(7, 2.5, 2, 1, 'XGBoost Enhanced\n54% Specificity', '#F4B084', False)
ax.arrow(7.5, 5, 0, -1.3, head_width=0.2, head_length=0.1, fc='orange', ec='orange')
ax.text(7.8, 4, 'Yes', fontsize=10, color='orange', fontweight='bold')

# No branch
ax.arrow(6, 5.5, -2, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
draw_box(3, 3.5, 2, 1, 'Random Forest\nHigh Sensitivity', '#FFB3BA', False)
ax.text(5, 4.8, 'No', fontsize=10, color='red', fontweight='bold')

# Recommendation
draw_box(0.5, 0.5, 9, 1.5, '★ RECOMMENDED: XGBoost ★\nBest Overall Balance\n81% Accuracy | 98% Sensitivity | 37% Specificity', 
         '#90EE90', False)

plt.title('Model Selection Decision Flow', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_selection_flowchart.png'), 
            bbox_inches='tight', dpi=300)
print("   DONE: model_selection_flowchart.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Successfully generated 6 presentation images!")
print(f"Images saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. confusion_matrices_comparison.png")
print("  2. transfer_learning_comparison.png")
print("  3. feature_comparison_across_models.png")
print("  4. performance_radar_chart.png")
print("  5. sensitivity_specificity_tradeoff.png")
print("  6. model_selection_flowchart.png")
print("\n" + "=" * 60)
print("\nStill need to create manually:")
print("  * Data sources map (use mapping library or PowerPoint)")
print("  * Pipeline architecture diagram (use draw.io or Lucidchart)")
print("  * API architecture diagram (use draw.io or Lucidchart)")
print("  * Heatmap screenshot (run: python -m models.xgboost.generate_heatmap_api)")
print("\nTip: Use these images in your presentation slides!")
print("=" * 60)
