# Rule-Based Prediction Baseline

This directory contains rule-based prediction methods that serve as a baseline comparison for the machine learning approaches.

## Overview

Rule-based methods provide a simple, interpretable baseline for invasion prediction using basic environmental thresholds and heuristics. These serve as a comparison point to demonstrate the superior performance of the Random Forest machine learning model.

## Generated Files

- **Basic Prediction Maps**: Simple threshold-based invasion predictions
- **Static Risk Assessment**: Rule-based environmental suitability mapping

## File Naming Convention

- `invasion_prediction_map_YYYYMMDD_HHMMSS.html` - Rule-based prediction maps with timestamp

## Comparison with ML Model

### Rule-Based Approach
- **Method**: Simple environmental thresholds
- **Performance**: Basic risk assessment
- **Interpretability**: High (clear rules)
- **Adaptability**: Limited (manual rule updates)

### Random Forest ML Model
- **Method**: Advanced machine learning with 400 decision trees
- **Performance**: 95.2% accuracy, 83.5% AUC
- **Interpretability**: Good (feature importance analysis)
- **Adaptability**: High (learns from new data)

## Scientific Value

Rule-based methods provide:
- **Baseline comparison**: Demonstrates ML model superiority
- **Interpretable predictions**: Simple threshold-based logic
- **Quick implementation**: Rapid prototyping capability
- **Domain knowledge integration**: Expert rule incorporation

The dramatic performance improvement from rule-based (~random) to ML (95%+) demonstrates the value of advanced machine learning approaches for ecological prediction.

## Note

This directory contains baseline methods for comparison purposes. The production system uses the high-performance Random Forest model in `../random_forest/` which achieves 95%+ accuracy.

---

*Baseline comparison methods - see `../random_forest/` for production model*
