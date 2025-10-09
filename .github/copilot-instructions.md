# GitHub Copilot Instructions for PRJ381 Species Distribution Modeling

## Project Overview

This is a FastAPI service for global invasive species data processing, environmental enrichment, and transfer learning. The core workflow follows a **data pipeline**: `status → environmental → datasets → predictions` for creating ML-ready datasets from GBIF occurrences enriched with WorldClim climate and SRTM elevation data.

## Architecture & Data Flow

### Core Pipeline Structure

- **`app/main.py`**: FastAPI app with 4 main routers following dependency order
- **`app/routers/`**: API endpoints organized by function (status, environmental, datasets, predictions)
- **`app/services/`**: Business logic for data extraction and processing
- **`data/`**: ML-ready outputs (`global_training_ml_ready.csv`, `local_validation_ml_ready.csv`)
- **`models/`**: XGBoost models (xgboost, xgboost_enhanced) with training scripts

### Key Data Sources & Standards

- **Species Data**: GBIF Global Database (via `gbif_fetcher.py`)
- **Climate Data**: WorldClim v2.1 GeoTIFFs (via `worldclim_extractor.py`)
- **Elevation Data**: SRTM 30m via Open-Topo-Data API (via `elevation_extractor.py`)
- **ML Features**: Standardized 17-feature format (location, climate, temporal, topographic)

## Development Patterns

### API Router Dependencies

Follow the dependency chain: `/status` → `/environmental` → `/datasets` → `/predictions`. Environmental extraction is required for both datasets generation and predictions heatmaps.

### Data Integrity Policy

**REAL DATA ONLY**: No fake/dummy environmental values. Missing data returns `None/NaN`, never fabricated values. All data sources must be clearly labeled and trackable.

### Async Processing Pattern

Use batch processing with progress tracking for large datasets:

```python
# From generate_ml_ready_datasets.py
async def process_batch(coordinates, batch_size=100):
    # Split into batches, track progress
```

### Model Training Structure

Each XGBoost model variant (`xgboost/`, `xgboost_enhanced/`) follows the pattern:

- `train_model.py`: Training script with hyperparameter tuning
- `model.pkl` + `optimal_threshold.pkl`: Saved artifacts
- `feature_importance.png`, `roc_curve.png`: Visualization outputs
- `generate_heatmap_api.py`: API integration for predictions

## Key Commands & Workflows

### Local Development

```bash
# Start API server
uvicorn app.main:app --reload --port 8000

# Generate ML datasets directly
python -m app.services.generate_ml_ready_datasets --max-global 2000 --max-local 500 --batch-size 100 --verbose

# Train models (from models/ subdirectories)
python train_model.py
```

### Docker Deployment

```bash
# Build with production requirements
docker build -t prj381-api .
docker run -p 8000:8000 prj381-api
```

### Testing

```bash
# Run API tests
pytest tests/ -v

# Test specific router
pytest tests/test_predictions.py -v
```

## File Conventions

### Service Module Pattern

- `*_extractor.py`: External data source integrations (WorldClim, SRTM, GBIF)
- `generate_*.py`: Data processing pipelines
- Keep async/await for I/O-bound operations

### Model Training Pattern

- Scripts are executable from their directory: `python train_model.py`
- Always save both model and optimal threshold: `model.pkl`, `optimal_threshold.pkl`
- Generate standard plots: `feature_importance.png`, `roc_curve.png`

### API Response Standards

- Include `request_info` metadata in responses
- Use consistent error handling with FastAPI HTTPException
- Return structured JSON with status, data, and metadata fields

## External Dependencies & Rate Limits

- **Open-Topo-Data API**: 1-second rate limiting for elevation extraction
- **WorldClim GeoTIFFs**: Must be present in `data/worldclim/` directory
- **GBIF API**: No rate limiting, but use batch processing for large requests

## When Making Changes

1. **Router changes**: Update corresponding test in `tests/test_*.py`
2. **New endpoints**: Add to appropriate router with proper tags and documentation
3. **Model updates**: Use XGBoost variants (basic → enhanced) with consistent artifact patterns
4. **Data pipeline changes**: Ensure backward compatibility with 17-feature ML format
