# PRJ381 Data Preprocessing & Species Distribution Modeling API

**Status:** Production Ready (2025)

A robust FastAPI service for global invasive species data processing, environmental enrichment, and transfer learning. This API supports ML-ready dataset creation, model training, and risk prediction, with full Docker/Heroku deployment support.

---

## Project Overview

- **Global Data Integration:** Retrieve and process worldwide species occurrence records (GBIF)
- **Environmental Enrichment:** Integrate WorldClim climate and SRTM elevation
- **Transfer Learning Datasets:** Create global training and local validation datasets
- **ML-Ready Exports:** Standardized 13-feature CSV for XGBoost training
- **Async Processing:** Efficient handling of large datasets with progress tracking

### Transfer Learning Workflow

1. **Train globally** on ~7.7K Pyracantha occurrences with environmental data
2. **Validate locally** on South African subset
3. **Export** 13 standardized features for all datasets
4. **Deploy** models for global risk prediction

---

## Main Routers & Their Roles

- **/status**: Health checks and system info
- **/environmental**: Batch extraction of climate and elevation data
- **/datasets**: ML-ready dataset generation and export
- **/predictions**: Model training and risk map generation

---

## Core API Endpoints (Detailed)

### 1. Health Check

**GET /api/v1/status/health**

- **Purpose:** Check API and dependency health (FastAPI, external services, storage)
- **Parameters:** None
- **Returns:** `{ "status": "healthy", "timestamp": "..." }`

### 2. Environmental Data Extraction

**POST /api/v1/environmental/extract-batch**

- **Purpose:** Extract climate and elevation data for a batch of coordinates
- **Body:**
  ```json
  {
    "coordinates": [
      { "latitude": -33.925, "longitude": 18.424 },
      { "latitude": -33.895, "longitude": 18.505 }
    ]
  }
  ```
- **Query Parameters:**
  - `variables` (list of str, optional): Climate variables to extract (default: `["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]`)
- **Returns:**
  - `request_info`: Metadata (timestamp, endpoint, variables)
  - `results`: List of dicts with climate variables and elevation for each coordinate

### 3. Generate ML-Ready Datasets

**POST /api/v1/datasets/generate-ml-ready-files**

- **Purpose:** Generate and save ML-ready CSVs for global training and local validation.
- **Query Parameters:**
  - `max_global` (int, optional): Maximum number of global records (omit for all)
  - `max_local` (int, optional): Maximum number of local South African records (omit for all)
  - `batch_size` (int, default 100): Batch size for environmental data extraction
  - `verbose` (bool, default False): Enable verbose logging output
- **Returns:**
  - JSON with status, written file paths, and a success message

### 4. Generate XGBoost Heatmap

**POST /api/v1/predictions/generate-xgboost-heatmap**

- **Purpose:** Generate a high-res invasion risk heatmap using XGBoost and real-time environmental data
- **Query Parameters:**
  - `region` (str, default `western_cape_core`): Predefined or custom region
  - `lat_min`, `lat_max`, `lon_min`, `lon_max` (float, required for custom region): Geographic bounds
  - `grid_size` (int, default 20): Grid resolution (5-100)
  - `month` (int, default 3): Month for prediction (1-12)
  - `batch_size` (int, default 20): API batch size
  - `rate_limit_delay` (float, default 1.0): Delay between batches (seconds)
  - `include_stats` (bool, default True): Include stats panel
  - `return_html` (bool, default True): Return HTML in response
  - `save_file` (bool, default False): Save HTML to disk
  - `download` (bool, default False): Download HTML as file
- **Returns:**
  - If `download=True`: HTML file
  - Else: JSON with HTML content, stats, and file info

### 5. Train XGBoost Model

**POST /api/v1/predictions/train-xgboost-model**

- **Purpose:** Train XGBoost model using the standardized pipeline
- **Query Parameters:**
  - `save_artifacts` (bool, default True): Save model and plots
  - `return_metrics` (bool, default True): Include metrics in response
  - `limit_rows` (int, default 0): Subsample training data (0 = all)
- **Returns:**
  - Training metrics, artifact locations, and status

---

## Example ML Workflow

```python
import pandas as pd
import xgboost as xgb
# Load data
train = pd.read_csv('data/global_training_ml_ready.csv')
val = pd.read_csv('data/local_validation_ml_ready.csv')
# Train XGBoost
model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100)
model.fit(train.drop('presence', axis=1), train['presence'])
# Validate
preds = model.predict(val.drop('presence', axis=1))
accuracy = (preds == val['presence']).mean()
print(f"Accuracy: {accuracy:.2%}")
```

---

## ML Features (13)

- **Location:** latitude, longitude
- **Topographic:** elevation (SRTM 30m)
- **Climate:** bio1, bio4, bio5, bio6, bio12, bio13, bio14, bio15 (WorldClim v2.1)
- **Temporal:** month_sin, month_cos (cyclical encoding)

---

## Deployment (Docker & Heroku)

### Local Docker

```bash
# Build and run locally
$ docker build -t prj381-api .
$ docker run -p 8000:8000 prj381-api
# API: http://localhost:8000/docs
```

### Heroku (Container Stack)

1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Log in:
   ```
   heroku login
   heroku container:login
   ```
3. Create/set up app:
   ```
   heroku create your-app-name
   heroku stack:set container -a your-app-name
   ```
4. Build and push:
   ```
   $env:DOCKER_DEFAULT_PLATFORM="linux/amd64"
   heroku container:push web -a your-app-name
   heroku container:release web -a your-app-name
   ```
5. Open:
   ```
   heroku open -a your-app-name
   # https://your-app-name.herokuapp.com/
   ```

**Deployment files:**

- `Dockerfile`, `requirements.txt`, `Procfile`, `.dockerignore`

> **Note:** Heroku Container Registry may require Docker Desktop <= 4.22.0 for compatibility.

---

## Documentation

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Sphinx Docs:** `docs/` (build with `python build_docs.py`)

---

## Tech Stack

| Component     | Technology            |
| ------------- | --------------------- |
| Language      | Python 3.12+          |
| Web Framework | FastAPI               |
| Geospatial    | `geopy`, `folium`     |
| ML            | scikit-learn, xgboost |
| Deployment    | Docker, Heroku        |
| Docs          | Sphinx, OpenAPI       |
