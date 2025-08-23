# Invasive Plant Observation Collector

**Status:** Work in Progress

A standalone asynchronous data collection and enrichment service for invasive plant monitoring.
It supports **iNaturalist** species observation retrieval, **NASA POWER API** environmental enrichment, and persistent storage in **MongoDB**.
It forms part of a larger ecological monitoring and spread prediction platform.

---

## Project Overview

This microservice:

* Retrieves geospatial observation data from **iNaturalist** for *Pyracantha angustifolia* within a defined Western Cape, South Africa bounding box.
* Enriches observations with environmental variables (temperature, precipitation, humidity, wind speed, radiation, etc.) from **NASA POWER API**.
* Stores observations and weather data in **MongoDB** for later use in ML training and prediction.
* Provides an API for fetching, refreshing, merging, and exporting datasets.

This service is intended to:

* Run periodically (scheduled job or on-demand)
* Provide **clean, retrainable datasets** for ML models predicting seasonal spread
* Serve as a **data ingestion and enrichment layer** in a multi-service architecture

---

## Current Functionality

* **iNaturalist API Integration**

  * Asynchronous fetching of observation data
  * Filtering by date range, location bounding box, species taxon ID
  * Validation of positional accuracy
  * Cleaned observation structure ready for ML use
  * Optional storage in MongoDB

* **NASA POWER API Integration**

  * Fetch daily weather data for given coordinates & date ranges
  * Enrich iNat observations with environmental context
  * Optional storage in MongoDB

* **Dataset Building & Management**

  * Merge iNat and NASA data into a unified dataset
  * Refresh weather data for stored iNat observations
  * Export merged datasets as CSV
  * Retrieve and manage stored records

* **Prediction & Modeling**

  * **Seasonal Machine Learning**: Enhanced Random Forest models with biological pattern recognition
  * **Peak Season Analysis**: 53% higher invasion risk during flowering months (Autumn)  
  * **Real-time Weather Integration**: Live NASA POWER API data for current predictions
  * **Interactive Heatmaps**: Comparative peak vs off-season invasion risk visualizations
  * **Pyracantha Biology**: Model learns 66% of observations occur during flowering period

* **Machine Learning Experiments** (`experiments/random_forest/`)

  * **Seasonal Random Forest**: Captures flowering/observation timing patterns
  * **Temporal Features**: Distance from peak season, flowering intensity, observation recency
  * **Grid Predictions**: High-resolution invasion risk maps with API integration
  * **Performance**: 71 features, 0.496 accuracy, 1.53x seasonal enhancement ratio

* **FastAPI REST Endpoints**

  * Modular routers for `/observations`, `/weather`, `/datasets`, `/predictions`
  * MongoDB persistence layer
  * Full JSON API responses

---

## API Endpoints

### Status Router (`/status`)

| Method | Endpoint                   | Description                                            |
| ------ | -------------------------- | ------------------------------------------------------ |
| `GET`  | `/status/health`           | Health check - returns service status and timestamp   |
| `GET`  | `/status/service_info`     | Service information - returns API details and version |

---

### Observations Router (`/observations`)

| Method   | Endpoint                         | Description                                                           |
| -------- | -------------------------------- | --------------------------------------------------------------------- |
| `GET`    | `/observations`                  | Fetch all iNat observations (no date filter). Optionally store in DB. |
| `GET`    | `/observations/from`             | Fetch iNat observations from a specific date. Optionally store in DB. |
| `GET`    | `/observations/{observation_id}` | Retrieve a single observation by ID from iNat API.                    |
| `GET`    | `/observations/db`               | Retrieve stored iNat observations from MongoDB.                       |
| `DELETE` | `/observations/db`               | Delete all stored iNat observations.                                  |

---

### Weather Router (`/weather`)

| Method   | Endpoint          | Description                                                                           |
| -------- | ----------------- | ------------------------------------------------------------------------------------- |
| `GET`    | `/weather`        | Fetch NASA POWER weather data for coordinates and date range. Optionally store in DB. |
| `GET`    | `/weather/db`     | Retrieve stored weather data from MongoDB.                                            |
| `GET`    | `/weather/recent` | Retrieve most recent weather records for a given location.                            |
| `DELETE` | `/weather/db`     | Delete all stored weather data.                                                       |

---

### Datasets Router (`/datasets`)

| Method | Endpoint                    | Description                                                              |
| ------ | --------------------------- | ------------------------------------------------------------------------ |
| `GET`  | `/datasets/merge`           | Fetch iNat + NASA data concurrently, store in DB, return merged dataset. |
| `POST` | `/datasets/refresh-weather` | Re-fetch weather for all stored iNat observations and update DB.         |
| `GET`  | `/datasets/export`          | Export merged dataset as CSV.                                            |

---

### Predictions Router (`/predictions`)

| Method | Endpoint                         | Description                                                                     |
| ------ | -------------------------------- | ------------------------------------------------------------------------------- |
| `GET`  | `/predictions/presence_baseline` | Get recent observations that form the presence baseline for predictions.        |
| `GET`  | `/predictions/suitability_map`   | Generate habitat suitability map for the study area with grid predictions.      |
| `GET`  | `/predictions/visualize_map`     | Create an interactive Folium map showing invasion spread predictions.           |

---

## Documentation

This project uses a **hybrid documentation approach** combining FastAPI's automatic API documentation with Sphinx's comprehensive project documentation:

### **Available Documentation**

* **Interactive API Docs**: Live API testing and exploration
  * **Swagger UI**: http://localhost:8000/docs
  * **ReDoc**: http://localhost:8000/redoc
* **Comprehensive Docs**: Sphinx-generated documentation including:
  * **API Reference**: Auto-generated from code docstrings
  * **User Guide**: Setup and usage instructions  
  * **Architecture**: System design and data flow
  * **Testing**: Testing strategies and procedures

### **Quick Start**

```bash
# Build and serve documentation (all platforms)
python build_docs.py --serve --open

# Just build documentation
python build_docs.py

# Clean build and serve
python build_docs.py --clean --serve
```

### **Advanced Options**

```bash
# Check FastAPI server status only
python build_docs.py --no-sphinx

# Custom port for documentation server
python build_docs.py --serve --port 9000

# Get help with all options
python build_docs.py --help
```

### **Manual Build**

```bash
# Manual Sphinx build
sphinx-build -b html docs/ docs/_build/html

# Start FastAPI server for interactive docs
uvicorn app.main:app --reload
```

### **Cross-Platform Compatibility**

The `build_docs.py` script automatically:
* Detects the project structure from any location
* Installs required documentation dependencies
* Works on Windows, macOS, and Linux
* Uses relative paths (no hardcoded system paths)
* Provides comprehensive error handling and status reporting

**For detailed documentation strategy, configuration, and best practices, see the [comprehensive documentation guide](docs/README.md).**

---

## Tech Stack

| Component         | Technology                    |
| ----------------- | ----------------------------- |
| Language          | Python 3.11+                 |
| Web Framework     | FastAPI                       |
| Async HTTP Client | `httpx`                       |
| Data Handling     | `pandas`, `numpy`             |
| Database          | MongoDB (`pymongo`)           |
| Geospatial        | `geopy`, `folium`             |
| Environment       | `python-dotenv`               |
| Logging           | Python `logging`              |
| Concurrency       | `asyncio`                     |
| Documentation     | Sphinx + autodoc              |
| Testing           | pytest + async testing       |
| API Documentation | Swagger UI + ReDoc (FastAPI)  |

---

## Example Workflow

1. **Check Service Health**

```http
GET /api/v1/status/health
GET /api/v1/status/service_info
```

2. **Fetch & Store Observations**

```http
GET /api/v1/observations?store_in_db=true
```

3. **Fetch & Store Weather Data**

```http
GET /api/v1/weather?latitude=-33.9249&longitude=18.4073&start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10&store_in_db=true
```

4. **Merge & Enrich**

```http
GET /api/v1/datasets/merge?start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10&years_back=5
```

5. **Refresh Weather for Stored Observations**

```http
POST /api/v1/datasets/refresh-weather
```

6. **Export Dataset**

```http
GET /api/v1/datasets/export
```

7. **Generate Predictions**

```http
GET /api/v1/predictions/presence_baseline?days_back=100
GET /api/v1/predictions/suitability_map?days_back=100&grid_resolution=0.5
GET /api/v1/predictions/visualize_map?days_back=100&grid_resolution=0.5&save_file=true
```

---

## Future Steps

* [x] **Documentation & Testing**
  * [x] Complete Sphinx documentation setup with auto-generated API docs
  * [x] Comprehensive test coverage (131 tests) with error handling
  * [ ] Add performance benchmarking documentation
* [ ] **Data Pipeline Enhancements**
  * [ ] Add scheduled background jobs for periodic data sync
  * [ ] Introduce update-skipping for already up-to-date weather records
  * [ ] Enhance filtering and query parameters
* [x] **Machine Learning & Prediction**
  * [x] Seasonal Random Forest models with biological pattern recognition
  * [x] Interactive invasion risk heatmaps with real weather data
  * [x] Peak vs off-season comparative analysis (+52.9% seasonal enhancement)
  * [x] Pyracantha flowering cycle integration (66% autumn observations)
* [ ] **Deployment & Infrastructure**
  * [ ] Dockerize and prepare for deployment
  * [ ] User authentication and API keys
  * [x] Integration with ML training pipeline
* [ ] **System Reliability**
  * [ ] Add comprehensive error handling and retry mechanisms
  * [ ] Implement data validation and quality checks
  * [ ] Add monitoring and alerting capabilities

---

## Maintainer

This is a component by **Martinus Christoffel Wolmarans** and will contribute to the larger Invasive Plant Monitoring System final year project.