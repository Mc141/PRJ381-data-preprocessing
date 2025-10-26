# PRJ381 Data Preprocessing Project - Technical Overview

## Executive Summary

This project implements a comprehensive data preprocessing and machine learning pipeline for species distribution modeling, specifically designed for invasive species risk assessment. The system processes global biodiversity occurrence data, enriches it with environmental variables, and generates machine learning-ready datasets for training predictive models using transfer learning principles.

## Project Purpose

The primary objective is to create a data preprocessing pipeline that:

1. Fetches global species occurrence records from the Global Biodiversity Information Facility (GBIF)
2. Enriches occurrence data with environmental variables (climate and elevation)
3. Generates standardized machine learning-ready datasets for model training
4. Supports transfer learning from global to local validation datasets
5. Provides API endpoints for environmental data extraction and model prediction

## Technology Stack

### Programming Languages

- **Python 3.12**: Primary programming language for all system components

### Core Frameworks and Libraries

#### Web Framework

- **FastAPI 0.116.1**: Modern, high-performance web framework for building APIs with automatic OpenAPI documentation
- **Starlette 0.47.2**: ASGI framework that powers FastAPI
- **Uvicorn 0.35.0**: ASGI server for running the FastAPI application
- **Pydantic 2.11.3**: Data validation using Python type annotations
- **python-multipart 0.0.20**: Support for file uploads and form data

#### Asynchronous HTTP and File Operations

- **aiohttp 3.11.16**: Asynchronous HTTP client for API requests
- **aiofiles 24.1.0**: Asynchronous file operations
- **httpx 0.28.1**: Modern HTTP client library
- **httpcore 1.0.8**: Low-level HTTP transport
- **anyio 4.9.0**: Asynchronous networking and concurrency

#### Data Processing and Numerical Computing

- **NumPy 2.2.6**: Fundamental package for scientific computing with arrays
- **Pandas 2.2.3**: Data manipulation and analysis library
- **scikit-learn 1.6.1**: Machine learning library for preprocessing, model evaluation, and metrics

#### Machine Learning

- **XGBoost 3.0.4**: Gradient boosting framework for model training
- **SHAP 0.48.0**: SHapley Additive exPlanations for model interpretability
- **scipy 1.15.2**: Scientific computing library
- **joblib 1.4.2**: Utilities for saving and loading Python objects (model serialization)

#### Geospatial Data Processing

- **Rasterio 1.4.3**: Geospatial raster I/O for reading WorldClim GeoTIFF files
- **Affine 2.4.0**: Spatial coordinate transformation library
- **GDAL 3.9.3**: Geospatial Data Abstraction Library

#### Visualization

- **Matplotlib 3.10.1**: Plotting library for creating figures and visualizations
- **Seaborn 0.13.2**: Statistical data visualization
- **Folium 0.20.0**: Interactive map generation and visualization
- **Plotly 6.0.1**: Interactive graphing library

#### Utilities and Tools

- **python-dotenv 1.1.0**: Environment variable management
- **python-dateutil 2.9.0**: Date parsing utilities
- **requests 2.31.0**: HTTP library for synchronous requests
- **urllib3 2.5.0**: HTTP client library
- **certifi 2025.8.3**: SSL certificate bundle
- **pytz 2024.1**: Timezone definitions
- **tqdm 4.67.1**: Progress bars for long-running operations
- **click 8.1.8**: Command-line interface creation toolkit

#### Additional Dependencies

- **typing_extensions 4.14.1**: Backported type hints
- **annotated-types 0.7.0**: Runtime type annotations
- **Jinja2 3.1.6**: Templating engine
- **MarkupSafe 3.0.2**: Safe string implementation
- **attrs 25.3.0**: Python classes without boilerplate
- **idna 3.7**: Internationalized Domain Names support
- **sniffio 1.3.1**: Sniff which async library is being used
- **charset-normalizer 3.3.2**: Character encoding detection
- **greenlet 3.0.1**: Lightweight in-process pseudo-threads
- **packaging 24.1**: Utilities for Python packages
- **threadpoolctl 3.5.0**: Control thread pool size
- **pillow 10.4.0**: Python Imaging Library
- **contourpy 1.3.1**: Contour algorithm implementation
- **cycler 0.12.1**: Composable style cycles
- **fonttools 4.56.0**: Font file processing
- **kiwisolver 1.4.8**: Constraint solving library
- **pyparsing 3.2.1**: General parsing module
- **colorama 0.4.6**: Cross-platform colored terminal text
- **cloudpickle 3.1.1**: Extended pickling support
- **llvmlite 0.44.0**: LLVM Python bindings
- **numba 0.61.2**: JIT compiler for numerical functions
- **slicer 0.0.8**: Sliceable sequences

### Development and Deployment Tools

- **Docker**: Containerization for deployment
- **Sphinx**: Documentation generation
- **Render**: Cloud deployment platform

### Data Sources

- **GBIF (Global Biodiversity Information Facility)**: Source for species occurrence records
- **WorldClim v2.1**: Climate data (19 bioclimatic variables at 10m resolution)
- **SRTM (Shuttle Radar Topography Mission)**: Elevation data via Open-Topo-Data API

## System Architecture

### Component Structure

The system is organized into four main architectural layers:

#### 1. API Layer (`app/routers/`)

- **status.py**: Health monitoring and system status checks
- **environmental.py**: Environmental data extraction endpoints (climate variables)
- **elevation.py**: Elevation data extraction endpoints
- **datasets.py**: ML-ready dataset generation endpoints
- **predictions.py**: Model training and prediction endpoints

#### 2. Service Layer (`app/services/`)

- **gbif_fetcher.py**: Fetches species occurrence data from GBIF API
- **worldclim_extractor.py**: Extracts climate data from WorldClim GeoTIFF files
- **elevation_extractor.py**: Extracts elevation data from SRTM via Open-Topo-Data API
- **generate_ml_ready_datasets.py**: Orchestrates the full data pipeline to create ML-ready CSV files

#### 3. Utilities Layer (`app/utils/`)

- **metrics_utils.py**: Standardized metrics calculation for model evaluation

#### 4. Model Layer (`models/xgboost/`)

- **train_model.py**: XGBoost model training script
- **train_model_api.py**: API-integrated model training
- **generate_heatmap_api.py**: Risk heatmap generation

## Detailed Functionality

### Data Collection Workflow

#### Step 1: GBIF Data Fetching (`app/services/gbif_fetcher.py`)

The GBIF Fetcher service implements asynchronous data retrieval from the Global Biodiversity Information Facility API:

**Core Functionality:**

- Fetches species occurrence records for Pyracantha angustifolia globally
- Applies quality filters to ensure data reliability (coordinate accuracy, date validation)
- Supports pagination for large datasets
- Implements rate limiting and retry logic to handle API limits
- Provides progress tracking for long-running downloads

**Key Features:**

- Async context manager for proper resource management
- Configurable quality filters (coordinate uncertainty, geospatial issues)
- Species key lookup using scientific names
- Batch processing with configurable batch sizes
- Error handling with exponential backoff for rate limits

**Data Retrieved:**

- Latitude and longitude coordinates
- Event date (year, month, day)
- Coordinate uncertainty information
- Country of origin

#### Step 2: Environmental Data Enrichment

##### Climate Data Extraction (`app/services/worldclim_extractor.py`)

Extracts bioclimatic variables from WorldClim v2.1 raster data:

**Supported Variables:**

- bio1: Annual Mean Temperature
- bio4: Temperature Seasonality
- bio5: Max Temperature of Warmest Month
- bio6: Min Temperature of Coldest Month
- bio12: Annual Precipitation
- bio13: Precipitation of Wettest Month
- bio14: Precipitation of Driest Month
- bio15: Precipitation Seasonality

**Technical Implementation:**

- Reads GeoTIFF raster files using Rasterio
- Performs point-in-raster sampling for each coordinate
- Applies scaling transformations (temperature values converted from °C × 10 to actual °C)
- Implements caching to avoid redundant file operations
- Handles nodata values and edge cases

##### Elevation Data Extraction (`app/services/elevation_extractor.py`)

Retrieves elevation data from SRTM 30m resolution via Open-Topo-Data API:

**Technical Implementation:**

- Asynchronous HTTP requests to Open-Topo-Data API
- Batch processing support for multiple coordinates
- Rate limiting with exponential backoff
- Caching at 100m precision to reduce API calls
- Fallback mechanisms for API failures
- Coordinate validation and error handling

#### Step 3: Temporal Feature Engineering

The system automatically computes temporal features from event dates:

**Generated Features:**

- month: Integer month (1-12)
- day_of_year: Day of year (1-365)
- sin_month: Sine transformation of month for cyclical encoding
- cos_month: Cosine transformation of month for cyclical encoding

**Purpose:**
Cyclical encoding (sin/cos) allows models to understand seasonal patterns as continuous cyclic features rather than discrete categories.

### Dataset Generation Workflow (`app/services/generate_ml_ready_datasets.py`)

This module orchestrates the complete pipeline to produce machine learning-ready datasets:

**Pipeline Steps:**

1. Fetch global occurrence records (worldwide Pyracantha data)
2. Fetch local occurrence records (South African subset)
3. Extract climate data for all coordinates
4. Extract elevation data for all coordinates
5. Compute temporal features
6. Combine all features into standardized format
7. Write CSV files to data directory

**Output Files:**

- `data/global_training_ml_ready.csv`: Global training dataset
- `data/local_validation_ml_ready.csv`: South African validation dataset

**Feature Set (15 features total):**

- Spatial: latitude, longitude
- Topographic: elevation
- Climate: bio1, bio4, bio5, bio6, bio12, bio13, bio14, bio15
- Temporal: month, day_of_year, sin_month, cos_month

**Quality Assurance:**

- Drops records with missing critical features
- Applies data type conversions
- Rounds floating-point values to 6 decimal places for consistency
- Validates data completeness before writing

### Model Training (`models/xgboost/train_model.py`)

The model training script implements an XGBoost classifier for binary classification:

**Training Process:**

1. Loads global training and local validation datasets
2. Creates background points if only presence data exists (for binary classification)
3. Prepares features (13 environmental variables)
4. Performs hyperparameter tuning using GridSearchCV
5. Trains final model with optimal parameters
6. Evaluates on local validation data
7. Calculates optimal classification threshold
8. Generates visualizations (ROC curve, feature importance)
9. Saves model artifacts and metrics

**Model Configuration:**

- Objective: binary:logistic (logistic regression for binary classification)
- Hyperparameter tuning: max_depth, learning_rate, n_estimators, subsample, colsample_bytree
- Evaluation metric: ROC-AUC
- Cross-validation: 3-fold CV during hyperparameter search

**Artifacts Generated:**

- `model.pkl`: Trained XGBoost model
- `optimal_threshold.pkl`: Optimal classification threshold
- `roc_curve.png`: ROC curve visualization
- `feature_importance.png`: Feature importance plot

### API Endpoints

The FastAPI application provides RESTful endpoints for various operations:

**1. Health Check (`GET /api/v1/status/health`)**

- Verifies API responsiveness
- Returns system health status

**2. Environmental Data Extraction (`POST /api/v1/environmental/extract-batch`)**

- Accepts batch of coordinates
- Returns climate variables for each coordinate
- Configurable variable selection

**3. Elevation Data Extraction (`POST /api/v1/elevation/extract-batch`)**

- Accepts batch of coordinates
- Returns elevation data via external API

**4. Dataset Generation (`POST /api/v1/datasets/generate-ml-ready-files`)**

- Orchestrates full data pipeline
- Generates ML-ready CSV files
- Configurable limits and batch sizes

**5. Model Training (`POST /api/v1/predictions/train-xgboost-model`)**

- Triggers model training
- Returns training metrics
- Saves model artifacts

**6. Heatmap Generation (`POST /api/v1/predictions/generate-xgboost-heatmap`)**

- Generates risk prediction heatmaps
- Uses trained model with real-time environmental data
- Returns interactive HTML maps with Folium

## Data Flow

1. **External Data Sources**

   - GBIF API → Species occurrence records
   - WorldClim GeoTIFF files → Climate variables
   - Open-Topo-Data API → Elevation data

2. **Data Processing**

   - Fetch occurrences (GBIF Fetcher)
   - Extract environmental variables (WorldClim Extractor, Elevation Extractor)
   - Engineer temporal features
   - Validate and clean data

3. **Dataset Generation**

   - Combine all features
   - Write standardized CSV files
   - Quality validation

4. **Model Training**

   - Load CSV datasets
   - Train XGBoost model
   - Evaluate performance
   - Generate visualizations

5. **Prediction and Visualization**
   - Load trained model
   - Extract environmental data for prediction grid
   - Generate risk scores
   - Create interactive heatmaps

## Transfer Learning Implementation

The system implements transfer learning by:

1. Training on global dataset (diverse environmental conditions worldwide)
2. Validating on local dataset (South African occurrences)
3. Applying globally learned patterns to local regions
4. Enabling adaptation to new geographic contexts without retraining

This approach leverages the richness of global occurrence data while maintaining local relevance for specific geographic applications.

## Deployment Architecture

### Docker Containerization

- Uses Python 3.12 slim base image
- Installs system dependencies (GDAL, build tools)
- Copies requirements and application code
- Exposes port 8000 for API access
- Runs with Uvicorn ASGI server

### Cloud Deployment (Render)

- Connects to GitHub repository
- Builds Docker container automatically
- Handles HTTPS and load balancing
- Provides monitoring and logging

## File Organization

```
PRJ381-data-preprocessing/
├── app/                          # Main application code
│   ├── routers/                  # API endpoint definitions
│   │   ├── datasets.py          # Dataset generation endpoints
│   │   ├── elevation.py         # Elevation data endpoints
│   │   ├── environmental.py     # Climate data endpoints
│   │   ├── predictions.py       # Model training/prediction endpoints
│   │   └── status.py            # Health check endpoints
│   ├── services/                 # Business logic layer
│   │   ├── gbif_fetcher.py      # GBIF data retrieval
│   │   ├── worldclim_extractor.py # Climate data extraction
│   │   ├── elevation_extractor.py # Elevation data extraction
│   │   └── generate_ml_ready_datasets.py # Complete pipeline
│   ├── utils/                    # Utility functions
│   │   └── metrics_utils.py     # Model evaluation metrics
│   └── main.py                   # FastAPI application
├── data/                         # Data storage
│   ├── global_training_ml_ready.csv    # Global training dataset
│   ├── local_validation_ml_ready.csv   # Local validation dataset
│   └── worldclim/               # WorldClim raster files
├── models/                       # Model artifacts
│   └── xgboost/                  # XGBoost model files
│       ├── model.pkl            # Trained model
│       ├── train_model.py       # Training script
│       ├── generate_heatmap_api.py # Heatmap generation
│       └── artifacts/            # Model visualizations
├── docs/                         # Documentation
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Key Technical Decisions

### Why FastAPI?

- High performance async/await support
- Automatic OpenAPI/Swagger documentation
- Native support for data validation with Pydantic
- Modern Python type hints throughout
- Built-in support for async HTTP clients

### Why XGBoost?

- Superior performance on structured data compared to Random Forest
- Built-in regularization to prevent overfitting
- Better handling of imbalanced datasets
- Efficient processing of large datasets
- Gradient boosting provides better generalization

### Why Asynchronous Processing?

- GBIF and elevation APIs require many HTTP requests
- Batch processing of coordinates benefits from parallelization
- Async/await pattern prevents blocking during I/O operations
- Significantly improves performance for large datasets

### Why Transfer Learning?

- Global training data provides diverse environmental representation
- Local validation tests model generalization
- Enables adaptation to new regions with minimal data
- Leverages global knowledge for local applications

## Quality Assurance

### Data Quality Measures

- Coordinate uncertainty filtering (max 10km uncertainty)
- Geospatial issue detection
- Missing data handling and validation
- Temporal data validation
- Feature completeness verification

### Model Evaluation

- ROC-AUC score for binary classification
- Precision, recall, F1 score
- Optimal threshold determination
- Feature importance analysis
- Cross-validation during training

## External Dependencies

### Data APIs

- **GBIF Occurrence API**: Species occurrence records
- **Open-Topo-Data API**: SRTM elevation data

### Required Files

- **WorldClim GeoTIFF files**: 19 bioclimatic variable files at 10m resolution (manually downloaded and stored in data/worldclim/)

## Performance Considerations

- Batch processing for environmental data extraction (configurable batch sizes)
- In-memory caching to reduce redundant computations
- Progress tracking with tqdm for long-running operations
- Rate limiting and exponential backoff for API requests
- Async operations for parallel data processing

## Security Considerations

- Input validation on all API endpoints
- Coordinate boundary validation
- Error handling without exposing internal details
- Safe file path construction
- Environment variable management for sensitive data

## Conclusion

This project implements a complete data preprocessing and machine learning pipeline for species distribution modeling, featuring asynchronous data processing, comprehensive environmental enrichment, and transfer learning capabilities. The system is designed for scalability, maintainability, and production deployment with Docker containerization.
