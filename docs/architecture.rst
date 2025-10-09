Architecture
============

System Overview
---------------

The PRJ381 Species Distribution Modeling API follows a modular, stateless architecture designed for Docker deployment, horizontal scalability, and production readiness.

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │      GBIF       │    │   WorldClim     │    │  Open-Topo-Data │
    │    Database     │    │   GeoTIFFs      │    │   SRTM API      │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                        │                        │
            │                        │                        │
            ▼                        ▼                        ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                  FastAPI Application                            │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
    │  │   Status    │ │Environmental│ │  Datasets   │ │ Predictions ││
    │  │   Router    │ │   Router    │ │   Router    │ │   Router    ││
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                     Services Layer                          ││
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐││
    │  │  │    GBIF     │ │  WorldClim  │ │       Elevation         │││
    │  │  │   Fetcher   │ │  Extractor  │ │       Extractor         │││
    │  │  └─────────────┘ └─────────────┘ └─────────────────────────┘││
    │  └─────────────────────────────────────────────────────────────┘│
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                     Data Layer                              ││
    │  │  CSV Exports: global_training.csv, local_validation.csv     ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘

Components
----------

API Layer
~~~~~~~~~

**FastAPI Framework**
    - High-performance async web framework
    - Automatic API documentation generation
    - Type hints and validation
    - OpenAPI/Swagger integration

**Router Architecture**
    - Modular endpoint organization
    - Separation of concerns
    - Consistent error handling
    - RESTful design principles

Services Layer
~~~~~~~~~~~~~~

**GBIF Fetcher Service** (``gbif_fetcher.py``)
    - Async HTTP client for GBIF occurrence API
    - Quality filtering and coordinate validation
    - Pagination and batch processing
    - Species-specific and geographic filtering

**WorldClim Extractor Service** (``worldclim_extractor.py``)
    - Rasterio-based GeoTIFF reading
    - Batch coordinate sampling
    - 8 bioclimate variable extraction (bio1, bio4-6, bio12-15)
    - Missing data handling (returns None)

**Elevation Extractor Service** (``elevation_extractor.py``)
    - Open-Topo-Data API integration
    - SRTM 30m elevation data
    - Rate limiting (1-second delays)
    - Batch processing with error handling

**Dataset Generator Service** (``generate_ml_ready_datasets.py``)
    - ML-ready CSV generation
    - 13-feature standardized format
    - Transfer learning dataset creation
    - Progress tracking and logging

Data Layer
~~~~~~~~~~

**CSV Export Format**

ML-ready datasets with 13 standardized features:

.. code-block:: text

    latitude,longitude,elevation,bio1,bio4,bio5,bio6,bio12,bio13,bio14,bio15,month_sin,month_cos
    -33.9249,18.4073,245.0,16.8,574.2,25.3,9.1,515.0,79.0,15.0,48.2,0.5,0.866

**Feature Descriptions:**

* ``latitude``: Decimal degrees (-90 to 90)
* ``longitude``: Decimal degrees (-180 to 180)
* ``elevation``: Meters above sea level (SRTM 30m)
* ``bio1``: Annual mean temperature (°C × 10)
* ``bio4``: Temperature seasonality (std dev × 100)
* ``bio5``: Max temperature of warmest month (°C × 10)
* ``bio6``: Min temperature of coldest month (°C × 10)
* ``bio12``: Annual precipitation (mm)
* ``bio13``: Precipitation of wettest month (mm)
* ``bio14``: Precipitation of driest month (mm)
* ``bio15``: Precipitation seasonality (coefficient of variation)
* ``month_sin/cos``: Temporal encoding (0-1)

Data Flow
---------

Environmental Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Coordinate Input**: Client provides latitude/longitude pairs
2. **WorldClim Extraction**: Rasterio samples 8 bioclimate variables from GeoTIFFs
3. **Elevation Extraction**: Open-Topo-Data API fetches SRTM elevation (with rate limiting)
4. **Data Validation**: Checks for None/NaN values, validates ranges
5. **Response**: Returns enriched environmental data

ML-Ready Dataset Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **GBIF Data Loading**: Load occurrence records (global or local subset)
2. **Coordinate Extraction**: Extract unique latitude/longitude pairs
3. **Batch Processing**: Process coordinates in batches (default 100)
4. **Environmental Enrichment**: Add climate + elevation data
5. **Feature Engineering**: Add temporal encoding (month_sin, month_cos)
6. **CSV Export**: Write to ``data/global_training_ml_ready.csv`` and ``data/local_validation_ml_ready.csv``

Prediction Heatmap Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Grid Creation**: Generate lat/lon grid for target region
2. **Real-Time Enrichment**: Extract environmental data for each grid point
3. **Model Inference**: XGBoost predicts invasion probability
4. **Visualization**: Create Folium heatmap with stats panel
5. **Export**: Return HTML for display or download

Async Architecture
------------------

Concurrency Model
~~~~~~~~~~~~~~~~~

The application uses Python's ``asyncio`` for high-performance concurrent processing:

.. code-block:: python

    # Concurrent weather data fetching
    async def fetch_weather_for_observations(observations):
        tasks = [fetch_weather(obs) for obs in observations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

Benefits:
    - 3-5x performance improvement over synchronous processing
    - Efficient resource utilization
    - Non-blocking I/O operations
    - Scalable request handling

Error Handling Strategy
~~~~~~~~~~~~~~~~~~~~~~~

**Graceful Degradation**
    - Continue processing valid records when some fail
    - Comprehensive logging for debugging
    - Detailed error responses with context

**Retry Logic**
    - Automatic retry for transient failures
    - Exponential backoff for rate-limited APIs
    - Circuit breaker pattern for external services

**Data Validation**
    - Input validation at API boundaries
    - Schema validation for external API responses
    - Data quality checks throughout pipeline

Scalability Considerations
--------------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

* **Stateless Design**: No server-side session state
* **File-based Storage**: CSV exports for distributed processing
* **Load Balancing**: Multiple FastAPI instances behind load balancer
* **Caching Layer**: WorldClim extraction results cache

Vertical Scaling
~~~~~~~~~~~~~~~~

* **Async Processing**: Efficient CPU and memory utilization with asyncio
* **Batch Processing**: Large coordinate sets processed in chunks
* **Memory Management**: Streaming processing for large datasets
* **Resource Monitoring**: Performance metrics and alerting

Security Architecture
---------------------

API Security
~~~~~~~~~~~~

* **Input Validation**: Comprehensive parameter validation
* **Rate Limiting**: Protection against abuse
* **Error Sanitization**: No sensitive data in error responses
* **HTTPS Enforcement**: Secure communication protocols

Data Security
~~~~~~~~~~~~~

* **File System Access Control**: Protected WorldClim data directory
* **Data Encryption**: Encrypted data transmission via HTTPS
* **Audit Logging**: Comprehensive access and operation logging
* **Backup Strategy**: Version-controlled datasets and model files

Deployment Architecture
-----------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Local development setup
    uvicorn app.main:app --reload --port 8000

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # Docker Compose example
    version: '3.8'
    services:
      api:
        build: .
        ports:
          - "8000:8000"
        volumes:
          - ./data:/app/data
          - ./models:/app/models
        environment:
          - WORLDCLIM_PATH=/app/data/worldclim/
          - LOG_LEVEL=INFO

Monitoring and Observability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Health Checks**: Endpoint monitoring and service health
* **Metrics Collection**: Performance and usage metrics
* **Log Aggregation**: Centralized logging and analysis
* **Alerting**: Automated alerts for failures and performance issues
