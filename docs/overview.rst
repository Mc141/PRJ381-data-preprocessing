Overview
========

Project Purpose
---------------

The PRJ381 Species Distribution Modeling API is a production-ready FastAPI service designed for invasive species risk assessment and transfer learning. It provides a complete pipeline for creating ML-ready datasets from global biodiversity data enriched with real environmental variables (WorldClim climate + SRTM elevation).

Key Features
------------

Data Collection
~~~~~~~~~~~~~~~

* **GBIF Integration**: Fetches global species occurrence data from the GBIF database (1.4+ billion records)
* **Transfer Learning Focus**: Global training data for local validation (e.g., global → South Africa)
* **Asynchronous Processing**: Batch processing with progress tracking
* **Data Validation**: Coordinate validation and quality filtering
* **CSV Export**: Direct ML-ready dataset generation

Environmental Enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~

* **WorldClim v2.1 Integration**: Real bioclimate variable extraction from scientific-grade GeoTIFF rasters
* **SRTM Elevation Data**: NASA SRTM 30m elevation via Open-Topo-Data API
* **8 Core Climate Variables**: Temperature, precipitation, and seasonality (bio1, bio4-6, bio12-15)
* **Global Coverage**: Worldwide environmental data at 10 arcminute resolution
* **Missing Values**: Some grid cells may be unavailable; these are returned as None

Machine Learning Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

* **ML-Ready Datasets**: CSV exports optimized for XGBoost and other tree-based models
* **13 Base Features**: Latitude, longitude, elevation, 8 bioclimate variables, temporal encoding
* **Transfer Learning Support**: Global training datasets for local validation
* **Production Model**: XGBoost classifier (81% accuracy, 98% sensitivity)
* **Risk Mapping**: Real-time invasion probability heatmaps

Data Management
~~~~~~~~~~~~~~~

* **File-Based Storage**: Direct CSV export for ML training (no database required)
* **Batch Processing**: Efficient handling of large-scale environmental data extraction
* **API Rate Limiting**: Respectful Open-Topo-Data API usage (1-second delays)
* **Stateless Design**: Docker-ready, horizontally scalable architecture

API Architecture
~~~~~~~~~~~~~~~~

* **FastAPI Framework**: Modern, high-performance async web framework with automatic documentation
* **Workflow-Guided Interface**: Clear step-by-step API organization for optimal user experience
* **Modular Design**: Separated routers for different data types and processing stages
* **Comprehensive Documentation**: Swagger UI and ReDoc interfaces with detailed examples
* **Error Handling**: Robust error handling with detailed logging and user feedback

Data provenance
---------------

This project uses publicly available datasets for environmental and species data:

* WorldClim v2.1 (bioclimate rasters)
* SRTM elevation via Open-Topo-Data
* GBIF (occurrence records)

Use Cases
---------

* **Species Distribution Modeling**: Create training datasets for invasion risk assessment
* **Transfer Learning Applications**: Global training data for local model validation
* **Climate Change Research**: Environmental data integration for climate impact studies
* **Conservation Planning**: Data-driven approaches to species management
* **Ecological Research**: Multi-source data fusion for comprehensive ecological analysis
* **Machine Learning Development**: High-quality, feature-rich datasets for algorithm development

Technical Highlights
--------------------

* **Async/Await**: Full asynchronous processing for optimal performance
* **Concurrent API Calls**: 3-5x speedup through parallel processing
* **Professional Documentation**: Auto-generated API docs with Sphinx
* **Robust Error Handling**: Graceful handling of API failures and data inconsistencies
