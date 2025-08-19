Overview
========

Project Purpose
---------------

The PRJ381 Data Preprocessing API is a standalone asynchronous data collection and enrichment service designed for invasive plant monitoring. It serves as a critical component in a larger ecological monitoring platform focused on predicting the spread patterns of invasive plant species.

Key Features
------------

Data Collection
~~~~~~~~~~~~~~~

* **iNaturalist Integration**: Fetches geospatial observation data for *Pyracantha angustifolia* within the Western Cape, South Africa
* **Asynchronous Processing**: High-performance concurrent data retrieval
* **Data Validation**: Ensures positional accuracy and data quality
* **Flexible Filtering**: Date range, location bounding box, and species-specific queries

Environmental Enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~

* **NASA POWER API Integration**: Retrieves comprehensive weather and climate data
* **Multi-year Historical Data**: Supports up to 10 years of historical weather context
* **Feature Engineering**: Computes temporal weather features using rolling window aggregations
* **Environmental Variables**: Temperature, precipitation, humidity, wind speed, solar radiation

Data Management
~~~~~~~~~~~~~~~

* **MongoDB Persistence**: Scalable document storage for observations and weather data
* **Data Deduplication**: Automatic handling of duplicate records
* **Incremental Updates**: Efficient refreshing of existing datasets
* **Export Capabilities**: CSV export functionality for downstream analysis

API Architecture
~~~~~~~~~~~~~~~~

* **FastAPI Framework**: Modern, high-performance async web framework
* **REST Endpoints**: Comprehensive API with automatic documentation
* **Modular Design**: Separated routers for different data types
* **Error Handling**: Robust error handling with detailed logging

Use Cases
---------

* **Machine Learning Pipeline**: Provides clean, feature-rich datasets for ML model training
* **Ecological Research**: Supports research into invasive species spread patterns
* **Environmental Monitoring**: Enables long-term monitoring of species-environment interactions
* **Data Integration**: Serves as a data fusion layer for multi-source ecological data

Technical Highlights
--------------------

* **Async/Await**: Full asynchronous processing for optimal performance
* **Concurrent API Calls**: 3-5x speedup through parallel processing
* **Comprehensive Testing**: 131+ test cases covering all functionality
* **Professional Documentation**: Auto-generated API docs with Sphinx
* **Robust Error Handling**: Graceful handling of API failures and data inconsistencies
