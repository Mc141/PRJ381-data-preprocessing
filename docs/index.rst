PRJ381 Data Preprocessing API Documentation
==========================================

Welcome to the documentation for the PRJ381 Data Preprocessing API, a comprehensive FastAPI-based service for collecting and enriching species observation data for machine learning and transfer learning applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   user_guide
   api_reference
   architecture
   testing

Documentation Types
===================

This project provides multiple types of documentation to serve different needs:

**For API Users**
   * **Interactive API Documentation**: :swagger:`Swagger UI </docs>` - Test endpoints directly in your browser
   * **Alternative API Documentation**: :redoc:`ReDoc </redoc>` - Clean, responsive API reference
   * **User Guide**: :doc:`user_guide` - Step-by-step usage examples

**For Developers**
   * **API Reference**: :doc:`api_reference` - Complete code documentation  
   * **Architecture Guide**: :doc:`architecture` - System design and components
   * **Testing Guide**: :doc:`testing` - Testing strategies and examples

.. tip::
   **Quick Start**: If you just want to try the API, head to the :swagger:`interactive documentation </docs>` and start making requests!

Overview
========

This microservice provides:

* **GBIF API Integration**: Global species occurrence data collection and management
* **WorldClim Integration**: Real climate data extraction from v2.1 bioclimate variables
* **NASA POWER API Integration**: Environmental data enrichment with weather variables
* **MongoDB Storage**: Persistent storage for observations, environmental data, and ML features
* **FastAPI REST API**: Modern, async REST endpoints with automatic documentation
* **ML Pipeline**: Complete data fusion pipeline for machine learning dataset creation
* **Transfer Learning Support**: Global training datasets for local model validation

The service is designed for creating high-quality training datasets for species distribution modeling and invasion risk assessment.

Data Integrity Policy
=====================

This API maintains strict data integrity standards:

* **Real Data Only**: No fake, dummy, or placeholder environmental values
* **Transparent Missing Data**: When data unavailable, returns None/NaN (never fake values)
* **Clear Data Sources**: All data sources clearly labeled and trackable
* **Scientific Standards**: Uses WorldClim v2.1 and GBIF data trusted by researchers worldwide

Quick Start
===========

1. **Install Dependencies**::

    pip install -r requirements.txt

2. **Start MongoDB**::

    mongod --dbpath /path/to/your/db

3. **Run the API**::

    uvicorn app.main:app --reload

4. **Build Documentation**::

    python build_docs.py --serve --open

API Workflow
============

Follow this sequence for optimal results:

1. **System Status**: `GET /api/v1/status/health` - Verify system health
2. **Species Data**: `GET /api/v1/gbif/occurrences?store_in_db=true` - Collect GBIF data
3. **Environmental Data**: `POST /api/v1/worldclim/ensure-data` - Download climate data
4. **Dataset Creation**: `GET /api/v1/datasets/merge-global` - Create enriched dataset
5. **ML Export**: `GET /api/v1/datasets/export-ml-ready` - Export for model training
6. **Predictions**: `GET /api/v1/predictions/*` - Generate invasion risk maps

.. note::
   When running locally, the API will be available at http://localhost:8000
   
   The documentation will be built and served automatically using the universal build script.
   
   For manual builds::
   
       # Build only
       python build_docs.py
       
       # Clean build and serve
       python build_docs.py --clean --serve
* **MongoDB Storage**: Persistent storage for observations, weather data, and computed features
* **FastAPI REST API**: Modern, async REST endpoints with automatic documentation
* **Data Pipeline**: Complete data fusion pipeline with error handling and validation

The service is designed to run as part of a larger ecological monitoring platform for predicting invasive plant spread patterns.

Quick Start
===========

1. **Install Dependencies**::

    pip install -r requirements.txt

2. **Start MongoDB**::

    mongod --dbpath /path/to/your/db

3. **Run the API**::

    uvicorn app.main:app --reload

4. **Build Documentation**::

    python build_docs.py --serve --open

.. note::
   When running locally, the API will be available at http://localhost:8000
   
   The documentation will be built and served automatically using the universal build script.
   
   For manual builds::
   
       # Build only
       python build_docs.py
       
       # Clean build and serve
       python build_docs.py --clean --serve

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

