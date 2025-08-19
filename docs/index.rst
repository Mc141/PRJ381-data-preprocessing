PRJ381 Data Preprocessing API Documentation
==========================================

Welcome to the documentation for the PRJ381 Data Preprocessing API, a comprehensive FastAPI-based service for collecting and enriching invasive plant observation data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   api_reference
   user_guide
   architecture
   testing

Overview
========

This microservice provides:

* **iNaturalist API Integration**: Asynchronous fetching of species observation data
* **NASA POWER API Integration**: Environmental data enrichment with weather variables
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

4. **Access Documentation**:
   
   * Swagger UI: http://localhost:8000/docs
   * ReDoc: http://localhost:8000/redoc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

