PRJ381 Data Preprocessing API Documentation
==========================================

Welcome to the documentation for the PRJ381 Data Preprocessing API, a comprehensive FastAPI-based service for collecting and enriching invasive plant observation data.

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

