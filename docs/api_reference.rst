API Reference
=============

This section provides comprehensive documentation for all API modules, automatically generated from the source code docstrings.

Interactive API Documentation
-----------------------------

For interactive testing and exploration of the API endpoints, use the automatically generated documentation:

* **Swagger UI**: :swagger:`/ <docs>` - Interactive API documentation with request/response examples
* **ReDoc**: :redoc:`/ <redoc>` - Clean, responsive API documentation

.. note::
   The interactive documentation is automatically generated from the FastAPI application and is always up-to-date with the current API implementation.

Main Application
----------------

.. automodule:: app.main
   :members:
   :undoc-members:
   :show-inheritance:

Routers
-------

Status Router
~~~~~~~~~~~~~

System health monitoring and pipeline validation.

.. automodule:: app.routers.status
   :members:
   :undoc-members:
   :show-inheritance:

GBIF Router
~~~~~~~~~~~

Global species occurrence data collection from GBIF database.

.. automodule:: app.routers.gbif
   :members:
   :undoc-members:
   :show-inheritance:

WorldClim Router
~~~~~~~~~~~~~~~~

Real climate data download and extraction from WorldClim v2.1.

.. automodule:: app.routers.worldclim
   :members:
   :undoc-members:
   :show-inheritance:

Datasets Router
~~~~~~~~~~~~~~~

ML dataset creation, enrichment, and export functionality.

.. automodule:: app.routers.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Predictions Router
~~~~~~~~~~~~~~~~~~

Invasion risk prediction and mapping visualization.

.. automodule:: app.routers.predictions
   :members:
   :undoc-members:
   :show-inheritance:

Weather Router
~~~~~~~~~~~~~~

NASA POWER API integration for meteorological data.

.. automodule:: app.routers.weather
   :members:
   :undoc-members:
   :show-inheritance:

Observations Router (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy iNaturalist integration (deprecated - use GBIF instead).

.. automodule:: app.routers.observations
   :members:
   :undoc-members:
   :show-inheritance:

Weather Router
~~~~~~~~~~~~~~

.. automodule:: app.routers.weather
   :members:
   :undoc-members:
   :show-inheritance:

Datasets Router
~~~~~~~~~~~~~~~

.. automodule:: app.routers.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Services
--------

Database Service
~~~~~~~~~~~~~~~~

MongoDB connection and data management.

.. automodule:: app.services.database
   :members:
   :undoc-members:
   :show-inheritance:

GBIF Fetcher
~~~~~~~~~~~~

GBIF API integration service for species occurrence data.

.. automodule:: app.services.gbif_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

WorldClim Extractor Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified service for WorldClim data operations and environmental data extraction.

.. automodule:: app.services.worldclim_extractor
   :members:
   :undoc-members:
   :show-inheritance:

NASA POWER Fetcher
~~~~~~~~~~~~~~~~~~

NASA POWER API integration for meteorological data.

.. automodule:: app.services.nasa_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

iNaturalist Fetcher (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy iNaturalist API integration.

.. automodule:: app.services.inat_fetcher
   :members:
   :undoc-members:
   :show-inheritance:
