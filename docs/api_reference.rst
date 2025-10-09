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

Environmental Router
~~~~~~~~~~~~~~~~~~~~

WorldClim and SRTM elevation data extraction for environmental enrichment.

.. automodule:: app.routers.environmental
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

Invasion risk prediction and mapping visualization using XGBoost.

.. automodule:: app.routers.predictions
   :members:
   :undoc-members:
   :show-inheritance:

Services
--------

GBIF Fetcher
~~~~~~~~~~~~

GBIF API integration service for species occurrence data.

.. automodule:: app.services.gbif_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

WorldClim Extractor
~~~~~~~~~~~~~~~~~~~

WorldClim v2.1 bioclimate variable extraction from GeoTIFF files.

.. automodule:: app.services.worldclim_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Elevation Extractor
~~~~~~~~~~~~~~~~~~~

SRTM 30m elevation data extraction via Open-Topo-Data API.

.. automodule:: app.services.elevation_extractor
   :members:
   :undoc-members:
   :show-inheritance:

ML Dataset Generator
~~~~~~~~~~~~~~~~~~~~

Generate ML-ready CSV datasets with environmental enrichment.

.. automodule:: app.services.generate_ml_ready_datasets
   :members:
   :undoc-members:
   :show-inheritance:
