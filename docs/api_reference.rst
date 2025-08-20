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

Configuration
-------------

.. automodule:: app.config
   :members:
   :undoc-members:
   :show-inheritance:

Routers
-------

Status Router
~~~~~~~~~~~~~

.. automodule:: app.routers.status
   :members:
   :undoc-members:
   :show-inheritance:

Observations Router
~~~~~~~~~~~~~~~~~~~

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

.. automodule:: app.services.database
   :members:
   :undoc-members:
   :show-inheritance:

iNaturalist Fetcher
~~~~~~~~~~~~~~~~~~~

.. automodule:: app.services.inat_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

NASA Fetcher
~~~~~~~~~~~~

.. automodule:: app.services.nasa_fetcher
   :members:
   :undoc-members:
   :show-inheritance:
