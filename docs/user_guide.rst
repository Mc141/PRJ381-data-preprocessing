User Guide
==========

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.11 or higher
* MongoDB 4.4 or higher
* Internet connection for API access

Setup Steps
~~~~~~~~~~~

1. **Clone the Repository**::

    git clone https://github.com/Mc141/PRJ381-data-preprocessing.git
    cd PRJ381-data-preprocessing

2. **Install Dependencies**::

    pip install -r requirements.txt

3. **Start MongoDB**::

    # Using MongoDB service
    sudo systemctl start mongod
    
    # Or directly
    mongod --dbpath /path/to/your/db

4. **Run the Application**::

    uvicorn app.main:app --reload --port 8000

5. **Access Documentation**:
   
   * Swagger UI: http://localhost:8000/docs
   * ReDoc: http://localhost:8000/redoc

Basic Usage
-----------

Health Check
~~~~~~~~~~~~

Check if the service is running::

    GET /api/v1/status/health
    GET /api/v1/status/service_info

Fetching Observations
~~~~~~~~~~~~~~~~~~~~~

Get iNaturalist observations for a date range::

    GET /api/v1/observations/from?year=2024&month=8&day=1&store_in_db=true

Retrieve stored observations::

    GET /api/v1/observations/db

Fetching Weather Data
~~~~~~~~~~~~~~~~~~~~~

Get NASA POWER weather data::

    GET /api/v1/weather?latitude=-33.9249&longitude=18.4073&start_year=2024&start_month=1&start_day=1&end_year=2024&end_month=12&end_day=31&store_in_db=true

Creating Datasets
~~~~~~~~~~~~~~~~~

Merge observations with weather data::

    GET /api/v1/datasets/merge?start_year=2024&start_month=1&start_day=1&end_year=2024&end_month=12&end_day=31&years_back=5

Export merged dataset::

    GET /api/v1/datasets/export

Advanced Usage
--------------

Batch Processing
~~~~~~~~~~~~~~~~

For large datasets, use the async processing capabilities::

    # Process multiple years of data
    GET /api/v1/datasets/merge?start_year=2020&start_month=1&start_day=1&end_year=2024&end_month=12&end_day=31&years_back=10

Data Refresh
~~~~~~~~~~~~

Update weather data for existing observations::

    POST /api/v1/datasets/refresh-weather

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The application supports the following environment variables:

* ``MONGODB_URL``: MongoDB connection string (default: mongodb://localhost:27017/invasive_db)
* ``LOG_LEVEL``: Logging level (default: INFO)
* ``API_TIMEOUT``: API request timeout in seconds (default: 30)

Database Configuration
~~~~~~~~~~~~~~~~~~~~~~

MongoDB collections used:

* ``inat_observations``: Species observation data
* ``weather_data``: Daily weather time series
* ``weather_features``: Computed weather features

Error Handling
--------------

The API provides comprehensive error handling:

* **400 Bad Request**: Invalid parameters or date ranges
* **404 Not Found**: No data found for specified criteria
* **500 Internal Server Error**: Database or API communication errors

Common Issues
~~~~~~~~~~~~~

**MongoDB Connection Issues**::

    # Check if MongoDB is running
    sudo systemctl status mongod
    
    # Check connection string
    mongo mongodb://localhost:27017/invasive_db

**API Timeout Issues**::

    # Reduce date range for large queries
    # Use smaller years_back values
    # Check internet connection

**Memory Issues**::

    # Process smaller date ranges
    # Increase system memory
    # Use pagination for large datasets

Performance Tips
----------------

* Use concurrent processing for multiple observations
* Limit date ranges for initial testing
* Monitor MongoDB storage usage
* Use appropriate years_back values (1-10 years)
* Export data regularly to prevent large accumulations
