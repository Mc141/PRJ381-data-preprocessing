Architecture
============

System Overview
---------------

The PRJ381 Data Preprocessing API follows a modular, microservice-oriented architecture designed for scalability, maintainability, and performance.

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   iNaturalist   │    │   NASA POWER    │    │    MongoDB      │
    │      API        │    │      API        │    │   Database      │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                        │                        │
            │                        │                        │
            ▼                        ▼                        ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                  FastAPI Application                            │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
    │  │   Status    │ │Observations │ │   Weather   │ │  Datasets   ││
    │  │   Router    │ │   Router    │ │   Router    │ │   Router    ││
    │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                     Services Layer                          ││
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐││
    │  │  │  Database   │ │iNat Fetcher │ │    NASA Fetcher         │││
    │  │  │   Service   │ │   Service   │ │      Service            │││
    │  │  └─────────────┘ └─────────────┘ └─────────────────────────┘││
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

**Database Service**
    - MongoDB connection management
    - Connection pooling and lifecycle management
    - Error handling and retry logic
    - Data persistence abstraction

**iNaturalist Fetcher Service**
    - Async HTTP client for iNaturalist API
    - Rate limiting and request optimization
    - Data validation and cleaning
    - Pagination handling

**NASA POWER Fetcher Service**
    - Weather data retrieval from NASA POWER API
    - Concurrent processing for multiple locations
    - Date range validation
    - Feature engineering pipeline

**Dataset Builder Service**
    - Data fusion and integration
    - Feature computation algorithms
    - Data quality validation
    - Export functionality

Data Layer
~~~~~~~~~~

**MongoDB Collections**

.. code-block:: javascript

    // inat_observations
    {
        "id": "observation_id",
        "latitude": -33.9249,
        "longitude": 18.4073,
        "time_observed_at": "2024-08-01T10:30:00Z",
        "species": "Pyracantha angustifolia",
        // ... other observation fields
    }

    // weather_data
    {
        "inat_id": "observation_id",
        "date": "2024-08-01",
        "temperature": 18.5,
        "precipitation": 0.0,
        "humidity": 65.2,
        // ... other weather variables
    }

    // weather_features
    {
        "inat_id": "observation_id",
        "obs_date": "2024-08-01",
        "years_back": 5,
        "features": {
            "temp_mean_30d": 17.8,
            "precip_sum_7d": 12.5,
            // ... computed features
        }
    }

Data Flow
---------

Observation Processing
~~~~~~~~~~~~~~~~~~~~~~

1. **API Request**: Client requests observations for date range
2. **iNaturalist Query**: Service fetches data from iNaturalist API
3. **Data Validation**: Validates coordinates, dates, and data quality
4. **Storage**: Stores validated observations in MongoDB
5. **Response**: Returns processed observations to client

Weather Enrichment
~~~~~~~~~~~~~~~~~~

1. **Location Extraction**: Extracts coordinates from observations
2. **Date Range Calculation**: Computes historical weather period
3. **NASA API Calls**: Fetches weather data concurrently
4. **Feature Engineering**: Computes temporal aggregations
5. **Storage**: Stores weather data and features
6. **Response**: Returns enriched dataset

Dataset Fusion
~~~~~~~~~~~~~~

1. **Data Retrieval**: Fetches observations and weather data
2. **Temporal Alignment**: Aligns weather data with observation dates
3. **Feature Computation**: Calculates rolling window features
4. **Quality Checks**: Validates data completeness and consistency
5. **Export**: Provides merged dataset for analysis

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
* **Database Clustering**: MongoDB replica sets and sharding
* **Load Balancing**: Multiple FastAPI instances behind load balancer
* **Caching Layer**: Redis for frequently accessed data

Vertical Scaling
~~~~~~~~~~~~~~~~

* **Async Processing**: Efficient CPU and memory utilization
* **Connection Pooling**: Optimized database connections
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

* **Database Access Control**: Authenticated MongoDB connections
* **Data Encryption**: Encrypted data transmission
* **Audit Logging**: Comprehensive access and operation logging
* **Backup Strategy**: Regular automated backups

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
        environment:
          - MONGODB_URL=mongodb://mongo:27017/invasive_db
        depends_on:
          - mongo
      
      mongo:
        image: mongo:5.0
        volumes:
          - mongo_data:/data/db
    
    volumes:
      mongo_data:

Monitoring and Observability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Health Checks**: Endpoint monitoring and service health
* **Metrics Collection**: Performance and usage metrics
* **Log Aggregation**: Centralized logging and analysis
* **Alerting**: Automated alerts for failures and performance issues
