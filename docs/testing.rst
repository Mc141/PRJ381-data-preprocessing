Testing
=======

The PRJ381 Data Preprocessing API includes comprehensive testing coverage with both automated unit tests and manual testing procedures.

Automated Testing
-----------------

Test Framework
~~~~~~~~~~~~~~

The project uses **pytest** with async support for comprehensive test coverage:

* **Total Tests**: 131 test cases
* **Coverage**: All major functionality covered
* **Test Types**: Unit tests, integration tests, async tests
* **Framework**: pytest with pytest-asyncio

Test Structure
~~~~~~~~~~~~~~

.. code-block:: text

    tests/
    ├── __init__.py
    ├── test_datasets.py          # Dataset router and fusion tests
    ├── test_inat_fetcher.py      # iNaturalist API integration tests
    ├── test_nasa_fetcher.py      # NASA POWER API integration tests
    ├── test_observations.py      # Observations router tests
    └── test_weather.py           # Weather router tests

Running Tests
~~~~~~~~~~~~~

**Run All Tests**::

    pytest

**Run with Coverage**::

    pytest --cov=app --cov-report=html

**Run Specific Test File**::

    pytest tests/test_datasets.py

**Run Async Tests Only**::

    pytest -k "async"

**Verbose Output**::

    pytest -v

Test Categories
~~~~~~~~~~~~~~~

**Unit Tests**
    - Individual function testing
    - Mock external API calls
    - Input validation testing
    - Error handling verification

**Integration Tests**
    - Database operations
    - API endpoint functionality
    - Service layer integration
    - Data pipeline testing

**Async Tests**
    - Concurrent processing verification
    - Performance testing
    - Race condition detection
    - Resource cleanup validation

Key Test Cases
~~~~~~~~~~~~~~

**Data Validation Tests**

.. code-block:: python

    def test_observation_validation():
        """Test observation data validation"""
        # Valid observation
        valid_obs = {
            "id": "test123",
            "latitude": -33.9249,
            "longitude": 18.4073,
            "time_observed_at": "2024-08-01T10:30:00Z"
        }
        
        # Invalid observation (missing time)
        invalid_obs = {
            "id": "test456",
            "latitude": -33.9249,
            "longitude": 18.4073,
            "time_observed_at": None
        }

**API Endpoint Tests**

.. code-block:: python

    @pytest.mark.asyncio
    async def test_merge_datasets():
        """Test dataset merging functionality"""
        response = await client.get(
            "/api/v1/datasets/merge",
            params={
                "start_year": 2024,
                "start_month": 1,
                "start_day": 1,
                "end_year": 2024,
                "end_month": 1,
                "end_day": 31,
                "years_back": 2
            }
        )
        assert response.status_code == 200

**Error Handling Tests**

.. code-block:: python

    def test_none_time_handling():
        """Test handling of None time_observed_at values"""
        obs_with_none = {
            "time_observed_at": None,
            "latitude": 25.0,
            "longitude": -80.0
        }
        # Should return None and not crash
        result = process_observation(obs_with_none)
        assert result is None

Manual Testing
--------------

Interactive API Testing
~~~~~~~~~~~~~~~~~~~~~~~

The application provides **Swagger UI** for manual testing:

1. **Start the Server**::

    uvicorn app.main:app --reload --port 8000

2. **Access Swagger UI**: http://localhost:8000/docs

3. **Access ReDoc**: http://localhost:8000/redoc

Test Plan Structure
~~~~~~~~~~~~~~~~~~~

The manual testing is organized into phases as documented in ``testplan.md``:

**Phase 1: System Health**
    - Service health checks
    - Database connectivity
    - API responsiveness

**Phase 2: Observations**
    - Fetch observations from iNaturalist
    - Store and retrieve from database
    - Data validation and formatting

**Phase 3: Weather Data**
    - Fetch weather data from NASA POWER
    - Multi-location processing
    - Historical data retrieval

**Phase 4: Dataset Operations**
    - Data fusion and merging
    - Feature engineering
    - Export functionality

Test Scenarios
~~~~~~~~~~~~~~

**Basic Functionality**

.. code-block:: http

    # Health check
    GET /api/v1/status/health
    
    # Service information
    GET /api/v1/status/service_info
    
    # Fetch observations
    GET /api/v1/observations/from?year=2024&month=8&day=1

**Data Integration**

.. code-block:: http

    # Merge small dataset
    GET /api/v1/datasets/merge?start_year=2024&start_month=8&start_day=1&end_year=2024&end_month=8&end_day=2&years_back=1
    
    # Export dataset
    GET /api/v1/datasets/export

**Error Handling**

.. code-block:: http

    # Invalid date range
    GET /api/v1/datasets/merge?start_year=2025&start_month=1&start_day=1&end_year=2024&end_month=1&end_day=1
    
    # Out of range parameters
    GET /api/v1/datasets/merge?years_back=15

Performance Testing
-------------------

Load Testing
~~~~~~~~~~~~

**Concurrent Requests**::

    # Using Apache Bench
    ab -n 100 -c 10 http://localhost:8000/api/v1/status/health
    
    # Using curl with xargs for parallel requests
    echo "http://localhost:8000/api/v1/observations/from?year=2024&month=8&day=1" | xargs -n 1 -P 10 curl

**Large Dataset Testing**::

    # Test with multiple years
    GET /api/v1/datasets/merge?start_year=2020&start_month=1&start_day=1&end_year=2024&end_month=12&end_day=31&years_back=5

Memory and Resource Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Usage Monitoring**::

    # Monitor during large requests
    ps aux | grep python
    top -p $(pgrep -f uvicorn)

**Database Performance**::

    # MongoDB performance stats
    db.stats()
    db.inat_observations.getIndexes()

Test Data Management
--------------------

Test Database Setup
~~~~~~~~~~~~~~~~~~~

**Separate Test Database**::

    # Use different database for testing
    export MONGODB_URL="mongodb://localhost:27017/test_invasive_db"

**Data Cleanup**::

    # Clean test data after tests
    pytest --setup-clean

Mock Data Generation
~~~~~~~~~~~~~~~~~~~~

**Sample Observations**:

.. code-block:: python

    sample_observations = [
        {
            "id": "test_obs_1",
            "latitude": -33.9249,
            "longitude": 18.4073,
            "time_observed_at": "2024-08-01T10:30:00Z",
            "species": "Pyracantha angustifolia"
        }
    ]

**Sample Weather Data**:

.. code-block:: python

    sample_weather = {
        "date": "2024-08-01",
        "temperature": 18.5,
        "precipitation": 0.0,
        "humidity": 65.2
    }

Continuous Integration
----------------------

Automated Test Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

**GitHub Actions Workflow** (example):

.. code-block:: yaml

    name: Tests
    on: [push, pull_request]
    
    jobs:
      test:
        runs-on: ubuntu-latest
        services:
          mongodb:
            image: mongo:5.0
            ports:
              - 27017:27017
        
        steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: 3.11
        
        - name: Install dependencies
          run: pip install -r requirements.txt
        
        - name: Run tests
          run: pytest --cov=app

Quality Metrics
~~~~~~~~~~~~~~~

**Code Coverage**: Target 90%+ coverage
**Test Performance**: All tests complete within 60 seconds
**Integration Tests**: Pass against live APIs (with rate limiting)
**Documentation**: All public functions have docstrings and tests

Debugging Failed Tests
----------------------

Common Issues
~~~~~~~~~~~~~

**API Rate Limiting**::

    # Reduce test frequency
    # Use mock responses for external APIs
    # Implement retry logic in tests

**Database Connection Issues**::

    # Verify MongoDB is running
    # Check connection string
    # Ensure test database permissions

**Async Test Issues**::

    # Use proper pytest-asyncio decorators
    # Ensure proper cleanup of async resources
    # Handle race conditions in tests

**Memory Issues in Large Tests**::

    # Process smaller datasets in tests
    # Use streaming for large data operations
    # Implement proper cleanup

Best Practices
--------------

Test Organization
~~~~~~~~~~~~~~~~~

* **One assertion per test** when possible
* **Descriptive test names** that explain what is being tested
* **Setup and teardown** for consistent test environments
* **Parameterized tests** for testing multiple scenarios

Mock Usage
~~~~~~~~~~

* **Mock external APIs** to avoid rate limiting and network issues
* **Use realistic mock data** that matches actual API responses
* **Test both success and failure scenarios**
* **Verify mock calls** to ensure proper API usage

Async Testing
~~~~~~~~~~~~~

* **Use pytest-asyncio** for async test support
* **Proper resource cleanup** in async tests
* **Test concurrent operations** for race conditions
* **Monitor resource usage** during async tests
