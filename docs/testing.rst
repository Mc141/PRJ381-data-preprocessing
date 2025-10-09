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
    ├── test_status.py            # Health check and system status tests
    ├── test_datasets.py          # ML dataset generation tests
    ├── test_predictions.py       # Model training and heatmap tests
    ├── test_observations.py      # GBIF data fetching tests (legacy)
    └── test_weather.py           # Environmental data extraction tests (legacy)

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
    - WorldClim file availability
    - XGBoost model status

**Phase 2: Environmental Data**
    - WorldClim climate variable extraction
    - SRTM elevation data fetching
    - Batch coordinate processing

**Phase 3: ML Dataset Generation**
    - Global training dataset creation
    - Local validation dataset creation
    - Feature engineering (13 features)

**Phase 4: Predictions**
    - Model training with XGBoost
    - Heatmap generation with real-time environmental data
    - Risk probability visualization

Test Scenarios
~~~~~~~~~~~~~~

**Basic Functionality**

.. code-block:: http

    # Health check
    GET /api/v1/status/health
    
    # Service information
    GET /api/v1/status/service_info
    
    # Extract environmental data
    POST /api/v1/environmental/extract-batch
    Body: {"coordinates": [{"latitude": -33.925, "longitude": 18.424}]}

**Dataset Generation**

.. code-block:: http

    # Generate ML-ready datasets
    POST /api/v1/datasets/generate-ml-ready-files?max_global=100&max_local=50&batch_size=20
    
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

**File System Performance**::

    # Check WorldClim file access
    ls -lh data/worldclim/*.tif
    # Check model file
    ls -lh models/xgboost/model.pkl

Test Data Management
--------------------

Test Dataset Setup
~~~~~~~~~~~~~~~~~~

**Sample Test Data**::

    # Create small test datasets for fast testing
    python -m app.services.generate_ml_ready_datasets --max-global 10 --max-local 5

**Data Cleanup**::

    # Remove test datasets after tests
    rm data/*_test.csv

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
        
        steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: 3.12
        
        - name: Cache WorldClim data
          uses: actions/cache@v2
          with:
            path: data/worldclim
            key: worldclim-v2.1
        
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

    # Open-Topo-Data API has rate limits
    # Use mock responses for tests
    # Implement 1-second delays for real requests

**WorldClim File Access**::

    # Verify GeoTIFF files exist in data/worldclim/
    # Check file permissions (read access required)
    # Ensure sufficient disk space for extraction

**Async Test Issues**::

    # Use proper pytest-asyncio decorators
    # Ensure proper cleanup of async resources
    # Handle race conditions in batch processing

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
