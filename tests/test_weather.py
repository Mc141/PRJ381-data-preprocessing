import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.routers.weather import router


@pytest.fixture
def test_client():
    """Create a test client for the weather router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing."""
    return {
        "location": {
            "latitude": -33.9,
            "longitude": 18.4,
            "elevation": 100.5
        },
        "data": [
            {
                "date": "2023-07-01",
                "T2M": 15.5,
                "PRECTOT": 0.0,
                "elevation": 100.5,
                "latitude": -33.9,
                "longitude": 18.4
            }
        ],
        "parameters": {
            "T2M": {"longname": "Temperature at 2 Meters", "units": "C"},
            "PRECTOT": {"longname": "Precipitation", "units": "mm/day"}
        }
    }


@pytest.fixture
def mock_database():
    """Mock database setup for testing."""
    mock_db = MagicMock()
    mock_collection = Mock()
    
    # Mock the chained methods for MongoDB operations
    mock_find_result = Mock()
    mock_sort_result = Mock()
    mock_limit_result = []  # Use an actual list for limit result
    
    # Setup the chain: find() -> sort() -> limit()
    mock_collection.find.return_value = mock_find_result
    mock_find_result.sort.return_value = mock_sort_result
    mock_sort_result.limit.return_value = mock_limit_result
    
    # For basic find().limit() pattern
    mock_find_result.limit.return_value = []
    
    # For other operations
    mock_collection.insert_many.return_value = Mock(inserted_ids=[])
    mock_collection.delete_many.return_value = Mock(deleted_count=0)
    mock_collection.count_documents.return_value = 0
    
    mock_db.__getitem__.return_value = mock_collection
    return mock_db, mock_collection


# Tests for /weather endpoint
@patch('app.routers.weather.PowerAPI')
def test_get_weather_data_success(mock_power_api_class, test_client, sample_weather_data):
    """Test successful weather data retrieval."""
    # Setup mocks - Note: The TestClient handles async endpoints automatically
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value=sample_weather_data)
    mock_power_api_class.return_value = mock_power_api
    
    response = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "location" in data
    assert "data" in data
    assert len(data["data"]) == 1
    
    # Verify PowerAPI was called correctly
    mock_power_api_class.assert_called_once()
    mock_power_api.get_weather.assert_called_once()


@patch('app.routers.weather.PowerAPI')
@patch('app.routers.weather.get_database')
def test_get_weather_data_with_db_storage(mock_get_database, mock_power_api_class, 
                                        test_client, sample_weather_data, mock_database):
    """Test weather data retrieval with database storage."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value=sample_weather_data)
    mock_power_api_class.return_value = mock_power_api
    
    response = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
        "&store_in_db=true"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    
    # Verify database storage
    mock_collection.insert_many.assert_called_once()


@patch('app.routers.weather.PowerAPI')
def test_get_weather_data_api_error(mock_power_api_class, test_client):
    """Test weather data retrieval when NASA API fails."""
    # Setup mock to raise an exception - Note: TestClient handles async endpoints
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(side_effect=Exception("NASA API error"))
    mock_power_api_class.return_value = mock_power_api
    
    response = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
    )
    
    assert response.status_code == 500
    assert "Failed to retrieve weather data" in response.json()["detail"]


def test_get_weather_data_missing_parameters(test_client):
    """Test weather data retrieval with missing required parameters."""
    # Missing latitude
    response = test_client.get(
        "/weather?longitude=18.4"
        "&start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
    )
    assert response.status_code == 422  # Validation error


def test_get_weather_data_invalid_dates(test_client):
    """Test weather data retrieval with invalid date parameters."""
    # Invalid month
    response = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=13&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
    )
    assert response.status_code == 400
    assert "Invalid date" in response.json()["detail"]


def test_get_weather_data_end_before_start(test_client):
    """Test weather data retrieval when end date is before start date."""
    response = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=7&start_day=5"
        "&end_year=2023&end_month=7&end_day=1"
    )
    assert response.status_code == 400
    assert "End date must be after start date" in response.json()["detail"]


# Tests for /weather/db endpoint
@patch('app.routers.weather.get_database')
def test_get_weather_from_db_success(mock_get_database, test_client, mock_database):
    """Test successful retrieval of weather data from database."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    
    sample_db_records = [
        {"_id": "obj1", "date": "2023-07-01", "T2M": 15.5},
        {"_id": "obj2", "date": "2023-07-02", "T2M": 16.0}
    ]
    mock_collection.find.return_value.limit.return_value = sample_db_records
    
    response = test_client.get("/weather/db")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["date"] == "2023-07-01"


@patch('app.routers.weather.get_database')
def test_get_weather_from_db_with_limit(mock_get_database, test_client, mock_database):
    """Test weather data retrieval with limit parameter."""
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_collection.find.return_value.limit.return_value = []
    
    response = test_client.get("/weather/db?limit=50")
    assert response.status_code == 200
    assert response.json() == []


@patch('app.routers.weather.get_database')
def test_get_weather_from_db_empty_result(mock_get_database, test_client, mock_database):
    """Test weather data retrieval when database is empty."""
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_collection.find.return_value.limit.return_value = []
    
    response = test_client.get("/weather/db")
    assert response.status_code == 200
    assert response.json() == []


# Tests for DELETE /weather/db endpoint
@patch('app.routers.weather.get_database')
def test_delete_weather_from_db_success(mock_get_database, test_client, mock_database):
    """Test successful deletion of weather data from database."""
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    
    mock_result = Mock()
    mock_result.deleted_count = 5
    mock_collection.delete_many.return_value = mock_result
    
    response = test_client.delete("/weather/db")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Deleted 5 weather records"


@patch('app.routers.weather.get_database')
def test_delete_weather_from_db_no_records(mock_get_database, test_client, mock_database):
    """Test deletion when no records exist."""
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    
    mock_result = Mock()
    mock_result.deleted_count = 0
    mock_collection.delete_many.return_value = mock_result
    
    response = test_client.delete("/weather/db")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Deleted 0 weather records"


# Tests for /weather/recent endpoint
@patch('app.routers.weather.get_database')
def test_get_recent_weather_success(mock_get_database, test_client, mock_database):
    """Test successful retrieval of recent weather data."""
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    
    sample_recent_data = [
        {"date": "2023-07-02", "T2M": 16.0},
        {"date": "2023-07-01", "T2M": 15.5}
    ]
    mock_collection.find.return_value.sort.return_value.limit.return_value = sample_recent_data
    
    response = test_client.get("/weather/recent?latitude=-33.9&longitude=18.4&days=7")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


# Integration and error handling tests
@patch('app.routers.weather.PowerAPI')
@patch('app.routers.weather.get_database')
def test_full_weather_workflow(mock_get_database, mock_power_api_class,
                             test_client, sample_weather_data, mock_database):
    """Test the complete weather data workflow."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value=sample_weather_data)
    mock_power_api_class.return_value = mock_power_api
    
    # 1. Fetch and store weather data
    response1 = test_client.get(
        "/weather?latitude=-33.9&longitude=18.4"
        "&start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=2"
        "&store_in_db=true"
    )
    assert response1.status_code == 200
    
    # 2. Retrieve stored data
    mock_collection.find.return_value.limit.return_value = sample_weather_data["data"]
    response2 = test_client.get("/weather/db")
    assert response2.status_code == 200
    
    # 3. Delete stored data
    mock_result = Mock()
    mock_result.deleted_count = 2
    mock_collection.delete_many.return_value = mock_result
    response3 = test_client.delete("/weather/db")
    assert response3.status_code == 200


@patch('app.routers.weather.get_database')
def test_database_error_handling(mock_get_database, test_client):
    """Test handling of database connection errors."""
    mock_get_database.side_effect = Exception("Database connection failed")
    
    response = test_client.get("/weather/db")
    assert response.status_code == 500
