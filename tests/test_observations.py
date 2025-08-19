import pytest
import datetime
import copy
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.routers.observations import router


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def test_client():
    """Create a test client for the observations router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_observations():
    """Sample observation data."""
    return [
        {
            "id": 1,
            "uuid": "test-uuid-1",
            "time_observed_at": "2025-07-01T12:00:00Z",
            "created_at": "2025-07-01T13:00:00Z",
            "latitude": -33.9,
            "longitude": 18.4,
            "positional_accuracy": 50,
            "place_guess": "Cape Town",
            "scientific_name": "Exampleus plantus",
            "common_name": "Example Plant",
            "quality_grade": "research",
            "image_url": "https://example.com/image.jpg",
            "user_id": 101
        },
        {
            "id": 2,
            "uuid": "test-uuid-2",
            "time_observed_at": "2025-07-02T14:00:00Z",
            "created_at": "2025-07-02T15:00:00Z",
            "latitude": -33.8,
            "longitude": 18.5,
            "positional_accuracy": 75,
            "place_guess": "Stellenbosch",
            "scientific_name": "Testus specimen",
            "common_name": "Test Specimen",
            "quality_grade": "research",
            "image_url": "https://example.com/image2.jpg",
            "user_id": 102
        }
    ]


@pytest.fixture
def mock_database():
    """Mock database and collections."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    return mock_db, mock_collection


# -----------------------------
# Tests for /observations endpoint
# -----------------------------

@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
def test_read_observations_success(mock_get_observations, mock_get_pages, test_client, sample_observations):
    """Test successful retrieval of observations."""
    # Setup mocks
    mock_get_pages.return_value = [{"results": []}]
    mock_get_observations.return_value = sample_observations
    
    response = test_client.get("/observations")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[1]["id"] == 2
    
    # Verify service calls
    mock_get_pages.assert_called_once()
    mock_get_observations.assert_called_once()


@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
@patch('app.routers.observations.get_database')
def test_read_observations_with_db_storage(mock_get_database, mock_get_observations, 
                                         mock_get_pages, test_client, sample_observations, 
                                         mock_database):
    """Test observation retrieval with database storage."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = [{"results": []}]
    mock_get_observations.return_value = sample_observations
    
    response = test_client.get("/observations?store_in_db=true")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Verify database storage
    mock_collection.insert_many.assert_called_once()
    inserted_data = mock_collection.insert_many.call_args[0][0]
    assert len(inserted_data) == 2


@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
def test_read_observations_empty_result(mock_get_observations, mock_get_pages, test_client):
    """Test observation retrieval with empty results."""
    # Setup mocks
    mock_get_pages.return_value = []
    mock_get_observations.return_value = []
    
    response = test_client.get("/observations")
    
    assert response.status_code == 200
    data = response.json()
    assert data == []


@patch('app.routers.observations.get_pages')
def test_read_observations_service_error(mock_get_pages, test_client):
    """Test observation retrieval when service raises an error."""
    # Setup mock to raise an exception
    mock_get_pages.side_effect = Exception("Service error")
    
    response = test_client.get("/observations")
    
    # FastAPI should return 500 for unhandled exceptions
    assert response.status_code == 500


# -----------------------------
# Tests for /observations/from endpoint
# -----------------------------

@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
def test_read_observations_from_date_success(mock_get_observations, mock_get_pages, 
                                           test_client, sample_observations):
    """Test successful retrieval of observations from a specific date."""
    # Setup mocks
    mock_get_pages.return_value = [{"results": []}]
    mock_get_observations.return_value = sample_observations
    
    response = test_client.get("/observations/from?year=2023&month=7&day=1")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Verify service was called with correct date
    mock_get_pages.assert_called_once()
    call_args = mock_get_pages.call_args
    # The start_date should be passed as a positional argument
    if call_args[0]:  # positional args exist
        start_date = call_args[0][0]  # first positional argument
    else:  # keyword arguments
        start_date = call_args[1].get('start_date')  # keyword argument
    assert start_date == datetime.datetime(2023, 7, 1)


@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
@patch('app.routers.observations.get_database')
def test_read_observations_from_date_with_db_storage(mock_get_database, mock_get_observations,
                                                   mock_get_pages, test_client, 
                                                   sample_observations, mock_database):
    """Test observation retrieval from date with database storage."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = [{"results": []}]
    mock_get_observations.return_value = sample_observations
    
    response = test_client.get("/observations/from?year=2023&month=6&day=15&store_in_db=true")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Verify database storage
    mock_collection.insert_many.assert_called_once()


def test_read_observations_from_date_invalid_date(test_client):
    """Test observation retrieval with invalid date parameters."""
    # Invalid month
    response = test_client.get("/observations/from?year=2023&month=13&day=1")
    assert response.status_code == 400
    assert "Invalid date" in response.json()["detail"]
    
    # Invalid day
    response = test_client.get("/observations/from?year=2023&month=2&day=30")
    assert response.status_code == 400
    assert "Invalid date" in response.json()["detail"]
    
    # Invalid year (negative)
    response = test_client.get("/observations/from?year=-1&month=1&day=1")
    assert response.status_code == 400


def test_read_observations_from_date_missing_parameters(test_client):
    """Test observation retrieval with missing required parameters."""
    # Missing year
    response = test_client.get("/observations/from?month=7&day=1")
    assert response.status_code == 422  # Unprocessable Entity
    
    # Missing month
    response = test_client.get("/observations/from?year=2023&day=1")
    assert response.status_code == 422
    
    # Missing day
    response = test_client.get("/observations/from?year=2023&month=7")
    assert response.status_code == 422


@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
def test_read_observations_from_date_edge_cases(mock_get_observations, mock_get_pages, test_client):
    """Test observation retrieval with edge case dates."""
    mock_get_pages.return_value = []
    mock_get_observations.return_value = []
    
    # Leap year date
    response = test_client.get("/observations/from?year=2020&month=2&day=29")
    assert response.status_code == 200
    
    # Last day of month
    response = test_client.get("/observations/from?year=2023&month=12&day=31")
    assert response.status_code == 200
    
    # First day of year
    response = test_client.get("/observations/from?year=2023&month=1&day=1")
    assert response.status_code == 200


# -----------------------------
# Tests for /observations/{observation_id} endpoint (if it exists)
# -----------------------------

def test_read_observations_by_id_endpoint_exists(test_client):
    """Check if the individual observation endpoint exists."""
    # First, let's check if this endpoint exists by looking at the routes
    routes = [route.path for route in test_client.app.routes]
    
    # If the endpoint doesn't exist in the current implementation,
    # this test documents the expected behavior for future implementation
    if "/observations/{observation_id}" not in routes:
        pytest.skip("Individual observation endpoint not yet implemented")


# -----------------------------
# Integration tests
# -----------------------------

@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
@patch('app.routers.observations.get_database')
def test_full_observation_workflow(mock_get_database, mock_get_observations, 
                                 mock_get_pages, test_client, sample_observations, 
                                 mock_database):
    """Test the complete observation retrieval and storage workflow."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = [{"results": sample_observations}]
    mock_get_observations.return_value = sample_observations
    
    # Test without storage
    response1 = test_client.get("/observations")
    assert response1.status_code == 200
    assert len(response1.json()) == 2
    
    # Verify no database call was made
    mock_get_database.assert_not_called()
    
    # Test with storage
    response2 = test_client.get("/observations?store_in_db=true")
    assert response2.status_code == 200
    assert len(response2.json()) == 2
    
    # Verify database storage
    mock_get_database.assert_called_once()
    mock_collection.insert_many.assert_called_once()
    
    # Test from date endpoint
    response3 = test_client.get("/observations/from?year=2023&month=7&day=1&store_in_db=true")
    assert response3.status_code == 200
    assert len(response3.json()) == 2


# -----------------------------
# Error handling tests
# -----------------------------

@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
@patch('app.routers.observations.get_database')
def test_database_error_handling(mock_get_database, mock_get_observations, 
                                mock_get_pages, test_client, sample_observations):
    """Test handling of database errors during storage."""
    # Setup mocks
    mock_get_pages.return_value = []
    mock_get_observations.return_value = sample_observations
    mock_get_database.side_effect = Exception("Database connection failed")
    
    # This should raise an exception since database storage fails
    response = test_client.get("/observations?store_in_db=true")
    assert response.status_code == 500


# -----------------------------
# Parameter validation tests
# -----------------------------

@pytest.mark.parametrize("store_in_db_value,expected_call", [
    ("true", True),
    ("false", False),
    ("1", True),
    ("0", False),
    ("", False),
])
@patch('app.routers.observations.get_pages')
@patch('app.routers.observations.get_observations')
@patch('app.routers.observations.get_database')
def test_store_in_db_parameter_parsing(mock_get_database, mock_get_observations, 
                                     mock_get_pages, test_client, sample_observations,
                                     mock_database, store_in_db_value, expected_call):
    """Test parsing of store_in_db parameter."""
    # Setup mocks
    mock_db, mock_collection = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = []
    mock_get_observations.return_value = sample_observations
    
    response = test_client.get(f"/observations?store_in_db={store_in_db_value}")
    assert response.status_code == 200
    
    if expected_call:
        mock_get_database.assert_called_once()
        mock_collection.insert_many.assert_called_once()
    else:
        mock_get_database.assert_not_called()


def test_query_parameter_types(test_client):
    """Test that query parameters accept correct types."""
    # Valid integer parameters
    response = test_client.get("/observations/from?year=2023&month=7&day=1")
    
    # Invalid parameter types should be handled by FastAPI validation
    response = test_client.get("/observations/from?year=abc&month=7&day=1")
    assert response.status_code == 422  # Validation error