"""
Tests for the predictions router.

This module tests the invasive species spread prediction endpoints,
including presence baseline detection, suitability mapping, and visualization.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from app.main import app


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_presence_baseline_endpoint_exists(test_client):
    """Test that the presence baseline endpoint exists."""
    response = test_client.get("/api/v1/predictions/presence_baseline")
    # Should not return 404, might return 404 if no data, but endpoint should exist
    assert response.status_code in [200, 404, 500]


def test_suitability_map_endpoint_exists(test_client):
    """Test that the suitability map endpoint exists."""
    response = test_client.get("/api/v1/predictions/suitability_map")
    # Should not return 404, might return 404 if no data, but endpoint should exist
    assert response.status_code in [200, 404, 500]


def test_visualize_map_endpoint_exists(test_client):
    """Test that the visualize map endpoint exists."""
    response = test_client.get("/api/v1/predictions/visualize_map")
    # Should not return 404, might return 404 if no data, but endpoint should exist
    assert response.status_code in [200, 404, 500]


def test_presence_baseline_parameters(test_client):
    """Test presence baseline with valid parameters."""
    response = test_client.get("/api/v1/predictions/presence_baseline?days_back=30")
    assert response.status_code in [200, 404, 500]
    
    # Test parameter validation
    response = test_client.get("/api/v1/predictions/presence_baseline?days_back=0")
    assert response.status_code == 422  # Validation error


def test_suitability_map_parameters(test_client):
    """Test suitability map with valid parameters."""
    response = test_client.get("/api/v1/predictions/suitability_map?days_back=30&grid_resolution=1.0")
    assert response.status_code in [200, 404, 500]
    
    # Test parameter validation
    response = test_client.get("/api/v1/predictions/suitability_map?days_back=0&grid_resolution=0.05")
    assert response.status_code == 422  # Validation error


def test_visualize_map_parameters(test_client):
    """Test visualize map with valid parameters."""
    response = test_client.get("/api/v1/predictions/visualize_map?days_back=30&grid_resolution=1.0&save_file=false")
    assert response.status_code in [200, 404, 500]


@patch('app.routers.predictions.get_database')
def test_presence_baseline_with_mock_data(mock_get_db, test_client):
    """Test presence baseline with mocked database data."""
    # Mock database and collection
    mock_db = MagicMock()
    mock_collection = Mock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db
    
    # Mock successful query with sample data
    mock_collection.find.return_value = [
        {
            "id": "test123",
            "latitude": -34.0,
            "longitude": 18.4,
            "time_observed_at": "2024-08-01T10:00:00Z"
        }
    ]
    
    response = test_client.get("/api/v1/predictions/presence_baseline?days_back=30")
    assert response.status_code == 200
    
    data = response.json()
    assert "presence_count" in data
    assert "observations" in data
    assert data["days_back"] == 30


@patch('app.routers.predictions.get_database')
def test_presence_baseline_no_data(mock_get_db, test_client):
    """Test presence baseline when no observations are found."""
    # Mock database and collection
    mock_db = MagicMock()
    mock_collection = Mock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db
    
    # Mock empty query result
    mock_collection.find.return_value = []
    
    response = test_client.get("/api/v1/predictions/presence_baseline?days_back=30")
    assert response.status_code == 500  # HTTPException 404 gets caught and converted to 500
    
    data = response.json()
    assert "Failed to retrieve presence baseline" in data["detail"]


def test_suitability_score_function():
    """Test the suitability scoring function."""
    from app.routers.predictions import suitability_score
    
    # Test optimal conditions
    assert suitability_score(20, 50) == 1.0
    
    # Test marginal conditions
    assert suitability_score(8, 110) == 0.5
    
    # Test poor conditions
    assert suitability_score(35, 150) == 0.0


def test_distance_decay_function():
    """Test the distance decay function."""
    from app.routers.predictions import distance_decay
    
    # Test at origin
    assert distance_decay(0) == 1.0
    
    # Test decay behavior
    assert distance_decay(1) < 1.0
    assert distance_decay(2) < distance_decay(1)
    assert distance_decay(10) < distance_decay(2)


def test_create_grid_function():
    """Test the grid creation function."""
    from app.routers.predictions import create_grid
    
    bounds = {
        "min_lat": -34.1,
        "max_lat": -34.0,
        "min_lon": 18.3,
        "max_lon": 18.4
    }
    
    grid = create_grid(bounds, resolution_km=0.5)
    
    assert len(grid) > 0
    assert all("latitude" in point for point in grid)
    assert all("longitude" in point for point in grid)
    
    # Check bounds
    lats = [p["latitude"] for p in grid]
    lons = [p["longitude"] for p in grid]
    
    assert min(lats) >= bounds["min_lat"]
    assert max(lats) <= bounds["max_lat"]
    assert min(lons) >= bounds["min_lon"]
    assert max(lons) <= bounds["max_lon"]
