"""
Tests for status router endpoints.

Tests health checks and service information endpoints
for monitoring and system status.
"""

import pytest
from fastapi.testclient import TestClient
from app.routers.status import router


@pytest.fixture
def test_client():
    """Create a test client for the status router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/status/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_service_info(test_client):
    """Test the service info endpoint."""
    response = test_client.get("/status/service_info")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PRJ381 Data Preprocessing API"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)
    assert len(data["endpoints"]) > 0
    
    # Check that required endpoints are mentioned
    endpoints_text = " ".join(data["endpoints"])
    assert "observations" in endpoints_text.lower()
    assert "weather" in endpoints_text.lower()


def test_endpoints_return_json(test_client):
    """Test that all endpoints return valid JSON."""
    endpoints = ["/status/health", "/status/service_info"]
    
    for endpoint in endpoints:
        response = test_client.get(endpoint)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        # Should not raise an exception
        data = response.json()
        assert isinstance(data, dict)
