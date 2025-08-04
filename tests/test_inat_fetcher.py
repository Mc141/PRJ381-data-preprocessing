import pytest
import datetime
import httpx
from unittest.mock import AsyncMock
from app.services.inat_fetcher import get_pages, get_observations



# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_api_response():
    return {
        "total_results": 60,
        "results": [
            {
                "id": 1,
                "uuid": "abc-123",
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
            }
        ]
    }


@pytest.fixture
def mock_httpx_get(mocker, mock_api_response):
    """
    Mock httpx.AsyncClient.get to return a fake API response.
    This will be used for both the initial and paginated requests.
    """
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_api_response
    mock_response.raise_for_status = mocker.Mock()

    mock_get = AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient.get", mock_get)
    return mock_get


# -----------------------------
# Tests for get_pages
# -----------------------------

@pytest.mark.asyncio
async def test_get_pages_returns_pages(mock_httpx_get):
    pages = await get_pages()

    assert isinstance(pages, list)
    assert len(pages) == 2  # 1 initial + 1 extra page since total_results=60, 30 per page
    assert all("results" in page for page in pages)
    assert mock_httpx_get.call_count == 2


# -----------------------------
# Parameterized test for get_observations
# -----------------------------

@pytest.mark.parametrize("positional_accuracy,expected_count", [
    (50, 1),  # valid
    (150, 0),  # invalid
    (None, 0),  # invalid
    ("not-an-int", 0)  # invalid
])
def test_get_observations_filters_accuracy(positional_accuracy, expected_count):
    page = {
        "results": [
            {
                "id": 1,
                "uuid": "abc-123",
                "time_observed_at": "2025-07-01T12:00:00Z",
                "created_at": "2025-07-01T13:00:00Z",
                "latitude": -33.9,
                "longitude": 18.4,
                "positional_accuracy": positional_accuracy,
                "place_guess": "Cape Town",
                "scientific_name": "Exampleus plantus",
                "common_name": "Example Plant",
                "quality_grade": "research",
                "image_url": "https://example.com/image.jpg",
                "user_id": 101
            }
        ]
    }
    result = get_observations([page])
    assert len(result) == expected_count


# -----------------------------
# Test for observation structure
# -----------------------------

def test_get_observations_output_structure(mock_api_response):
    result = get_observations([mock_api_response])

    assert isinstance(result, list)
    assert len(result) == 1
    obs = result[0]
    expected_keys = [
        "id", "uuid", "time_observed_at", "created_at", "latitude", "longitude",
        "positional_accuracy", "place_guess", "scientific_name", "common_name",
        "quality_grade", "image_url", "user_id"
    ]
    for key in expected_keys:
        assert key in obs
