import pytest
import datetime
import httpx
import requests
from unittest.mock import AsyncMock, Mock, patch
from app.services.inat_fetcher import (
    get_pages, get_observations, get_observation_by_id,
    is_positional_accuracy_valid, extract_coordinates, extract_observation_data,
    fetch_page, _log
)


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
                "taxon": {
                    "name": "Exampleus plantus",
                    "preferred_common_name": "Example Plant"
                },
                "quality_grade": "research",
                "photos": [{"url": "https://example.com/image.jpg"}],
                "user": {"id": 101}
            }
        ]
    }


@pytest.fixture
def sample_observation():
    return {
        "id": 123,
        "uuid": "test-uuid-456",
        "time_observed_at": "2025-07-15T10:30:00Z",
        "created_at": "2025-07-15T11:00:00Z",
        "latitude": -33.8,
        "longitude": 18.5,
        "positional_accuracy": 25,
        "place_guess": "Stellenbosch",
        "taxon": {
            "name": "Testus scientificus",
            "preferred_common_name": "Test Plant"
        },
        "quality_grade": "research",
        "photos": [{"url": "https://example.com/test.jpg"}],
        "user": {"id": 202}
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
# Tests for _log function
# -----------------------------

def test_log_with_logger():
    """Test _log function with a logger provided."""
    mock_logger = Mock()
    _log(mock_logger, "test message")
    mock_logger.assert_called_once_with("test message")


def test_log_without_logger(capsys):
    """Test _log function without a logger (should print to console)."""
    _log(None, "test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out


# -----------------------------
# Tests for is_positional_accuracy_valid
# -----------------------------

@pytest.mark.parametrize("accuracy,max_accuracy,expected", [
    (50, 100, True),   # Valid accuracy
    (100, 100, True),  # Edge case: exactly max
    (150, 100, False), # Too high
    (None, 100, False), # None value
    ("50", 100, False), # String instead of int
    (50.5, 100, False), # Float instead of int
    (-10, 100, True),  # Negative value - the function allows this currently
])
def test_is_positional_accuracy_valid(accuracy, max_accuracy, expected):
    observation = {"positional_accuracy": accuracy}
    result = is_positional_accuracy_valid(observation, max_accuracy)
    assert result == expected


def test_is_positional_accuracy_valid_missing_key():
    """Test with observation missing positional_accuracy key."""
    observation = {"id": 123}
    result = is_positional_accuracy_valid(observation)
    assert result is False


# -----------------------------
# Tests for extract_coordinates
# -----------------------------

def test_extract_coordinates_from_geojson():
    """Test coordinate extraction from geojson field."""
    observation = {
        "geojson": {
            "type": "Point",
            "coordinates": [18.4, -33.9]
        }
    }
    lat, lon = extract_coordinates(observation)
    assert lat == -33.9
    assert lon == 18.4


def test_extract_coordinates_from_location():
    """Test coordinate extraction from location string field."""
    observation = {
        "location": "-33.9,18.4"
    }
    lat, lon = extract_coordinates(observation)
    assert lat == -33.9
    assert lon == 18.4


def test_extract_coordinates_invalid_location():
    """Test coordinate extraction with invalid location format."""
    observation = {
        "location": "invalid-format"
    }
    lat, lon = extract_coordinates(observation)
    assert lat is None
    assert lon is None


def test_extract_coordinates_no_coordinates():
    """Test coordinate extraction when no coordinates are available."""
    observation = {"id": 123}
    lat, lon = extract_coordinates(observation)
    assert lat is None
    assert lon is None


def test_extract_coordinates_invalid_geojson():
    """Test coordinate extraction with invalid geojson."""
    observation = {
        "geojson": {
            "type": "Polygon",  # Not Point
            "coordinates": [18.4, -33.9]
        }
    }
    lat, lon = extract_coordinates(observation)
    assert lat is None
    assert lon is None


# -----------------------------
# Tests for extract_observation_data
# -----------------------------

def test_extract_observation_data_complete(sample_observation):
    """Test extraction with complete observation data."""
    result = extract_observation_data(sample_observation)
    
    expected_keys = [
        "id", "uuid", "time_observed_at", "created_at", "latitude", "longitude",
        "positional_accuracy", "place_guess", "scientific_name", "common_name",
        "quality_grade", "image_url", "user_id"
    ]
    
    for key in expected_keys:
        assert key in result
    
    assert result["id"] == 123
    assert result["uuid"] == "test-uuid-456"
    assert result["scientific_name"] == "Testus scientificus"
    assert result["common_name"] == "Test Plant"
    assert result["image_url"] == "https://example.com/test.jpg"
    assert result["user_id"] == 202


def test_extract_observation_data_minimal():
    """Test extraction with minimal observation data."""
    observation = {
        "id": 456,
        "uuid": "minimal-uuid"
    }
    result = extract_observation_data(observation)
    
    assert result["id"] == 456
    assert result["uuid"] == "minimal-uuid"
    assert result["scientific_name"] is None
    assert result["common_name"] is None
    assert result["image_url"] is None
    assert result["user_id"] is None


def test_extract_observation_data_no_photos():
    """Test extraction when no photos are available."""
    observation = {
        "id": 789,
        "photos": []
    }
    result = extract_observation_data(observation)
    assert result["image_url"] is None


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


@pytest.mark.asyncio
async def test_get_pages_with_start_date(mock_httpx_get):
    """Test get_pages with custom start date."""
    start_date = datetime.datetime(2023, 1, 1)
    pages = await get_pages(start_date=start_date)
    
    assert isinstance(pages, list)
    assert len(pages) == 2
    # Verify the call was made with the correct date format
    args, kwargs = mock_httpx_get.call_args_list[0]
    # The date should be in the params as "2023-01-01"
    assert kwargs["params"]["d1"] == "2023-01-01"


@pytest.mark.asyncio
async def test_get_pages_with_logger(mock_httpx_get):
    """Test get_pages with logger provided."""
    mock_logger = Mock()
    pages = await get_pages(logger=mock_logger)
    
    assert isinstance(pages, list)
    assert len(pages) == 2


@pytest.mark.asyncio
async def test_get_pages_http_error(mocker):
    """Test get_pages handling HTTP errors."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=Mock(), response=Mock(status_code=404, text="Not Found")
    )
    
    mock_get = AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient.get", mock_get)
    
    mock_logger = Mock()
    pages = await get_pages(logger=mock_logger)
    
    assert pages == []
    mock_logger.assert_called()


@pytest.mark.asyncio
async def test_get_pages_timeout_error(mocker):
    """Test get_pages handling timeout errors."""
    mock_get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    mocker.patch("httpx.AsyncClient.get", mock_get)
    
    mock_logger = Mock()
    pages = await get_pages(logger=mock_logger)
    
    assert pages == []
    mock_logger.assert_called()


# -----------------------------
# Tests for fetch_page
# -----------------------------

@pytest.mark.asyncio
async def test_fetch_page_success(mocker):
    """Test successful page fetching."""
    mock_response = Mock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = Mock()
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    result = await fetch_page(mock_client, "https://api.test.com", {"page": 1}, 1)
    
    assert result == {"results": []}
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_page_http_error(mocker):
    """Test fetch_page handling HTTP errors."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Error", request=Mock(), response=Mock(status_code=500, text="Server Error")
    )
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    mock_logger = Mock()
    result = await fetch_page(mock_client, "https://api.test.com", {"page": 1}, 1, mock_logger)
    
    assert result is None
    mock_logger.assert_called()


@pytest.mark.asyncio
async def test_fetch_page_timeout_error():
    """Test fetch_page handling timeout errors."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("Timeout")
    
    mock_logger = Mock()
    result = await fetch_page(mock_client, "https://api.test.com", {"page": 1}, 1, mock_logger)
    
    assert result is None
    mock_logger.assert_called()


# -----------------------------
# Tests for get_observations
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
                "taxon": {
                    "name": "Exampleus plantus",
                    "preferred_common_name": "Example Plant"
                },
                "quality_grade": "research",
                "photos": [{"url": "https://example.com/image.jpg"}],
                "user": {"id": 101}
            }
        ]
    }
    result = get_observations([page])
    assert len(result) == expected_count


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


def test_get_observations_empty_pages():
    """Test get_observations with empty pages."""
    result = get_observations([])
    assert result == []


def test_get_observations_pages_no_results():
    """Test get_observations with pages containing no results."""
    pages = [{"total_results": 0, "results": []}]
    result = get_observations(pages)
    assert result == []


def test_get_observations_multiple_pages(sample_observation):
    """Test get_observations with multiple pages."""
    page1 = {"results": [sample_observation]}
    page2 = {"results": [sample_observation]}
    
    result = get_observations([page1, page2])
    assert len(result) == 2


# -----------------------------
# Tests for get_observation_by_id
# -----------------------------

@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_success(mock_get, sample_observation):
    """Test successful retrieval of observation by ID."""
    mock_response = Mock()
    mock_response.json.return_value = {"results": [sample_observation]}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    result = get_observation_by_id(123)
    
    assert result is not None
    assert result["id"] == 123
    mock_get.assert_called_once()


@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_not_found(mock_get):
    """Test get_observation_by_id when observation is not found."""
    mock_response = Mock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    result = get_observation_by_id(999)
    
    assert result is None


@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_invalid_accuracy(mock_get):
    """Test get_observation_by_id with invalid positional accuracy."""
    invalid_observation = {
        "id": 123,
        "positional_accuracy": 200  # Invalid (too high)
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {"results": [invalid_observation]}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    result = get_observation_by_id(123)
    
    assert result is None


@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_timeout_error(mock_get):
    """Test get_observation_by_id handling timeout error."""
    mock_get.side_effect = requests.exceptions.Timeout("Timeout")
    
    mock_logger = Mock()
    result = get_observation_by_id(123, logger=mock_logger)
    
    assert result is None
    mock_logger.assert_called()


@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_request_error(mock_get):
    """Test get_observation_by_id handling request error."""
    mock_get.side_effect = requests.exceptions.RequestException("Request failed")
    
    mock_logger = Mock()
    result = get_observation_by_id(123, logger=mock_logger)
    
    assert result is None
    mock_logger.assert_called()


@patch('app.services.inat_fetcher.requests.get')
def test_get_observation_by_id_unexpected_error(mock_get):
    """Test get_observation_by_id handling unexpected error."""
    mock_get.side_effect = Exception("Unexpected error")
    
    mock_logger = Mock()
    result = get_observation_by_id(123, logger=mock_logger)
    
    assert result is None
    mock_logger.assert_called()
