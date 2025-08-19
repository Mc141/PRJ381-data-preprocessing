"""
Comprehensive tests for NASA POWER API service.

Tests the async PowerAPI class including initialization, date chunking,
URL building, and weather data retrieval functionality.
"""

import pytest
import datetime
from unittest.mock import Mock, AsyncMock, patch
from app.services.nasa_fetcher import PowerAPI, get_weather_sync


def test_power_api_init_with_dates():
    """Test initialization with date objects."""
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4
    )
    
    assert api.lat == -33.9
    assert api.long == 18.4
    assert api.start == datetime.datetime(2023, 7, 1)
    assert api.end == datetime.datetime(2023, 7, 2)


def test_power_api_init_with_datetime():
    """Test initialization with datetime objects."""
    api = PowerAPI(
        start=datetime.datetime(2023, 7, 1, 12, 0),
        end=datetime.datetime(2023, 7, 2, 12, 0),
        lat=40.7,
        long=-74.0
    )
    
    assert api.lat == 40.7
    assert api.long == -74.0
    assert api.start == datetime.datetime(2023, 7, 1, 12, 0)
    assert api.end == datetime.datetime(2023, 7, 2, 12, 0)


def test_power_api_init_default_parameters():
    """Test initialization with default API parameters."""
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4
    )
    
    # Check default parameters are set
    assert "T2M" in api.parameter
    assert "PRECTOTCORR" in api.parameter
    assert api.chunk_months == 12
    assert api.max_concurrent == 5


def test_power_api_init_custom_parameters():
    """Test initialization with custom parameters."""
    custom_params = ["T2M", "RH2M"]
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4,
        parameter=custom_params,
        chunk_months=36,
        max_concurrent=10
    )
    
    assert api.parameter == custom_params
    assert api.chunk_months == 36
    assert api.max_concurrent == 10


def test_date_chunking_short_range():
    """Test that short date ranges don't get chunked."""
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4,
        chunk_months=36
    )
    
    chunks = api._generate_date_chunks()
    assert len(chunks) == 1
    assert chunks[0][0] == datetime.datetime(2023, 7, 1)
    assert chunks[0][1] == datetime.datetime(2023, 7, 2)


def test_date_chunking_multi_year():
    """Test chunking for multi-year date ranges."""
    api = PowerAPI(
        start=datetime.date(2020, 1, 1),
        end=datetime.date(2023, 12, 31),
        lat=-33.9,
        long=18.4,
        chunk_months=36  # 3 years per chunk
    )
    
    chunks = api._generate_date_chunks()
    assert len(chunks) == 2  # Should create 2 chunks for 4-year range
    
    # First chunk: 2020-2022 (3 years)
    assert chunks[0][0] == datetime.datetime(2020, 1, 1)
    assert chunks[0][1].year == 2022
    
    # Second chunk: 2023
    assert chunks[1][0].year == 2023
    assert chunks[1][1] == datetime.datetime(2023, 12, 31)


def test_date_chunking_custom_chunk_size():
    """Test chunking with custom chunk size."""
    api = PowerAPI(
        start=datetime.date(2022, 1, 1),
        end=datetime.date(2023, 12, 31),
        lat=-33.9,
        long=18.4,
        chunk_months=12  # 1 year per chunk
    )
    
    chunks = api._generate_date_chunks()
    assert len(chunks) == 2  # Should create 2 chunks for 2-year range


def test_build_request_format():
    """Test that request URLs are built correctly."""
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4,
        parameter=["T2M", "PRECTOTCORR"]
    )
    
    url = api._build_request(
        datetime.datetime(2023, 7, 1),
        datetime.datetime(2023, 7, 2)
    )
    
    # Check URL contains required components
    assert "power.larc.nasa.gov" in url
    assert "latitude=-33.9" in url
    assert "longitude=18.4" in url
    assert "start=20230701" in url
    assert "end=20230702" in url
    assert "T2M" in url
    assert "PRECTOTCORR" in url


def test_sync_wrapper_exists():
    """Test that sync wrapper function is available."""
    assert callable(get_weather_sync)


@patch('app.services.nasa_fetcher.asyncio.run')
def test_sync_wrapper_calls_async(mock_run):
    """Test that sync wrapper properly calls async function."""
    mock_run.return_value = {"data": []}
    
    result = get_weather_sync(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4
    )
    
    # Verify asyncio.run was called
    mock_run.assert_called_once()
    assert result == {"data": []}


def test_power_api_init_validation():
    """Test initialization parameter validation."""
    api = PowerAPI(
        start=datetime.date(2023, 7, 1),
        end=datetime.date(2023, 7, 2),
        lat=-33.9,
        long=18.4,
        parameter=["T2M", "PRECTOTCORR"]
    )
    
    # Verify parameters are stored correctly
    assert "T2M" in api.parameter
    assert "PRECTOTCORR" in api.parameter
    assert len(api.parameter) == 2
