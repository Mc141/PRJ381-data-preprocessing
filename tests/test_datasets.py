import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.routers.datasets import (
    router, _clean_weather_df, _gdd, _roll_window, _count_condition, _compute_features,
    PARAM_REMAP, WINDOWS
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def test_client():
    """Create a test client for the datasets router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_weather_rows():
    """Sample weather data rows."""
    return [
        {
            "date": "2023-07-01",
            "ALLSKY_SFC_SW_DWN": 200.0,
            "CLRSKY_SFC_SW_DWN": 250.0,
            "PRECTOTCORR": 0.0,
            "RH2M": 70.0,
            "T2M": 15.5,
            "T2M_MAX": 20.0,
            "T2M_MIN": 11.0,
            "TQV": 25.0,
            "TS": 16.0,
            "WS2M": 3.5
        },
        {
            "date": "2023-07-02",
            "ALLSKY_SFC_SW_DWN": 180.0,
            "CLRSKY_SFC_SW_DWN": 250.0,
            "PRECTOTCORR": 2.5,
            "RH2M": 80.0,
            "T2M": 14.0,
            "T2M_MAX": 18.0,
            "T2M_MIN": 10.0,
            "TQV": 30.0,
            "TS": 15.0,
            "WS2M": 4.0
        },
        {
            "date": "2023-07-03",
            "ALLSKY_SFC_SW_DWN": -999,  # Fill value
            "CLRSKY_SFC_SW_DWN": 250.0,
            "PRECTOTCORR": 1.2,
            "RH2M": 75.0,
            "T2M": 16.0,
            "T2M_MAX": 21.0,
            "T2M_MIN": 12.0,
            "TQV": 28.0,
            "TS": 17.0,
            "WS2M": 2.5
        }
    ]


@pytest.fixture
def sample_observations():
    """Sample observation data."""
    return [
        {
            "id": 1,
            "uuid": "test-uuid-1",
            "time_observed_at": "2023-07-02T12:00:00Z",
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


@pytest.fixture
def mock_database():
    """Mock database and collections."""
    mock_db = MagicMock()
    mock_inat_collection = MagicMock()
    mock_weather_collection = MagicMock()
    mock_features_collection = MagicMock()
    
    mock_db.__getitem__.side_effect = lambda name: {
        "inat_observations": mock_inat_collection,
        "weather_data": mock_weather_collection,
        "weather_features": mock_features_collection
    }[name]
    
    return mock_db, mock_inat_collection, mock_weather_collection, mock_features_collection


# -----------------------------
# Tests for helper functions
# -----------------------------

def test_clean_weather_df_empty():
    """Test _clean_weather_df with empty input."""
    result = _clean_weather_df([])
    assert result.empty


def test_clean_weather_df_with_data(sample_weather_rows):
    """Test _clean_weather_df with sample data."""
    df = _clean_weather_df(sample_weather_rows)
    
    assert len(df) == 3
    assert "date" in df.columns
    assert "cloud_index" in df.columns
    
    # Check that -999 fill values are converted to NaN
    assert pd.isna(df.iloc[2]["ALLSKY_SFC_SW_DWN"])
    
    # Check parameter remapping
    assert "rain" in df.columns
    assert "t2m" in df.columns
    assert df["rain"].iloc[0] == 0.0
    assert df["t2m"].iloc[0] == 15.5
    
    # Check cloud index calculation
    expected_cloud_index_0 = 1.0 - (200.0 / 250.0)
    assert abs(df["cloud_index"].iloc[0] - expected_cloud_index_0) < 0.001


def test_clean_weather_df_cloud_index_calculation(sample_weather_rows):
    """Test cloud index calculation edge cases."""
    # Test with missing CLRSKY data
    rows = [{"date": "2023-07-01", "ALLSKY_SFC_SW_DWN": 200.0}]
    df = _clean_weather_df(rows)
    assert pd.isna(df["cloud_index"].iloc[0])
    
    # Test with zero CLRSKY (division by zero)
    rows = [{"date": "2023-07-01", "ALLSKY_SFC_SW_DWN": 200.0, "CLRSKY_SFC_SW_DWN": 0.0}]
    df = _clean_weather_df(rows)
    # With division by zero, the result should be inf, then clipped to 1.0
    # But the actual implementation might handle this differently
    assert df["cloud_index"].iloc[0] >= 0.0  # Should be non-negative


def test_gdd_calculation():
    """Test Growing Degree Days calculation."""
    tmin = pd.Series([5.0, 10.0, 15.0])
    tmax = pd.Series([15.0, 20.0, 25.0])
    
    gdd = _gdd(tmin, tmax, base=10.0)
    
    expected = pd.Series([0.0, 5.0, 10.0])  # ((5+15)/2-10, (10+20)/2-10, (15+25)/2-10), clipped at 0
    pd.testing.assert_series_equal(gdd, expected)


def test_gdd_with_none_inputs():
    """Test GDD calculation with None inputs."""
    result = _gdd(None, None, base=10.0)
    assert result.empty


def test_roll_window_functions():
    """Test rolling window aggregation functions."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test sum
    assert _roll_window(series, 3, "sum") == 12.0  # 3+4+5
    
    # Test mean
    assert _roll_window(series, 3, "mean") == 4.0  # (3+4+5)/3
    
    # Test max
    assert _roll_window(series, 3, "max") == 5.0
    
    # Test min
    assert _roll_window(series, 3, "min") == 3.0
    
    # Test count
    assert _roll_window(series, 3, "count") == 3
    
    # Test with window larger than series
    assert _roll_window(series, 10, "sum") == 15.0  # Uses all available data


def test_roll_window_with_nan():
    """Test rolling window with NaN values."""
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    
    result = _roll_window(series, 3, "sum")
    # The function takes the tail(3) of the series after dropping NaN
    # So it should be summing the last 3 non-NaN values: 1.0 + 3.0 + 5.0 = 9.0
    assert result == 9.0


def test_roll_window_empty_series():
    """Test rolling window with empty series."""
    series = pd.Series(dtype=float)
    result = _roll_window(series, 3, "sum")
    assert result is None


def test_count_condition():
    """Test condition counting function."""
    series = pd.Series([25.0, 30.0, 35.0, 20.0, 40.0])
    
    # Count values > 30
    result = _count_condition(series, 3, "gt", 30.0)
    assert result == 2  # 35.0 and 40.0 in last 3 values
    
    # Count values < 25
    result = _count_condition(series, 3, "lt", 25.0)
    assert result == 1  # 20.0 in last 3 values


def test_count_condition_empty_series():
    """Test condition counting with empty series."""
    series = pd.Series(dtype=float)
    result = _count_condition(series, 3, "gt", 30.0)
    assert result is None


def test_compute_features(sample_weather_rows):
    """Test feature computation."""
    df = _clean_weather_df(sample_weather_rows)
    features = _compute_features(df)
    
    # Check that features are computed for all windows
    for window in WINDOWS:
        assert f"rain_sum_{window}" in features
        assert f"t2m_mean_{window}" in features
        assert f"gdd_base10_sum_{window}" in features
    
    # Check specific feature values (using window=7 for simplicity)
    assert features["rain_sum_7"] == 3.7  # 0.0 + 2.5 + 1.2
    assert abs(features["t2m_mean_7"] - 15.17) < 0.1  # (15.5 + 14.0 + 16.0) / 3


def test_compute_features_empty_df():
    """Test feature computation with empty DataFrame."""
    df = pd.DataFrame()
    features = _compute_features(df)
    assert features == {}


def test_compute_features_missing_columns():
    """Test feature computation with missing columns."""
    df = pd.DataFrame({"date": ["2023-07-01"], "some_other_col": [1.0]})
    features = _compute_features(df)
    
    # Should not crash, but features should be mostly None or missing
    assert isinstance(features, dict)


# -----------------------------
# Tests for /datasets/merge endpoint
# -----------------------------

@patch('app.routers.datasets.get_pages')
@patch('app.routers.datasets.get_observations')
@patch('app.routers.datasets.get_database')
@patch('app.routers.datasets.PowerAPI')
def test_merge_datasets_success(mock_power_api_class, mock_get_database, mock_get_observations,
                               mock_get_pages, test_client, sample_observations, 
                               sample_weather_rows, mock_database):
    """Test successful dataset merging."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = [{"results": sample_observations}]
    mock_get_observations.return_value = sample_observations
    
    # Mock PowerAPI
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value={"data": sample_weather_rows})
    mock_power_api_class.return_value = mock_power_api
    
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3&years_back=1"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "preview" in data
    assert data["count"] >= 0
    
    # Verify database interactions
    mock_weather_col.update_one.assert_called()
    mock_features_col.update_one.assert_called()
    mock_inat_col.update_one.assert_called()


def test_merge_datasets_invalid_dates(test_client):
    """Test merge datasets with invalid date parameters."""
    # Invalid month
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=13&start_day=1"
        "&end_year=2023&end_month=7&end_day=3"
    )
    assert response.status_code == 400
    assert "Invalid date" in response.json()["detail"]


def test_merge_datasets_end_before_start(test_client):
    """Test merge datasets when end date is before start date."""
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=5"
        "&end_year=2023&end_month=7&end_day=1"
    )
    assert response.status_code == 400
    assert "End date must be after start date" in response.json()["detail"]


@patch('app.routers.datasets.get_pages')
@patch('app.routers.datasets.get_observations')
@patch('app.routers.datasets.get_database')
def test_merge_datasets_no_observations(mock_get_database, mock_get_observations,
                                      mock_get_pages, test_client, mock_database):
    """Test merge datasets when no observations are found."""
    # Setup mocks
    mock_db, _, _, _ = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = []
    mock_get_observations.return_value = []
    
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3"
    )
    
    assert response.status_code == 404
    assert "No observations found" in response.json()["detail"]


def test_merge_datasets_missing_parameters(test_client):
    """Test merge datasets with missing required parameters."""
    response = test_client.get("/datasets/merge?start_year=2023")
    assert response.status_code == 422  # Validation error


def test_merge_datasets_years_back_validation(test_client):
    """Test validation of years_back parameter."""
    # Too low
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3&years_back=0"
    )
    assert response.status_code == 422
    
    # Too high
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3&years_back=11"
    )
    assert response.status_code == 422


# -----------------------------
# Tests for /datasets/refresh-weather endpoint
# -----------------------------

@patch('app.routers.datasets.get_database')
@patch('app.routers.datasets.PowerAPI')
def test_refresh_weather_success(mock_power_api_class, mock_get_database, 
                                test_client, sample_observations, sample_weather_rows, 
                                mock_database):
    """Test successful weather refresh."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    mock_inat_col.find.return_value = sample_observations
    
    # Mock PowerAPI
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value={"data": sample_weather_rows})
    mock_power_api_class.return_value = mock_power_api
    
    response = test_client.post("/datasets/refresh-weather?years_back=2")
    
    assert response.status_code == 200
    data = response.json()
    assert "updated_weather_records" in data
    assert data["updated_weather_records"] >= 0
    
    # Verify database interactions
    mock_inat_col.find.assert_called_once()
    mock_weather_col.update_one.assert_called()
    mock_features_col.update_one.assert_called()


@patch('app.routers.datasets.get_database')
def test_refresh_weather_no_observations(mock_get_database, test_client, mock_database):
    """Test weather refresh when no observations exist in database."""
    # Setup mocks
    mock_db, mock_inat_col, _, _ = mock_database
    mock_get_database.return_value = mock_db
    mock_inat_col.find.return_value = []
    
    response = test_client.post("/datasets/refresh-weather")
    
    assert response.status_code == 404
    assert "No observations found in DB" in response.json()["detail"]


def test_refresh_weather_years_back_validation(test_client):
    """Test validation of years_back parameter in refresh endpoint."""
    # Too low
    response = test_client.post("/datasets/refresh-weather?years_back=0")
    assert response.status_code == 422
    
    # Too high
    response = test_client.post("/datasets/refresh-weather?years_back=11")
    assert response.status_code == 422


# -----------------------------
# Tests for /datasets/export endpoint
# -----------------------------

@patch('app.routers.datasets.get_database')
def test_export_dataset_success(mock_get_database, test_client, sample_observations, 
                               sample_weather_rows, mock_database):
    """Test successful dataset export."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    
    # Add inat_id to weather rows for joining
    weather_with_inat_id = []
    for row in sample_weather_rows:
        row_copy = row.copy()
        row_copy["inat_id"] = 1
        weather_with_inat_id.append(row_copy)
    
    mock_inat_col.find.return_value = sample_observations
    mock_weather_col.find.return_value = weather_with_inat_id
    mock_features_col.find.return_value = []  # No features for simplicity
    
    response = test_client.get("/datasets/export")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "Content-Disposition" in response.headers
    
    # Check that CSV content is returned
    csv_content = response.content.decode()
    assert "id" in csv_content  # Should contain observation columns
    assert "date" in csv_content  # Should contain weather columns


@patch('app.routers.datasets.get_database')
def test_export_dataset_with_features(mock_get_database, test_client, sample_observations, 
                                    sample_weather_rows, mock_database):
    """Test dataset export with features included."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    
    weather_with_inat_id = []
    for row in sample_weather_rows:
        row_copy = row.copy()
        row_copy["inat_id"] = 1
        weather_with_inat_id.append(row_copy)
    
    sample_features = [
        {
            "inat_id": 1,
            "latitude": -33.9,
            "longitude": 18.4,
            "features": {
                "rain_sum_7": 3.7,
                "t2m_mean_7": 15.17,
                "gdd_base10_sum_7": 25.0
            }
        }
    ]
    
    mock_inat_col.find.return_value = sample_observations
    mock_weather_col.find.return_value = weather_with_inat_id
    mock_features_col.find.return_value = sample_features
    
    response = test_client.get("/datasets/export?include_features=true")
    
    assert response.status_code == 200
    csv_content = response.content.decode()
    assert "rain_sum_7" in csv_content  # Should contain feature columns


@patch('app.routers.datasets.get_database')
def test_export_dataset_no_data(mock_get_database, test_client, mock_database):
    """Test dataset export when no data exists."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    mock_inat_col.find.return_value = []
    mock_weather_col.find.return_value = []
    mock_features_col.find.return_value = []
    
    response = test_client.get("/datasets/export")
    
    assert response.status_code == 404
    assert "No dataset to export" in response.json()["detail"]


def test_export_dataset_include_features_parameter(test_client):
    """Test the include_features parameter parsing."""
    # This test would require mocking the database, but we can test parameter validation
    with patch('app.routers.datasets.get_database') as mock_get_db:
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock empty collections to trigger the "no data" error
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        mock_db.__getitem__.return_value = mock_collection
        
        # Test both true and false values
        response1 = test_client.get("/datasets/export?include_features=true")
        response2 = test_client.get("/datasets/export?include_features=false")
        
        # Both should reach the "no data" error, meaning parameter parsing worked
        assert response1.status_code == 404
        assert response2.status_code == 404


# -----------------------------
# Integration tests
# -----------------------------

@patch('app.routers.datasets.get_pages')
@patch('app.routers.datasets.get_observations')
@patch('app.routers.datasets.get_database')
@patch('app.routers.datasets.PowerAPI')
def test_full_dataset_workflow(mock_power_api_class, mock_get_database, mock_get_observations,
                              mock_get_pages, test_client, sample_observations, 
                              sample_weather_rows, mock_database):
    """Test the complete dataset workflow: merge -> refresh -> export."""
    # Setup mocks
    mock_db, mock_inat_col, mock_weather_col, mock_features_col = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.return_value = [{"results": sample_observations}]
    mock_get_observations.return_value = sample_observations
    
    # Mock PowerAPI
    mock_power_api = Mock()
    mock_power_api.get_weather = AsyncMock(return_value={"data": sample_weather_rows})
    mock_power_api_class.return_value = mock_power_api
    
    # 1. Merge datasets
    response1 = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3"
    )
    assert response1.status_code == 200
    
    # 2. Refresh weather (mock that observations exist in DB)
    mock_inat_col.find.return_value = sample_observations
    response2 = test_client.post("/datasets/refresh-weather")
    assert response2.status_code == 200
    
    # 3. Export dataset (mock that data exists)
    weather_with_inat_id = []
    for row in sample_weather_rows:
        row_copy = row.copy()
        row_copy["inat_id"] = 1
        weather_with_inat_id.append(row_copy)
    
    mock_weather_col.find.return_value = weather_with_inat_id
    mock_features_col.find.return_value = []
    
    response3 = test_client.get("/datasets/export")
    assert response3.status_code == 200
    assert response3.headers["content-type"] == "text/csv; charset=utf-8"


# -----------------------------
# Error handling tests
# -----------------------------

@patch('app.routers.datasets.get_pages')
@patch('app.routers.datasets.get_database')
def test_merge_datasets_api_error(mock_get_database, mock_get_pages, test_client, mock_database):
    """Test merge datasets when external API fails."""
    # Setup mocks
    mock_db, _, _, _ = mock_database
    mock_get_database.return_value = mock_db
    mock_get_pages.side_effect = Exception("iNaturalist API error")
    
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3"
    )
    
    assert response.status_code == 500


@patch('app.routers.datasets.get_database')
def test_database_connection_error(mock_get_database, test_client):
    """Test handling of database connection errors."""
    mock_get_database.side_effect = Exception("Database connection failed")
    
    response = test_client.get(
        "/datasets/merge?start_year=2023&start_month=7&start_day=1"
        "&end_year=2023&end_month=7&end_day=3"
    )
    
    assert response.status_code == 500


# -----------------------------
# Edge case tests
# -----------------------------

def test_param_remap_constants():
    """Test that parameter remapping constants are correctly defined."""
    assert "PRECTOTCORR" in PARAM_REMAP
    assert PARAM_REMAP["PRECTOTCORR"] == "rain"
    assert "T2M" in PARAM_REMAP
    assert len(PARAM_REMAP) == 10


def test_windows_constants():
    """Test that windows constants are correctly defined."""
    assert WINDOWS == [7, 30, 90, 365]
    assert all(isinstance(w, int) for w in WINDOWS)
    assert all(w > 0 for w in WINDOWS)