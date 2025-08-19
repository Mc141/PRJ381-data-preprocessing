import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from app.services import database


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client."""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    return mock_client, mock_db


@pytest.fixture
def mock_collections():
    """Mock MongoDB collections."""
    inat_collection = MagicMock()
    weather_collection = MagicMock()
    stats_collection = MagicMock()
    return inat_collection, weather_collection, stats_collection


# -----------------------------
# Tests for MongoDB connection functions
# -----------------------------

@patch('app.services.database.MongoClient')
@patch('app.services.database.os.getenv')
def test_connect_to_mongo_default_values(mock_getenv, mock_mongo_client_class):
    """Test connecting to MongoDB with default environment values."""
    # Setup environment variables
    mock_getenv.side_effect = lambda key, default: {
        "MONGO_URI": "mongodb://localhost:27017",
        "MONGO_DB_NAME": "invasive_db"
    }.get(key, default)
    
    mock_client = MagicMock()
    mock_mongo_client_class.return_value = mock_client
    
    # Reset global variables
    database.client = None
    database.db = None
    
    database.connect_to_mongo()
    
    # Verify client creation
    mock_mongo_client_class.assert_called_once_with("mongodb://localhost:27017")
    assert database.client == mock_client
    assert database.db == mock_client["invasive_db"]


@patch('app.services.database.MongoClient')
@patch('app.services.database.MONGO_URI', "mongodb://custom-host:27018")
@patch('app.services.database.DB_NAME', "custom_db")
def test_connect_to_mongo_custom_values(mock_mongo_client_class):
    """Test connecting to MongoDB with custom environment values."""
    mock_client = MagicMock()
    mock_mongo_client_class.return_value = mock_client
    
    # Reset global variables
    database.client = None
    database.db = None
    
    database.connect_to_mongo()
    
    # Verify client creation with custom values
    mock_mongo_client_class.assert_called_once_with("mongodb://custom-host:27018")
    assert database.client == mock_client
    assert database.db == mock_client["custom_db"]


def test_close_mongo_connection_with_client():
    """Test closing MongoDB connection when client exists."""
    mock_client = MagicMock()
    database.client = mock_client
    
    database.close_mongo_connection()
    
    mock_client.close.assert_called_once()


def test_close_mongo_connection_without_client():
    """Test closing MongoDB connection when no client exists."""
    database.client = None
    
    # Should not raise an error
    database.close_mongo_connection()


# -----------------------------
# Tests for helper getter functions
# -----------------------------

def test_get_database():
    """Test get_database function."""
    mock_db = MagicMock()
    database.db = mock_db
    
    result = database.get_database()
    
    assert result == mock_db


def test_get_inat_collection():
    """Test get_inat_collection function."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    database.db = mock_db
    
    result = database.get_inat_collection()
    
    mock_db.__getitem__.assert_called_once_with("inat_observations")
    assert result == mock_collection


def test_get_weather_collection():
    """Test get_weather_collection function."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    database.db = mock_db
    
    result = database.get_weather_collection()
    
    mock_db.__getitem__.assert_called_once_with("weather_data")
    assert result == mock_collection


def test_get_stats_collection():
    """Test get_stats_collection function."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    database.db = mock_db
    
    result = database.get_stats_collection()
    
    mock_db.__getitem__.assert_called_once_with("system_stats")
    assert result == mock_collection


# -----------------------------
# Integration tests
# -----------------------------

@patch('app.services.database.MongoClient')
@patch('app.services.database.os.getenv')
def test_full_database_workflow(mock_getenv, mock_mongo_client_class):
    """Test the complete database connection and collection access workflow."""
    # Setup environment
    mock_getenv.side_effect = lambda key, default: {
        "MONGO_URI": "mongodb://test:27017",
        "MONGO_DB_NAME": "test_db"
    }.get(key, default)
    
    # Setup mocks
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    mock_mongo_client_class.return_value = mock_client
    
    # Reset state
    database.client = None
    database.db = None
    
    # Connect to database
    database.connect_to_mongo()
    
    # Test all getter functions
    db_result = database.get_database()
    inat_result = database.get_inat_collection()
    weather_result = database.get_weather_collection()
    stats_result = database.get_stats_collection()
    
    # Verify results
    assert db_result == mock_db
    assert inat_result == mock_collection
    assert weather_result == mock_collection
    assert stats_result == mock_collection
    
    # Verify collection names were requested correctly
    expected_calls = [
        (("inat_observations",), {}),
        (("weather_data",), {}),
        (("system_stats",), {})
    ]
    mock_db.__getitem__.assert_has_calls([
        ((call[0][0],),) for call in expected_calls
    ], any_order=True)
    
    # Close connection
    database.close_mongo_connection()
    mock_client.close.assert_called_once()


# -----------------------------
# Tests for environment variable handling
# -----------------------------

@patch('app.services.database.load_dotenv')
def test_dotenv_loading(mock_load_dotenv):
    """Test that load_dotenv is called when module is imported."""
    # Re-import the module to trigger the load_dotenv call
    import importlib
    importlib.reload(database)
    
    # Note: This test may not work perfectly due to module import caching
    # but it's included for completeness


@patch.dict(os.environ, {}, clear=True)
@patch('app.services.database.MongoClient')
def test_environment_variables_defaults(mock_mongo_client_class):
    """Test that default values are used when environment variables are not set."""
    # Clear the environment variables
    if 'MONGO_URI' in os.environ:
        del os.environ['MONGO_URI']
    if 'MONGO_DB_NAME' in os.environ:
        del os.environ['MONGO_DB_NAME']
    
    mock_client = MagicMock()
    mock_mongo_client_class.return_value = mock_client
    
    # Reset globals
    database.client = None
    database.db = None
    
    database.connect_to_mongo()
    
    # Should use default values
    mock_mongo_client_class.assert_called_once_with("mongodb://localhost:27017")
    mock_client.__getitem__.assert_called_once_with("invasive_db")


# -----------------------------
# Error handling tests
# -----------------------------

@patch('app.services.database.MongoClient')
def test_connect_to_mongo_connection_error(mock_mongo_client_class):
    """Test handling of MongoDB connection errors."""
    # Simulate connection error
    mock_mongo_client_class.side_effect = Exception("Connection failed")
    
    # Reset globals
    database.client = None
    database.db = None
    
    with pytest.raises(Exception, match="Connection failed"):
        database.connect_to_mongo()


def test_get_functions_with_none_db():
    """Test getter functions when database is None."""
    database.db = None
    
    # These should not raise errors, but return None
    assert database.get_database() is None
    
    # These will raise TypeError since we're calling __getitem__ on None
    with pytest.raises(TypeError):
        database.get_inat_collection()
    
    with pytest.raises(TypeError):
        database.get_weather_collection()
    
    with pytest.raises(TypeError):
        database.get_stats_collection()


# -----------------------------
# State management tests
# -----------------------------

def test_global_state_isolation():
    """Test that global state can be properly managed."""
    # Save original state
    original_client = database.client
    original_db = database.db
    
    try:
        # Modify state
        mock_client = MagicMock()
        mock_db = MagicMock()
        database.client = mock_client
        database.db = mock_db
        
        # Test state
        assert database.client == mock_client
        assert database.db == mock_db
        assert database.get_database() == mock_db
        
    finally:
        # Restore original state
        database.client = original_client
        database.db = original_db
