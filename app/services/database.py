"""
Database Service Module
=======================

This module provides MongoDB connection management and database access utilities
for the PRJ381 data preprocessing application.

The module handles:
    - MongoDB connection lifecycle management
    - Database and collection access
    - Environment-based configuration
    - Connection state management

Collections:
    - inat_observations: Species observation data from iNaturalist
    - weather_data: Weather data from NASA POWER API
    - system_stats: Application statistics and metadata

Environment Variables:
    - MONGO_URI: MongoDB connection string (default: mongodb://localhost:27017)
    - MONGO_DB_NAME: Database name (default: invasive_db)

Example:
    Basic usage::
    
        from app.services.database import connect_to_mongo, get_database
        
        # Connect to database
        connect_to_mongo()
        
        # Get database instance
        db = get_database()
        
        # Access collections
        observations = get_inat_collection()

Author: MC141
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
"""str: MongoDB connection URI from environment or default localhost"""

DB_NAME = os.getenv("MONGO_DB_NAME", "invasive_db")
"""str: Database name from environment or default 'invasive_db'"""

# Global variables for client and database
client: MongoClient = None
"""MongoClient: Global MongoDB client instance"""

db = None
"""Database: Global database instance"""


def connect_to_mongo():
    """
    Establish connection to MongoDB.
    
    Creates a global MongoDB client and database connection using the
    configured URI and database name from environment variables.
    
    Raises:
        pymongo.errors.ConnectionFailure: If connection to MongoDB fails
        pymongo.errors.ConfigurationError: If URI is malformed
        
    Note:
        This function should be called once at application startup.
        The connection is stored in global variables for reuse.
    """
    global client, db
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"Connected to MongoDB: {MONGO_URI}/{DB_NAME}")


def close_mongo_connection():
    """
    Close MongoDB connection.
    
    Properly closes the global MongoDB client connection if it exists.
    This should be called at application shutdown to prevent connection leaks.
    
    Note:
        Safe to call multiple times or when no connection exists.
    """
    global client
    if client:
        client.close()
        print("MongoDB connection closed")


# Collection access helper functions

def get_database():
    """
    Get the current database instance.
    
    Returns:
        Database: MongoDB database instance
        
    Raises:
        AttributeError: If database connection has not been established
        
    Note:
        Requires connect_to_mongo() to be called first.
    """
    return db


def get_inat_collection():
    """
    Get the iNaturalist observations collection.
    
    Returns:
        Collection: MongoDB collection containing species observation data
        
    Note:
        Collection stores documents with observation metadata, coordinates,
        species information, and timestamps from iNaturalist API.
    """
    return db["inat_observations"]


def get_weather_collection():
    """
    Get the weather data collection.
    
    Returns:
        Collection: MongoDB collection containing weather data
        
    Note:
        Collection stores daily weather records from NASA POWER API
        with meteorological parameters like temperature, precipitation, etc.
    """
    return db["weather_data"]


def get_stats_collection():
    """
    Get the system statistics collection.
    
    Returns:
        Collection: MongoDB collection for application statistics
        
    Note:
        Collection stores metadata about data processing runs,
        API usage statistics, and system performance metrics.
    """
    return db["system_stats"]
