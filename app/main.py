"""
PRJ381 Data Preprocessing Application
====================================

A FastAPI-based web service for preprocessing invasive species observation data.
This application integrates data from iNaturalist observations with NASA POWER weather data
to create comprehensive datasets for ecological analysis.

Main Features:
    - Fetch and process iNaturalist observation data
    - Retrieve weather data from NASA POWER API
    - Merge observations with historical weather data
    - Export processed datasets for analysis
    - MongoDB integration for data persistence

Modules:
    - observations: Endpoints for retrieving and managing species observations
    - weather: Endpoints for weather data retrieval and storage
    - datasets: Endpoints for creating merged datasets with features

Author: MC141
Project: PRJ381 Data Preprocessing
"""

from fastapi import FastAPI
from app.services.database import connect_to_mongo, close_mongo_connection
from app.routers import observations, weather, datasets, status
from contextlib import asynccontextmanager  


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for FastAPI.
    
    Handles startup and shutdown events for the application, including
    database connection management.
    
    Args:
        app (FastAPI): The FastAPI application instance
        
    Yields:
        None: Control back to FastAPI during application runtime
        
    Note:
        This ensures MongoDB connections are properly opened at startup
        and closed at shutdown to prevent connection leaks.
    """
    # Startup: Connect to MongoDB
    connect_to_mongo()
    yield
    # Shutdown: Close MongoDB connection
    close_mongo_connection()


app = FastAPI(
    title="PRJ381 Data Preprocessing API",
    description="A comprehensive API for preprocessing invasive species observation data with weather integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Include API routers
app.include_router(observations.router, prefix="/api/v1", tags=["observations"])
app.include_router(weather.router, prefix="/api/v1", tags=["weather"])
app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
