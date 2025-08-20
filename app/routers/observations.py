"""
Observations Router Module
==========================

FastAPI router for managing iNaturalist species observation data endpoints.
This module provides RESTful API endpoints for retrieving, processing, and storing
biodiversity observation data from the iNaturalist platform.

The router handles:
    - Fetching observation data from iNaturalist API
    - Optional database storage of retrieved observations
    - Date-based filtering for temporal analysis
    - Individual observation retrieval by ID
    - Database management operations

Endpoints:
    - GET /observations: Retrieve recent observations
    - GET /observations/from: Retrieve observations from specific date
    - GET /observations/{id}: Get individual observation by ID
    - GET /observations/db: Retrieve stored observations from database
    - DELETE /observations/db: Clear stored observations

Database Integration:
    Observations can be optionally stored in MongoDB for:
    - Offline analysis and processing
    - Historical data preservation  
    - Reduced API calls for repeated analysis
    - Data backup and archival

Usage Example:
    API client usage::
    
        import httpx
        
        # Get recent observations
        response = httpx.get("/observations")
        observations = response.json()
        
        # Get observations from specific date with storage
        response = httpx.get("/observations/from?year=2023&month=7&day=15&store_in_db=true")

Author: MC141
"""

import logging
import datetime
from typing import List, Dict
import copy

from fastapi import APIRouter, Query, Path, HTTPException
from app.services.inat_fetcher import get_pages, get_observations, get_observation_by_id
from app.services.database import get_database

# Setup logger
logger = logging.getLogger("inat")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter()

# MongoDB collection name
INAT_COLLECTION = "inat_observations"

@router.get("/observations", response_model=List[Dict])
async def read_observations(store_in_db: str = Query("false", description="Store fetched observations in MongoDB")):
    """
    Retrieve recent species observations from iNaturalist API.

    Fetches the most recent species observation data from iNaturalist for the
    configured geographic region (Cape Town area) and taxonomic group (plants).
    Optionally stores the retrieved data in MongoDB for offline analysis.

    Args:
        store_in_db (str, optional): Whether to store fetched observations in database.
            Accepts various formats: "true"/"false", "1"/"0", "yes"/"no", "on"/"off".
            Defaults to "false".

    Returns:
        List[Dict]: List of standardized observation records containing:
            - id: iNaturalist observation identifier
            - scientific_name: Species scientific name
            - common_name: Species common name  
            - latitude/longitude: GPS coordinates
            - observed_on: Observation date
            - image_url: Photo URL if available
            - Additional metadata fields

    Raises:
        HTTPException: 500 if API request fails or data processing errors occur

    Example:
        Retrieve observations without storage::
        
            GET /observations
            
        Retrieve and store observations::
        
            GET /observations?store_in_db=true

    Note:
        This endpoint retrieves all available recent observations without date filtering.
        For date-specific queries, use the /observations/from endpoint.
    """
    # Parse store_in_db parameter manually to handle various input formats
    store_in_db_bool = str(store_in_db).lower() in ('true', '1', 'yes', 'on')
    
    try:
        pages = await get_pages(logger=logger)
        observations = get_observations(pages)
        logger.info(f"Fetched {len(observations)} observations (no date filter)")

        if store_in_db_bool:
            db = get_database()
            db[INAT_COLLECTION].insert_many(copy.deepcopy(observations))  # Prevent _id mutation
            logger.info(f"Stored {len(observations)} observations in MongoDB")

        return observations
    except Exception as e:
        logger.error(f"Failed to fetch observations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch observations")


@router.get("/observations/from", response_model=List[Dict])
async def read_observations_from_date(
    year: int = Query(..., description="Year of the observation date"),
    month: int = Query(..., description="Month of the observation date"),
    day: int = Query(..., description="Day of the observation date"),
    store_in_db: str = Query("false", description="Store fetched observations in MongoDB")
):
    """
    Retrieve species observations from iNaturalist starting from a specific date.

    Fetches iNaturalist observation data from the specified start date onwards,
    filtered for the configured geographic region and taxonomic group.
    Useful for temporal analysis and building historical datasets.

    Args:
        year (int): Year component of the start date for observation filtering
        month (int): Month component of the start date (1-12)
        day (int): Day component of the start date (1-31)
        store_in_db (str, optional): Whether to store fetched observations in database.
            Accepts "true"/"false", "1"/"0", "yes"/"no", "on"/"off". 
            Defaults to "false".

    Returns:
        List[Dict]: List of standardized observation records from the specified date onwards

    Raises:
        HTTPException: 
            - 400 if date parameters are invalid or future date provided
            - 500 if API request fails or data processing errors occur

    Example:
        Get observations from a specific date::
        
            GET /observations/from?year=2024&month=1&day=15
            
        Get and store observations::
        
            GET /observations/from?year=2024&month=1&day=15&store_in_db=true

    Note:
        Start date must be in the past. Future dates will return a 400 error.
    """
    # Parse store_in_db parameter manually to handle various input formats
    store_in_db_bool = str(store_in_db).lower() in ('true', '1', 'yes', 'on')
    
    try:
        start_date = datetime.datetime(year, month, day)
    except ValueError as e:
        logger.warning(f"Invalid date input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    today = datetime.datetime.today()
    if start_date >= today:
        raise HTTPException(status_code=400, detail="Start date must be in the past")

    try:
        pages = await get_pages(start_date, logger=logger)
        observations = get_observations(pages)
        logger.info(f"Fetched {len(observations)} observations from {start_date.date()}")

        if store_in_db_bool:
            db = get_database()
            db[INAT_COLLECTION].insert_many(copy.deepcopy(observations))  # Prevent _id mutation
            logger.info(f"Stored {len(observations)} observations in MongoDB")

        return observations
    except Exception as e:
        logger.error(f"Failed to fetch observations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch observations")




@router.get("/observations/db", response_model=List[Dict])
def get_observations_from_db(limit: int = Query(100, description="Number of records to fetch")):
    """
    Retrieve stored iNaturalist observations from MongoDB.
    
    Fetches observation records that have been previously stored in the local
    MongoDB database. Useful for offline analysis and reducing API calls.

    Args:
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        List[Dict]: List of stored observation records without MongoDB object IDs

    Example:
        Get default number of stored observations::
        
            GET /observations/db
            
        Get specific number of observations::
        
            GET /observations/db?limit=50

    Note:
        Returns empty list if no observations have been stored in the database.
    """
    db = get_database()
    records = list(db[INAT_COLLECTION].find({}, {"_id": 0}).limit(limit))
    return records



@router.get("/observations/{observation_id}", response_model=Dict)
def read_observation(observation_id: int = Path(..., description="iNaturalist observation ID to retrieve")):
    """
    Fetch a single iNaturalist observation by its unique ID.
    
    Retrieves detailed information for a specific observation using the 
    iNaturalist API. Useful for getting complete details about a particular
    species sighting.

    Args:
        observation_id (int): The unique iNaturalist observation identifier

    Returns:
        Dict: Complete observation record with all available metadata

    Raises:
        HTTPException: 404 if observation with the specified ID is not found

    Example:
        Get a specific observation::
        
            GET /observations/12345

    Note:
        This endpoint fetches data directly from iNaturalist API, not local storage.
    """
    observation = get_observation_by_id(observation_id, logger=logger)
    if observation is None:
        raise HTTPException(status_code=404, detail=f"Observation {observation_id} not found")
    return observation



@router.delete("/observations/db")
def delete_observations_from_db():
    """
    Delete all stored iNaturalist observations from MongoDB.
    
    Permanently removes all observation records from the local database.
    Use with caution as this operation cannot be undone.

    Returns:
        Dict: Response containing the number of deleted records

    Example:
        Clear all stored observations::
        
            DELETE /observations/db
            
        Response::
        
            {"deleted_count": 42}

    Warning:
        This operation permanently deletes all stored observation data.
        Consider backing up data before using this endpoint.
    """
    db = get_database()
    result = db[INAT_COLLECTION].delete_many({})
    return {"deleted_count": result.deleted_count}
