import logging
from datetime import datetime
from typing import List, Dict
import copy

from fastapi import APIRouter, Query, HTTPException
from app.services.nasa_fetcher import PowerAPI
from app.services.database import get_database

# Setup logger
logger = logging.getLogger("weather")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter()

# MongoDB collection name
WEATHER_COLLECTION = "weather_data"


@router.get("/weather", response_model=Dict)
def get_weather_data(
    latitude: float = Query(..., description="Latitude of the point"),
    longitude: float = Query(..., description="Longitude of the point"),
    start_year: int = Query(..., description="Start year"),
    start_month: int = Query(..., description="Start month"),
    start_day: int = Query(..., description="Start day"),
    end_year: int = Query(..., description="End year"),
    end_month: int = Query(..., description="End month"),
    end_day: int = Query(..., description="End day"),
    store_in_db: bool = Query(False, description="Store fetched weather data in MongoDB")
):
    """
    Fetch weather data from NASA POWER API for given coordinates and date range.

    Optionally stores the fetched data in MongoDB.
    """
    try:
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
    except ValueError as e:
        logger.warning(f"Invalid date input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    try:
        power_api = PowerAPI(
            start=start_date,
            end=end_date,
            lat=latitude,
            long=longitude
        )
        weather_data = power_api.get_weather()
        logger.info(f"Fetched weather data for ({latitude}, {longitude}) from {start_date.date()} to {end_date.date()}")



        if store_in_db:
            db = get_database()
            db[WEATHER_COLLECTION].insert_many(copy.deepcopy(weather_data["data"]))  # Prevent mutation
            logger.info(f"Stored {len(weather_data['data'])} weather records in MongoDB")



        return weather_data
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve weather data: {e}")





@router.get("/weather/db", response_model=List[Dict])
def get_weather_from_db(limit: int = Query(100, description="Number of records to fetch")):
    """
    Retrieve stored weather data from MongoDB.
    """
    db = get_database()
    records = list(db[WEATHER_COLLECTION].find().limit(limit))
    for r in records:
        r.pop("_id", None)  # Prevent Mutation
    return records


@router.delete("/weather/db")
def delete_weather_from_db():
    """
    Delete all stored weather data from MongoDB.
    """
    db = get_database()
    result = db[WEATHER_COLLECTION].delete_many({})
    logger.info(f"Deleted {result.deleted_count} weather records from MongoDB")
    return {"deleted_count": result.deleted_count}



@router.get("/weather/recent", response_model=List[Dict])
def get_recent_weather_from_db(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    days: int = Query(30, description="Number of most recent days")
):
    """
    Retrieve recent weather records for a given location from MongoDB.
    """
    db = get_database()
    records = list(db[WEATHER_COLLECTION].find(
        {"latitude": latitude, "longitude": longitude}
    ).sort("date", -1).limit(days))
    for r in records:
        r.pop("_id", None) # Prevent Mutation
    return records