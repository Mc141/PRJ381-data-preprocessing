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
async def read_observations(store_in_db: bool = Query(False, description="Store fetched observations in MongoDB")):
    pages = await get_pages(logger=logger)
    observations = get_observations(pages)
    logger.info(f"Fetched {len(observations)} observations (no date filter)")

    if store_in_db:
        db = get_database()
        db[INAT_COLLECTION].insert_many(copy.deepcopy(observations))  # Prevent _id mutation
        logger.info(f"Stored {len(observations)} observations in MongoDB")

    return observations


@router.get("/observations/from", response_model=List[Dict])
async def read_observations_from_date(
    year: int = Query(..., description="Year of the observation date"),
    month: int = Query(..., description="Month of the observation date"),
    day: int = Query(..., description="Day of the observation date"),
    store_in_db: bool = Query(False, description="Store fetched observations in MongoDB")
):
    try:
        start_date = datetime.datetime(year, month, day)
    except ValueError as e:
        logger.warning(f"Invalid date input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    today = datetime.datetime.today()
    if start_date >= today:
        raise HTTPException(status_code=400, detail="Start date must be in the past")

    pages = await get_pages(start_date, logger=logger)
    observations = get_observations(pages)
    logger.info(f"Fetched {len(observations)} observations from {start_date.date()}")

    if store_in_db:
        db = get_database()
        db[INAT_COLLECTION].insert_many(copy.deepcopy(observations))  # Prevent _id mutation
        logger.info(f"Stored {len(observations)} observations in MongoDB")

    return observations




@router.get("/observations/db", response_model=List[Dict])
def get_observations_from_db(limit: int = Query(100, description="Number of records to fetch")):
    """
    Retrieve stored iNaturalist observations from MongoDB.
    """
    db = get_database()
    records = list(db[INAT_COLLECTION].find({}, {"_id": 0}).limit(limit))
    return records



@router.get("/observations/{observation_id}", response_model=Dict)
def read_observation(observation_id: int):
    """
    Fetch a single iNaturalist observation by its ID.
    """
    observation = get_observation_by_id(observation_id, logger=logger)
    if observation is None:
        raise HTTPException(status_code=404, detail=f"Observation {observation_id} not found")
    return observation



@router.delete("/observations/db")
def delete_observations_from_db():
    """
    Delete all stored iNaturalist observations from MongoDB.
    """
    db = get_database()
    result = db[INAT_COLLECTION].delete_many({})
    return {"deleted_count": result.deleted_count}
