import logging
import datetime
from fastapi import APIRouter, Query, Path, HTTPException
from app.services.inat_fetcher import get_pages, get_observations, get_observation_by_id

# Setup logger
logger = logging.getLogger("inat")
logger.setLevel(logging.INFO)

# Add stream handler if no handlers are already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter()


# Gets all observations from Inat
@router.get("/observations")
async def read_observations():
    pages = await get_pages(logger=logger)
    observations = get_observations(pages)
    logger.info(f"Fetched {len(observations)} observations (no date filter)")
    return observations


# Gets all observations from Inat from a specified date
@router.get("/observations/from-date")
async def read_observations_from_date(
    year: int = Query(..., description="Year of the observation date"),
    month: int = Query(..., description="Month of the observation date"),
    day: int = Query(..., description="Day of the observation date")
):
    try:
        start_date = datetime.datetime(year, month, day)
    except ValueError as e:
        logger.warning(f"Invalid date input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    today = datetime.datetime.today()

    if start_date >= today:
        logger.warning(f"Provided start_date {start_date.date()} is in the future.")
        raise HTTPException(
            status_code=400,
            detail=f"start_date must be less than the current date: {today.date()}"
        )

    pages = await get_pages(start_date, logger=logger)
    observations = get_observations(pages)
    logger.info(f"Fetched {len(observations)} observations from {start_date.date()}")
    return observations


# Gets a single observation based on ID
@router.get("/observations/{observation_id}")
def read_observation(observation_id: int = Path(..., description="ID of the specified observation")):
    observation = get_observation_by_id(observation_id, logger=logger)
    if observation is None:
        logger.warning(f"Observation ID {observation_id} not found or invalid.")
        raise HTTPException(status_code=404, detail=f"Observation with ID {observation_id} not found.")
    logger.info(f"Fetched observation ID {observation_id}")
    return observation



