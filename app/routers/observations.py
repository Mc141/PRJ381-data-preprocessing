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


@router.get("/observations")
async def read_observations():
    """
    Fetch all iNaturalist observations (no date filter).

    :returns: A list of observations.
    :rtype: List[Dict]
    :raises HTTPException: If the request fails or data cannot be fetched.
    """
    pages = await get_pages(logger=logger)
    observations = get_observations(pages)
    logger.info(f"Fetched {len(observations)} observations (no date filter)")
    return observations


@router.get("/observations/from")
async def read_observations_from_date(
    year: int = Query(..., description="Year of the observation date"),
    month: int = Query(..., description="Month of the observation date"),
    day: int = Query(..., description="Day of the observation date")
):
    """
    Fetch iNaturalist observations starting from a specific date.

    :param year: Year of the observation date.
    :type year: int
    :param month: Month of the observation date.
    :type month: int
    :param day: Day of the observation date.
    :type day: int

    :returns: A list of filtered observations.
    :rtype: List[Dict]
    :raises HTTPException: If the date is invalid or in the future.
    """
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


@router.get("/observations/{observation_id}")
def read_observation(observation_id: int = Path(..., description="ID of the specified observation")):
    """
    Fetch a single iNaturalist observation by its ID.

    :param observation_id: ID of the observation to fetch.
    :type observation_id: int

    :returns: A single observation dictionary.
    :rtype: Dict
    :raises HTTPException: If the observation is not found or invalid.
    """
    observation = get_observation_by_id(observation_id, logger=logger)
    if observation is None:
        logger.warning(f"Observation ID {observation_id} not found or invalid.")
        raise HTTPException(status_code=404, detail=f"Observation with ID {observation_id} not found.")
    logger.info(f"Fetched observation ID {observation_id}")
    return observation
