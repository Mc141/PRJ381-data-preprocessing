import logging
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from app.services.inat_fetcher import get_pages, get_observations
from app.services.nasa_fetcher import PowerAPI
from app.services.database import get_database

import pandas as pd
import io
import asyncio

router = APIRouter()
logger = logging.getLogger("datasets")


@router.get("/datasets/merge")
async def merge_datasets(
    start_year: int = Query(...),
    start_month: int = Query(...),
    start_day: int = Query(...),
    end_year: int = Query(...),
    end_month: int = Query(...),
    end_day: int = Query(...),
):
    """
    Fetch iNaturalist + NASA POWER data concurrently, store them in MongoDB, return merged dataset.
    """
    try:
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]

    logger.info(f"Fetching iNaturalist observations from {start_date} to {end_date}")
    pages = await get_pages(start_date, logger=logger)
    observations = get_observations(pages)

    if not observations:
        raise HTTPException(status_code=404, detail="No observations found")

    async def fetch_weather_for_observation(obs):
        if not obs.get("latitude") or not obs.get("longitude"):
            return None
        obs_date = pd.to_datetime(obs["time_observed_at"]).date()
        nasa_api = PowerAPI(
            start=obs_date,
            end=obs_date,
            lat=obs["latitude"],
            long=obs["longitude"]
        )
        weather_data = nasa_api.get_weather()
        for record in weather_data["data"]:
            record["inat_id"] = obs["id"]
            weather_collection.update_one(
                {"inat_id": obs["id"], "date": record["date"]},
                {"$set": record},
                upsert=True
            )
        return {"observation": obs, "weather": weather_data["data"]}

    merged_dataset = await asyncio.gather(
        *[fetch_weather_for_observation(obs) for obs in observations]
    )

    # Store iNat data after weather fetch
    for obs in observations:
        inat_collection.update_one({"id": obs["id"]}, {"$set": obs}, upsert=True)

    # Remove None results and _id fields
    merged_dataset = [m for m in merged_dataset if m is not None]
    for entry in merged_dataset:
        entry["observation"].pop("_id", None)
        for w in entry["weather"]:
            w.pop("_id", None)

    return {"count": len(merged_dataset), "dataset": merged_dataset}


@router.post("/datasets/refresh-weather")
async def refresh_weather():
    """
    Refresh NASA POWER weather data for all stored iNaturalist observations concurrently.
    """
    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]

    observations = list(inat_collection.find({}))
    if not observations:
        raise HTTPException(status_code=404, detail="No observations found in DB")

    async def refresh_weather_for_observation(obs):
        if not obs.get("latitude") or not obs.get("longitude"):
            return False
        obs_date = pd.to_datetime(obs["time_observed_at"]).date()
        nasa_api = PowerAPI(
            start=obs_date,
            end=obs_date,
            lat=obs["latitude"],
            long=obs["longitude"]
        )
        weather_data = nasa_api.get_weather()
        for record in weather_data["data"]:
            record["inat_id"] = obs["id"]
            weather_collection.update_one(
                {"inat_id": obs["id"], "date": record["date"]},
                {"$set": record},
                upsert=True
            )
        return True

    results = await asyncio.gather(
        *[refresh_weather_for_observation(obs) for obs in observations]
    )

    return {"updated_weather_records": sum(results)}


@router.get("/datasets/export")
def export_dataset():
    """
    Export merged iNat + NASA weather dataset as CSV.
    """
    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]

    observations = list(inat_collection.find({}))
    weather = list(weather_collection.find({}))

    if not observations or not weather:
        raise HTTPException(status_code=404, detail="No dataset to export")

    # Strip MongoDB _id to avoid ObjectId issues
    for obs in observations:
        obs.pop("_id", None)
    for w in weather:
        w.pop("_id", None)

    df_inat = pd.DataFrame(observations)
    df_weather = pd.DataFrame(weather)

    merged_df = pd.merge(df_inat, df_weather, left_on="id", right_on="inat_id", how="left")

    stream = io.StringIO()
    merged_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"}
    )
