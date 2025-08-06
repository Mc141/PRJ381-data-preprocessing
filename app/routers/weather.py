from fastapi import APIRouter, Query
import pandas as pd
from app.services.nasa_fetcher import PowerAPI





router = APIRouter()





@router.get("/weather")
def get_weather(
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """
    Fetch weather data from NASA POWER API for given coordinates and date range.
    """

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    api = PowerAPI(
        start=start_date,
        end=end_date,
        long=longitude,
        lat=latitude,
        use_long_names=True,
    )

    return api.get_weather()
