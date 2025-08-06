from typing import List, Union, Optional
from pathlib import Path
from datetime import date, datetime
import requests
import pandas as pd
import os



class PowerAPI:
    """
    Query the NASA Power API.
    Check https://power.larc.nasa.gov/ for documentation
    Attributes
    ----------
    url : str
        Base URL
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?"

    def __init__(self,
                 start: Union[date, datetime, pd.Timestamp],
                 end: Union[date, datetime, pd.Timestamp],
                 long: float, lat: float,
                 use_long_names: bool = False,
                 parameter: Optional[List[str]] = None):
        """
        Parameters
        ----------
        start: Union[date, datetime, pd.Timestamp]
        end: Union[date, datetime, pd.Timestamp]
        long: float
            Longitude as float
        lat: float
            Latitude as float
        use_long_names: bool
            NASA provides both identifier and human-readable names for the fields. If set to True this will parse
            the data with the latter
        parameter: Optional[List[str]]
            List with the parameters to query.
            Default is ['T2M_RANGE', 'TS', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M', 'QV2M', 'RH2M',
                        'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
                        'WS50M_MIN', 'WS50M_RANGE']
        """
        self.start = start
        self.end = end
        self.long = long
        self.lat = lat
        self.use_long_names = use_long_names
        if parameter is None:
            self.parameter = [
        "T2M",                # Mean Air Temperature at 2 meters
        "T2M_MAX",            # Maximum Daily Air Temperature
        "T2M_MIN",            # Minimum Daily Air Temperature
        "PRECTOT",            # Precipitation (mm/day)
        "RH2M",               # Relative Humidity at 2m
        "WS2M",               # Wind Speed at 2 meters
        "ALLSKY_SFC_SW_DWN",  # Total Solar Radiation
        "CLRSKY_SFC_SW_DWN",  # Clear Sky Radiation
        "TQV",                # Total Precipitable Water Vapor
        "TS"                  # Surface Temperature
    ]


        self.request = self._build_request()

    def _build_request(self) -> str:
        """
        Build the request
        Returns
        -------
        str
            Full request including parameter
        """
        r = self.url
        r += f"parameters={(',').join(self.parameter)}"
        r += '&community=RE'
        r += f"&longitude={self.long}"
        r += f"&latitude={self.lat}"
        r += f"&start={self.start.strftime('%Y%m%d')}"
        r += f"&end={self.end.strftime('%Y%m%d')}"
        r += '&format=JSON'

        return r

    def get_weather(self):
        """
        Main method to query the weather data
        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with DateTimeIndex
        """

        response = requests.get(self.request)

        assert response.status_code == 200

        data_json = response.json()

        
        # Extract metadata
        longitude, latitude, elevation = data_json.get("geometry", {}).get("coordinates", [None, None, None])
        parameters_meta = data_json.get("parameters", {})
        raw_params = data_json.get("properties", {}).get("parameter", {})

        if not raw_params:
            raise ValueError("No weather data returned from NASA POWER API.")

        # Flatten into one record per date
        records = []
        dates = list(next(iter(raw_params.values())).keys())  # Get all available dates

        for date in dates:
            record = {
                "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation
            }
            for param_code, values in raw_params.items():
                record[param_code] = values.get(date, None)
            records.append(record)

        # Final API-friendly JSON
        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation
            },
            "parameters_meta": parameters_meta,
            "data": records
        }

