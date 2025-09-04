"""
Weather API Router
=================

API endpoints for retrieving weather data for specific coordinates.
Provides endpoints for single point and batch extraction of WorldClim bioclimatic variables.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime

from app.services.worldclim_extractor import get_worldclim_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class CoordinatePoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

class CoordinateListRequest(BaseModel):
    coordinates: List[CoordinatePoint] = Field(..., description="List of coordinates to process")

@router.get("/weather/extract",
           summary="Extract Weather Data",
           description="Extract real climate data for specific coordinates using WorldClim bioclimatic variables",
           response_model=Dict[str, Any])
async def extract_weather_single(
    latitude: float = Query(..., description="Latitude in decimal degrees", ge=-90, le=90),
    longitude: float = Query(..., description="Longitude in decimal degrees", ge=-180, le=180),
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract (e.g., bio1, bio12)")
) -> Dict[str, Any]:
    """
    Extract real weather data for a single coordinate using WorldClim bioclimatic variables.
    
    This endpoint will:
    - Extract accurate climate data from WorldClim bioclimatic variables
    - Return temperature, precipitation and seasonality data
    - Include data source information for verification
    
    Available climate variables:
    - bio1: Annual Mean Temperature (°C)
    - bio4: Temperature Seasonality
    - bio5: Max Temperature of Warmest Month (°C)
    - bio6: Min Temperature of Coldest Month (°C)  
    - bio12: Annual Precipitation (mm)
    - bio13: Precipitation of Wettest Month (mm)
    - bio14: Precipitation of Driest Month (mm)
    - bio15: Precipitation Seasonality
    """
    try:
        logger.info(f"Extracting weather data for {latitude:.3f}, {longitude:.3f}")
        
        # Get worldclim extractor
        extractor = get_worldclim_extractor()
        
        # Default variables if none specified
        if variables is None:
            variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        # Make sure we're passing a list of strings to the extractor
        elif isinstance(variables, str):
            variables = [var.strip() for var in variables.split(',')]
        
        # Use async context manager properly
        async with extractor:
            # Extract climate data
            result = await extractor.extract_climate_data(latitude, longitude, variables)
            
            # Add API metadata
            result["api_request"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "/api/v1/weather/extract",
                "parameters": {"latitude": latitude, "longitude": longitude, "variables": variables}
            }
            
            return result
    
    except Exception as e:
        logger.error(f"Error extracting weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract weather data: {str(e)}")

@router.post("/weather/extract-batch",
            summary="Extract Weather Data Batch",
            description="Extract real climate data for multiple coordinates efficiently",
            response_model=Dict[str, Any])
async def extract_weather_batch(
    request: CoordinateListRequest,
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract")
) -> Dict[str, Any]:
    """
    Extract real weather data for multiple coordinates efficiently.
    
    Request body should contain a list of coordinates:
    ```json
    {
        "coordinates": [
            {"latitude": -33.925, "longitude": 18.424},
            {"latitude": -33.895, "longitude": 18.505}
        ]
    }
    ```
    
    Returns a batch of climate data results.
    """
    try:
        coordinates = [(point.latitude, point.longitude) for point in request.coordinates]
        logger.info(f"Extracting batch weather data for {len(coordinates)} coordinates")
        
        # Get worldclim extractor
        extractor = get_worldclim_extractor()
        
        # Default variables if none specified
        if variables is None:
            variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        # Make sure we're passing a list of strings to the extractor
        elif isinstance(variables, str):
            variables = [var.strip() for var in variables.split(',')]
        
        # Use async context manager properly
        async with extractor:
            # Extract climate data in batch
            results = await extractor.extract_climate_batch(coordinates, variables)
            
            # Add API metadata
            response = {
                "request_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": "/api/v1/weather/extract-batch",
                    "num_coordinates": len(coordinates),
                    "variables": variables
                },
                "results": results
            }
            
            return response
    
    except Exception as e:
        logger.error(f"Error extracting batch weather data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract batch weather data: {str(e)}")
