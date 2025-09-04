"""
Environmental Data Router
========================

API endpoints for retrieving combined environmental data (climate + elevation)
for specific coordinates. Provides a convenient way to get all environmental 
data needed for the model in a single API call.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
import asyncio

from app.services.worldclim_extractor import get_worldclim_extractor
from app.services.elevation_extractor import get_elevation_extractor

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

@router.get("/environmental/extract",
           summary="Extract Combined Environmental Data",
           description="Extract both climate and elevation data for specific coordinates in a single call",
           response_model=Dict[str, Any])
async def extract_environmental_single(
    latitude: float = Query(..., description="Latitude in decimal degrees", ge=-90, le=90),
    longitude: float = Query(..., description="Longitude in decimal degrees", ge=-180, le=180),
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract (e.g., bio1, bio12)")
) -> Dict[str, Any]:
    """
    Extract combined environmental data (climate + elevation) for a single coordinate.
    
    This endpoint will:
    - Extract accurate climate data from WorldClim bioclimatic variables
    - Extract elevation data from SRTM digital elevation models
    - Return all environmental data needed for the model in a single call
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
        logger.info(f"Extracting environmental data for {latitude:.3f}, {longitude:.3f}")
        
        # Get extractors
        worldclim_extractor = get_worldclim_extractor()
        elevation_extractor = get_elevation_extractor()
        
        # Default variables if none specified
        if variables is None:
            variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        # Make sure we're passing a list of strings to the extractor
        elif isinstance(variables, str):
            variables = [var.strip() for var in variables.split(',')]
        
        # Use async context managers properly for both extractors
        async with worldclim_extractor, elevation_extractor:
            # Extract data concurrently
            climate_task = worldclim_extractor.extract_climate_data(latitude, longitude, variables)
            elevation_task = elevation_extractor.extract_elevation(latitude, longitude)
            
            # Wait for both tasks to complete
            climate_data, elevation_data = await asyncio.gather(climate_task, elevation_task)
        
        # Combine results
        result = {**climate_data}
        if elevation_data and "elevation" in elevation_data:
            result["elevation"] = elevation_data["elevation"]
            
        # Add API metadata
        result["api_request"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/api/v1/environmental/extract",
            "parameters": {"latitude": latitude, "longitude": longitude, "variables": variables}
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error extracting environmental data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract environmental data: {str(e)}")

@router.post("/environmental/extract-batch",
            summary="Extract Environmental Data Batch",
            description="Extract combined climate and elevation data for multiple coordinates efficiently",
            response_model=Dict[str, Any])
async def extract_environmental_batch(
    request: CoordinateListRequest,
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract")
) -> Dict[str, Any]:
    """
    Extract combined environmental data (climate + elevation) for multiple coordinates efficiently.
    
    Request body should contain a list of coordinates:
    ```json
    {
        "coordinates": [
            {"latitude": -33.925, "longitude": 18.424},
            {"latitude": -33.895, "longitude": 18.505}
        ]
    }
    ```
    
    Returns a batch of environmental data results (climate + elevation).
    """
    try:
        coordinates = [(point.latitude, point.longitude) for point in request.coordinates]
        logger.info(f"Extracting batch environmental data for {len(coordinates)} coordinates")
        
        # Get extractors
        worldclim_extractor = get_worldclim_extractor()
        elevation_extractor = get_elevation_extractor()
        
        # Default variables if none specified
        if variables is None:
            variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        # Make sure we're passing a list of strings to the extractor
        elif isinstance(variables, str):
            variables = [var.strip() for var in variables.split(',')]
        
        # Use async context managers properly for both extractors
        async with worldclim_extractor, elevation_extractor:
            # Extract data concurrently
            climate_task = worldclim_extractor.extract_climate_batch(coordinates, variables)
            elevation_task = elevation_extractor.extract_elevation_batch(coordinates)
            
            # Wait for both tasks to complete
            climate_results, elevation_results = await asyncio.gather(climate_task, elevation_task)
        
        # Combine results
        results = []
        for i, (climate, elevation) in enumerate(zip(climate_results, elevation_results)):
            combined = {**climate}
            if elevation and "elevation" in elevation:
                combined["elevation"] = elevation["elevation"]
            results.append(combined)
        
        # Add API metadata
        response = {
            "request_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "/api/v1/environmental/extract-batch",
                "num_coordinates": len(coordinates),
                "variables": variables
            },
            "results": results
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error extracting batch environmental data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract batch environmental data: {str(e)}")
