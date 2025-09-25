
"""
Environmental Data Router
========================

API endpoints for retrieving combined environmental data (climate and elevation)
for specific coordinates. Provides a unified interface to obtain all environmental
variables required for modeling in a single API call.
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
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        return v


class CoordinateListRequest(BaseModel):
    coordinates: List[CoordinatePoint] = Field(..., description="List of coordinates to process.")

@router.post(
    "/environmental/extract-batch",
    summary="Extract Environmental Data Batch",
    description="Extract combined climate and elevation data for multiple coordinates efficiently.",
    response_model=Dict[str, Any],
)
async def extract_environmental_batch(
    request: CoordinateListRequest,
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract")
) -> Dict[str, Any]:
    """
    Extract combined climate and elevation data for multiple coordinates.

    The request body should contain a list of coordinates as:
        {"coordinates": [{"latitude": -33.925, "longitude": 18.424}, ...]}
    Returns a dictionary with request metadata and a list of results for each coordinate.
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
