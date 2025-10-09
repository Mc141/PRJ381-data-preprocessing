
"""
Environmental Data Router
========================

API endpoints for retrieving combined environmental data (climate and elevation)
for specific coordinates. Provides a unified interface to obtain all environmental
variables required for modeling in a single API call.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio
from app.services.worldclim_extractor import get_worldclim_extractor
from app.services.elevation_extractor import ElevationExtractor

logger = logging.getLogger(__name__)
router = APIRouter()


class CoordinatePoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")


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
        logger.info(f"Extracting environmental data for {len(coordinates)} coordinates")
        
        worldclim_extractor = get_worldclim_extractor()
        elevation_extractor = ElevationExtractor()
        
        if variables is None:
            variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        elif isinstance(variables, str):
            variables = [var.strip() for var in variables.split(',')]
        
        async with worldclim_extractor, elevation_extractor:
            climate_results, elevation_results = await asyncio.gather(
                worldclim_extractor.extract_climate_batch(coordinates, variables),
                elevation_extractor.extract_elevation_batch(coordinates)
            )
        
        results = []
        for climate, elevation in zip(climate_results, elevation_results):
            combined = {**climate}
            if elevation and "elevation" in elevation:
                combined["elevation"] = elevation["elevation"]
            results.append(combined)
        
        return {
            "request_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "/api/v1/environmental/extract-batch",
                "num_coordinates": len(coordinates),
                "variables": variables
            },
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error extracting environmental data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract environmental data: {str(e)}")
