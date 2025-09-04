"""
Elevation API Router
===================

API endpoints for retrieving elevation data for specific coordinates.
Provides endpoints for single point and batch extraction.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime

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

@router.get("/elevation/extract",
           summary="Extract Elevation Data",
           description="Extract real elevation data for specific coordinates using SRTM data",
           response_model=Dict[str, Any])
async def extract_elevation_single(
    latitude: float = Query(..., description="Latitude in decimal degrees", ge=-90, le=90),
    longitude: float = Query(..., description="Longitude in decimal degrees", ge=-180, le=180)
) -> Dict[str, Any]:
    """
    Extract real elevation data for a single coordinate using SRTM data.
    
    This endpoint will:
    - Extract accurate elevation data from SRTM digital elevation models
    - Return elevation in meters above sea level
    - Include data source information for verification
    """
    try:
        logger.info(f"Extracting elevation data for {latitude:.3f}, {longitude:.3f}")
        
        # Get elevation extractor and use it as an async context manager
        extractor = get_elevation_extractor()
        
        # Use async context manager properly
        async with extractor:
            # Extract elevation data
            result = await extractor.extract_elevation(latitude, longitude)
            
            # Add API metadata
            result["api_request"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": "/api/v1/elevation/extract",
                "parameters": {"latitude": latitude, "longitude": longitude}
            }
            
            return result
    
    except Exception as e:
        logger.error(f"Error extracting elevation data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract elevation data: {str(e)}")

@router.post("/elevation/extract-batch",
            summary="Extract Elevation Data Batch",
            description="Extract real elevation data for multiple coordinates efficiently",
            response_model=Dict[str, Any])
async def extract_elevation_batch(
    request: CoordinateListRequest
) -> Dict[str, Any]:
    """
    Extract real elevation data for multiple coordinates efficiently.
    
    Request body should contain a list of coordinates:
    ```json
    {
        "coordinates": [
            {"latitude": -33.925, "longitude": 18.424},
            {"latitude": -33.895, "longitude": 18.505}
        ]
    }
    ```
    
    Returns a batch of elevation data results.
    """
    try:
        coordinates = [(point.latitude, point.longitude) for point in request.coordinates]
        logger.info(f"Extracting batch elevation data for {len(coordinates)} coordinates")
        
        # Get elevation extractor
        extractor = get_elevation_extractor()
        
        # Use async context manager properly
        async with extractor:
            # Extract elevation data in batch
            results = await extractor.extract_elevation_batch(coordinates)
            
            # Add API metadata
            response = {
                "request_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": "/api/v1/elevation/extract-batch",
                    "num_coordinates": len(coordinates)
                },
                "results": results
            }
            
            return response
    
    except Exception as e:
        logger.error(f"Error extracting batch elevation data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract batch elevation data: {str(e)}")
