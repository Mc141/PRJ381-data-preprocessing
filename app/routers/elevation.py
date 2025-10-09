
"""
Elevation API Router
===================

Endpoints for retrieving elevation data for specific coordinates, supporting batch extraction. Designed for integration with environmental enrichment and modeling workflows.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from app.services.elevation_extractor import ElevationExtractor

logger = logging.getLogger(__name__)
router = APIRouter()


class CoordinatePoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")


class CoordinateListRequest(BaseModel):
    coordinates: List[CoordinatePoint] = Field(..., description="List of coordinates to process.")


@router.post(
    "/elevation/extract-batch",
    summary="Extract Elevation Data Batch",
    description="Extract elevation data for multiple coordinates efficiently.",
    response_model=Dict[str, Any],
)
async def extract_elevation_batch(
    request: CoordinateListRequest
) -> Dict[str, Any]:
    """
    Extract elevation data for multiple coordinates efficiently.

    The request body should contain a list of coordinates in the following format:
        {
            "coordinates": [
                {"latitude": -33.925, "longitude": 18.424},
                {"latitude": -33.895, "longitude": 18.505}
            ]
        }

    Returns:
        Dict[str, Any]: Batch of elevation data results and request metadata.
    """
    try:
        coordinates = [(point.latitude, point.longitude) for point in request.coordinates]
        logger.info(f"Extracting elevation data for {len(coordinates)} coordinates")
        
        extractor = ElevationExtractor()
        async with extractor:
            results = await extractor.extract_elevation_batch(coordinates)
            
            return {
                "request_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": "/api/v1/elevation/extract-batch",
                    "num_coordinates": len(coordinates)
                },
                "results": results
            }
    
    except Exception as e:
        logger.error(f"Error extracting elevation data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract elevation data: {str(e)}")
