"""
WorldClim API Router
===================

API endpoints for managing real WorldClim data operations.
Provides endpoints for downloading, status checking, and testing
real climate data extraction.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import logging
from datetime import datetime

from app.services.worldclim_extractor import get_worldclim_extractor
from app.services.worldclim_extractor import extract_climate_data as extract_climate_service
from app.services.worldclim_extractor import extract_climate_batch as extract_climate_batch_service

# Import elevation service
try:
    from app.services.elevation_extractor import get_elevation_extractor, extract_elevation_data
    ELEVATION_AVAILABLE = True
except ImportError:
    ELEVATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/worldclim/status",
           summary="Get WorldClim Service Status",
           description="Get current status of the WorldClim real data service",
           response_model=Dict[str, Any])
async def get_worldclim_status() -> Dict[str, Any]:
    """
    Get the current status of the WorldClim service including data availability.
    
    Returns information about:
    - Service configuration
    - Local data file status
    - Download readiness
    - Available variables
    """
    try:
        extractor = get_worldclim_extractor()
        status = extractor.get_service_status()
        
        # Add elevation service status if available
        if ELEVATION_AVAILABLE:
            try:
                elevation_extractor = get_elevation_extractor()
                elevation_status = elevation_extractor.get_service_status()
                status["elevation_service"] = elevation_status
            except Exception as e:
                logger.warning(f"Error getting elevation status: {e}")
                status["elevation_service"] = {"status": "error", "error": str(e)}
        else:
            status["elevation_service"] = {"status": "not_available", "message": "Elevation service not imported"}
        
        # Add API-specific information
        status.update({
            "api_version": "1.0", 
            "endpoint_base": "/api/v1/worldclim",
            "timestamp": datetime.utcnow().isoformat(),
            "data_size_estimate": "~900MB for full resolution data"
        })
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting WorldClim status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")

@router.post("/worldclim/ensure-data",
            summary="Download Real WorldClim Data",
            description="""
**STEP 3 of ML Pipeline** - Download authentic climate data

### Prerequisites:
1. **Storage Space**: Ensure ~900MB free disk space
2. **Network Access**: Stable internet connection for download

### What This Downloads:
- **WorldClim v2.1 Bioclimate Variables** (19 raster files)
- **10-minute resolution** (~18.5km spatial resolution)
- **Scientific-grade data** used by researchers worldwide
- **Real climate data** - NO placeholders or fake values!

### Data Details:
- **Source**: WorldClim v2.1 (Fick & Hijmans, 2017)
- **Variables**: bio1-bio19 (temperature, precipitation, seasonality)
- **Coverage**: Global terrestrial areas
- **Format**: GeoTIFF raster files for precise coordinate extraction

### Performance:
- **Download size**: ~900MB compressed, ~52MB extracted
- **Download time**: 2-5 minutes (depending on connection)
- **Storage location**: `data/worldclim/wc2.1_10m_bio/`

### 🔗 Next Steps:
After download, use `/datasets/merge-global` to enrich species data with real climate variables.

### Quality Guarantee:
This downloads the SAME data used in thousands of published scientific studies!
            """,
            response_model=Dict[str, Any])
async def ensure_worldclim_data(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Ensure WorldClim data is downloaded and available for use.
    
    This endpoint will:
    - Check if WorldClim data files exist locally
    - Download them if necessary (~900MB)
    - Prepare the service for real data extraction
    
    Returns status information about the operation.
    """
    try:
        extractor = get_worldclim_extractor()
        
        logger.info("Ensuring WorldClim data availability...")
        
        # Check current status
        initial_status = extractor.get_service_status()
        
        if initial_status.get("files_downloaded", False):
            return {
                "status": "already_available",
                "message": "WorldClim data is already downloaded and ready",
                "data_directory": initial_status["data_directory"],
                "files_count": initial_status.get("local_files_count", 0),
                "resolution": initial_status["resolution"]
            }
        
        # Attempt to download/prepare data
        result = await extractor.download_worldclim_data()
        
        if result["status"] in ["downloaded", "already_exists"]:
            final_status = extractor.get_service_status()
            return {
                "status": "success", 
                "message": result["message"],
                "data_directory": result["location"],
                "files_count": result["files_count"],
                "resolution": final_status["resolution"],
                "download_completed": True,
                "download_info": result
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to download or prepare WorldClim data",
                "error": result.get("error", "Download or extraction failed"),
                "data_directory": initial_status["data_directory"],
                "resolution": initial_status["resolution"]
            }
    
    except Exception as e:
        logger.error(f"Error ensuring WorldClim data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ensure data availability: {str(e)}")

@router.post("/worldclim/extract",
            summary="Extract Climate Data",
            description="Extract real WorldClim climate data for specific coordinates",
            response_model=Dict[str, Any])
async def extract_climate_data(
    latitude: float = Query(..., description="Latitude in decimal degrees", ge=-90, le=90),
    longitude: float = Query(..., description="Longitude in decimal degrees", ge=-180, le=180),
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract (e.g., bio1, bio12)")
) -> Dict[str, Any]:
    """
    Extract real WorldClim climate data for a single coordinate.
    
    This endpoint will:
    - Extract real bioclimate variables from WorldClim raster files
    - Return actual temperature and precipitation data
    - Include data source information for verification
    
    Climate variables available:
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
        logger.info(f"Extracting climate data for {latitude:.3f}, {longitude:.3f}")
        
        # Extract climate data
        result = await extract_climate_service(latitude, longitude, variables)
        
        # Add API metadata
        result["api_request"] = {
            "endpoint": "/worldclim/extract",
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "requested_variables": variables,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check data quality
        data_source = result.get("data_source", "unknown")
        if "real_data" in data_source:
            result["data_quality"] = "real_worldclim_data"
        elif "fallback" in data_source or "error" in data_source:
            result["data_quality"] = "fallback_nan_values"
        else:
            result["data_quality"] = "unknown"
        
        return result
    
    except Exception as e:
        logger.error(f"Error extracting climate data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract climate data: {str(e)}")

class CoordinateRequest(BaseModel):
    coordinates: List[Dict[str, float]]

@router.post("/worldclim/extract-batch",
            summary="Extract Climate Data Batch",
            description="Extract real WorldClim climate data for multiple coordinates",
            response_model=Dict[str, Any])
async def extract_climate_data_batch(
    request: CoordinateRequest,
    variables: Optional[List[str]] = Query(None, description="Climate variables to extract")
) -> Dict[str, Any]:
    """
    Extract real WorldClim climate data for multiple coordinates efficiently.
    
    Request body should contain a list of coordinates:
    ```json
    {
        "coordinates": [
            {"latitude": -33.925, "longitude": 18.424},
            {"latitude": 40.713, "longitude": -74.006}
        ]
    }
    ```
    
    Returns climate data for all coordinates with data quality indicators.
    """
    coordinates = request.coordinates
    
    if not coordinates:
        raise HTTPException(status_code=400, detail="No coordinates provided")
    
    try:
        # Convert coordinate format
        coord_tuples = []
        for coord in coordinates:
            if "latitude" not in coord or "longitude" not in coord:
                raise HTTPException(status_code=400, detail="Each coordinate must have 'latitude' and 'longitude'")
            coord_tuples.append((coord["latitude"], coord["longitude"]))
        
        logger.info(f"Batch extracting climate data for {len(coord_tuples)} coordinates")
        
        # Extract climate data
        results = await extract_climate_batch_service(coord_tuples, variables)
        
        # Analyze results
        real_data_count = sum(1 for r in results if "real_data" in r.get("data_source", ""))
        fallback_count = len(results) - real_data_count
        
        return {
            "request_info": {
                "endpoint": "/worldclim/extract-batch",
                "coordinates_count": len(coord_tuples),
                "requested_variables": variables,
                "timestamp": datetime.utcnow().isoformat()
            },
            "results": results,
            "summary": {
                "total_coordinates": len(results),
                "real_data_count": real_data_count,
                "fallback_count": fallback_count,
                "success_rate": f"{real_data_count/len(results)*100:.1f}%" if results else "0%"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch climate extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract batch climate data: {str(e)}")


@router.get("/worldclim/variables",
           summary="Get Available Variables",
           description="Get list of available WorldClim bioclimate variables",
           response_model=Dict[str, Any])
async def get_available_variables() -> Dict[str, Any]:
    """
    Get information about available WorldClim bioclimate variables.
    
    Returns descriptions and units for all available variables.
    """
    variables_info = {
        "bio1": {
            "name": "Annual Mean Temperature",
            "units": "°C",
            "description": "Mean temperature across all months"
        },
        "bio2": {
            "name": "Mean Diurnal Range",
            "units": "°C",
            "description": "Mean of monthly (max temp - min temp)"
        },
        "bio3": {
            "name": "Isothermality",
            "units": "ratio",
            "description": "BIO2/BIO7 × 100"
        },
        "bio4": {
            "name": "Temperature Seasonality",
            "units": "coefficient",
            "description": "Standard deviation × 100"
        },
        "bio5": {
            "name": "Max Temperature of Warmest Month",
            "units": "°C",
            "description": "Highest monthly maximum temperature"
        },
        "bio6": {
            "name": "Min Temperature of Coldest Month",
            "units": "°C",
            "description": "Lowest monthly minimum temperature"
        },
        "bio7": {
            "name": "Temperature Annual Range",
            "units": "°C",
            "description": "BIO5 - BIO6"
        },
        "bio8": {
            "name": "Mean Temperature of Wettest Quarter",
            "units": "°C",
            "description": "Average temperature during wettest 3 months"
        },
        "bio9": {
            "name": "Mean Temperature of Driest Quarter",
            "units": "°C",
            "description": "Average temperature during driest 3 months"
        },
        "bio10": {
            "name": "Mean Temperature of Warmest Quarter",
            "units": "°C",
            "description": "Average temperature during warmest 3 months"
        },
        "bio11": {
            "name": "Mean Temperature of Coldest Quarter",
            "units": "°C",
            "description": "Average temperature during coldest 3 months"
        },
        "bio12": {
            "name": "Annual Precipitation",
            "units": "mm",
            "description": "Total precipitation across all months"
        },
        "bio13": {
            "name": "Precipitation of Wettest Month",
            "units": "mm",
            "description": "Highest monthly precipitation"
        },
        "bio14": {
            "name": "Precipitation of Driest Month",
            "units": "mm",
            "description": "Lowest monthly precipitation"
        },
        "bio15": {
            "name": "Precipitation Seasonality",
            "units": "coefficient",
            "description": "Coefficient of variation"
        },
        "bio16": {
            "name": "Precipitation of Wettest Quarter",
            "units": "mm",
            "description": "Total precipitation during wettest 3 months"
        },
        "bio17": {
            "name": "Precipitation of Driest Quarter",
            "units": "mm",
            "description": "Total precipitation during driest 3 months"
        },
        "bio18": {
            "name": "Precipitation of Warmest Quarter",
            "units": "mm",
            "description": "Total precipitation during warmest 3 months"
        },
        "bio19": {
            "name": "Precipitation of Coldest Quarter",
            "units": "mm",
            "description": "Total precipitation during coldest 3 months"
        }
    }
    
    # Get extractor status to show which variables are commonly used
    extractor = get_worldclim_extractor()
    
    return {
        "variables": variables_info,
        "total_available": len(variables_info),
        "standard_set": ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"],
        "data_source": "WorldClim v2.1",
        "resolution_available": ["30s", "2.5m", "5m", "10m"],
        "current_resolution": extractor.resolution,
        "timestamp": datetime.utcnow().isoformat()
    }
