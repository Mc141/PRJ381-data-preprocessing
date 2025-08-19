"""
Status router for health checks and system information.

This module provides endpoints for monitoring the health and status 
of the data preprocessing API.
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter()


@router.get("/status/health")
async def health_check():
    """
    Perform health check of the API service.
    
    Returns:
        dict: Health status information with timestamp
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status/service_info")
async def service_info():
    """
    Get information about the API service.
    
    Returns:
        dict: Service information including version and available endpoints
    """
    return {
        "service": "PRJ381 Data Preprocessing API",
        "version": "1.0.0",
        "endpoints": ["observations", "weather", "datasets", "status"]
    }
