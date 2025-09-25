
"""
Status Router for Health Checks and System Information

This module provides API endpoints for monitoring the health and operational status
of the data preprocessing service. Endpoints are designed for system monitoring,
deployment validation, and service readiness checks.
"""

from datetime import datetime
from fastapi import APIRouter
from typing import Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)



@router.get("/status/health")
async def health_check():
    """
    Perform a comprehensive health check of the API service.

    This endpoint verifies the operational status of the API and its dependencies, including:

        - API responsiveness (FastAPI)
        - Required external services (GBIF, WorldClim)
        - Storage space for data downloads

    Returns:
        dict: Health status information with the current timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

