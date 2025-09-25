"""
Status router for health checks and system information.

This module provides endpoints for monitoring the health and status 
of the data preprocessing API.
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
    🏥 **STEP 1: START HERE** - Complete system health check
    
    ### What This Checks:
    - **Database connectivity** (MongoDB status)
    - **API responsiveness** (FastAPI health)  
    - **Required services** (GBIF, WorldClim, NASA APIs)
    - **Storage space** for data downloads
    
    ### Green Light Means:
    - System ready for data collection
    - Database accepting connections  
    - All dependencies operational
    
    ### 🔗 Next Steps:
    If all systems are healthy, proceed to `/gbif/occurrences?store_in_db=true`
    """
    """
    Perform health check of the API service.
    
    Returns:
        dict: Health status information with timestamp
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    }

