"""
PRJ381 Species Distribution Modeling API
=======================================

This FastAPI application provides a robust platform for creating machine learning datasets from global biodiversity observations, enriched with authentic environmental data for species distribution modeling and invasion risk assessment.

Current Workflow:
    1. Data Collection: Fetch global species occurrences from GBIF
    2. Environmental Enrichment: Add WorldClim climate & SRTM elevation data
    3. Dataset Creation: Build ML-ready datasets for transfer learning (direct CSV export)
    4. Model Training: Export data for XGBoost model training (external)
    5. Prediction & Mapping: Generate invasion risk predictions and visualizations (API endpoints)

API Organization:
    - Core Pipeline: Status → Environmental → Datasets → Predictions
    - Data Management: Status monitoring, data validation

Dependencies Flow:
    /status → /environmental → /datasets → /predictions

Author: MC141
Project: PRJ381 - Invasive Species Distribution Modeling
"""


# FastAPI application and router imports
from fastapi import FastAPI
from app.routers import datasets, status, predictions
# Environmental router is required for data extraction used by predictions and heatmaps
from app.routers import environmental




app = FastAPI(
    title="PRJ381 Species Distribution Modeling API",
    description="""
API for global species distribution modeling and transfer learning.

DATA INTEGRITY POLICY
---------------------
REAL DATA ONLY: No fake, dummy, or placeholder environmental values
Missing = NaN: When data unavailable, returns None/NaN (never fake values)
Transparent Sources: All data sources clearly labeled and trackable

QUICK START WORKFLOW
--------------------
Step 1: System Check
    GET /api/v1/status/health - Verify system health and service readiness
Step 2: Data Collection
    Use external scripts or endpoints to fetch global species occurrences
Step 3: Environmental Setup
    Use API endpoints to extract real WorldClim climate and SRTM elevation data
Step 4: Dataset Creation
    Use /api/v1/datasets/generate-ml-ready-files to create ML-ready CSVs
Step 5: Model Training & Prediction
    Train models externally and use /api/v1/predictions/* endpoints for risk mapping

DATA SOURCES
------------
Species Data: GBIF Global Database
Climate Data: WorldClim v2.1
Elevation Data: SRTM 30m via Open-Topo-Data API
Weather Data: NASA POWER API

ML CAPABILITIES
---------------
Transfer Learning: Global training → Local validation
Environmental Modeling: Climate + elevation + weather variables
Risk Assessment: Invasion suitability mapping
    """,
    version="3.0.0-minimal",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Martinus Christoffel Wolmarans",
        "email": "mc141@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "1. System Status",
            "description": "START HERE - Health checks, system monitoring, and dependency verification",
        },
        {"name": "Datasets", "description": "Generate ML-ready datasets"},
        {"name": "Predictions", "description": "Train model and generate heatmaps"},
        {"name": "Environmental Data", "description": "Real environmental variable extraction used by heatmaps"},
    ]
)




# Include only the core API surface
app.include_router(status.router, prefix="/api/v1", tags=["1. System Status"])
app.include_router(environmental.router, prefix="/api/v1", tags=["Environmental Data"])
app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
