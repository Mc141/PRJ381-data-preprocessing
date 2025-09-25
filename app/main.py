"""
PRJ381 Species Distribution Modeling & Transfer Learning API
===========================================================

**DATA INTEGRITY POLICY**: 
- **REAL DATA ONLY** - No fake, dummy, or placeholder environmental values
- **Missing = NaN** - When data unavailable, returns None/NaN (never fake values)
- **Transparent Sources** - All data sources clearly labeled and trackable

A comprehensive FastAPI application for creating machine learning datasets from global 
biodiversity observations, enriched with real environmental data for species distribution 
modeling and invasion risk assessment.

MAIN WORKFLOW:
1. Data Collection: Fetch global species occurrences from GBIF
2. Environmental Enrichment: Add WorldClim climate & SRTM elevation data  
3. Dataset Creation: Build ML-ready datasets for transfer learning
4. Model Training: Export data for Random Forest/ML model training
5. Prediction & Mapping: Generate invasion risk predictions and visualizations

API ORGANIZATION:
- Core Pipeline: GBIF → Environmental → Datasets → ML Export
- Data Management: Status monitoring, data validation, cleanup
- Legacy/Utilities: Deprecated endpoints, development tools

DEPENDENCIES FLOW:
/status → /gbif → /worldclim → /datasets → /predictions

Author: MC141
Project: PRJ381 - Invasive Species Distribution Modeling
"""
from fastapi import FastAPI
from app.routers import datasets, status, predictions
# Keep environmental and elevation routers for data extraction used by predictions
from app.routers import environmental


app = FastAPI(
    title="PRJ381 Species Distribution Modeling API",
    description="""
## Global Species Distribution Modeling & Transfer Learning Platform

Create machine learning datasets from **real global biodiversity data** enriched with **authentic environmental variables** for accurate species distribution modeling and invasion risk assessment.

### DATA INTEGRITY POLICY

**REAL DATA ONLY** - No fake, dummy, or placeholder environmental values
**Missing = NaN** - When data unavailable, returns None/NaN (never fake values)
**Transparent Sources** - All data sources clearly labeled and trackable

### QUICK START WORKFLOW

#### **Step 1: System Check** 
`GET /api/v1/status/health` - Verify system health and database connectivity

#### **Step 2: Data Collection** 
`GET /api/v1/gbif/occurrences?store_in_db=true` - Fetch global species occurrences (~1,700+ records)

#### **Step 3: Environmental Setup**
`POST /api/v1/worldclim/ensure-data` - Download real WorldClim climate data (~900MB)

#### **Step 4: Dataset Creation**
`GET /api/v1/datasets/merge-global?include_nasa_weather=false` - Create enriched training dataset

#### **Step 5: ML Export**
`GET /api/v1/datasets?format=csv` - Export finalized datasets for training

#### **Step 6: Model Validation**
`GET /api/v1/datasets/climate-comparison` - Validate environmental data quality

---

### DEPENDENCY REQUIREMENTS

**For Training Dataset:**
1. Run `/gbif/occurrences` with `store_in_db=true`
2. Run `/worldclim/ensure-data` to download climate data  
3. Then run `/datasets/merge-global`

**For ML Export:**
1. Complete training dataset creation above
2. Use `/datasets/export-ml-ready` with desired format

**For Predictions:**
1. Train model with exported dataset
2. Use `/predictions/*` endpoints for risk mapping

---

### DATA SOURCES
- **Species Data**: GBIF Global Database (1.4+ billion records)
- **Climate Data**: WorldClim v2.1 (19 bioclimate variables)  
- **Elevation Data**: SRTM 30m via Open-Topo-Data API
- **Weather Data**: NASA POWER API (meteorological data)
- **Taxonomy**: GBIF Taxonomic Backbone

### ML CAPABILITIES
- **Transfer Learning**: Global training → Local validation
- **Environmental Modeling**: 19 climate + elevation + weather variables
- **Risk Assessment**: Invasion suitability mapping
- **Model Validation**: Cross-regional performance testing
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
        # CORE PIPELINE (Primary workflow)
        {
            "name": "1. System Status",
            "description": "**START HERE** - Health checks, system monitoring, and dependency verification",
        },
        {"name": "Datasets", "description": "Generate ML-ready datasets (no DB)"},
        {"name": "Predictions", "description": "Train model and generate heatmaps (no DB)"},
        {"name": "Environmental Data", "description": "Real environmental variable extraction used by heatmaps"},
    ]
)

# Include only the minimal, DB-free API surface
app.include_router(status.router, prefix="/api/v1", tags=["1. System Status"])
app.include_router(environmental.router, prefix="/api/v1", tags=["Environmental Data"])
app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
