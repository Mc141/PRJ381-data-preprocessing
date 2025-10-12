User Guide
==========

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.12 or higher
* Internet connection for Open-Topo-Data API access
* At least 500MB free disk space for WorldClim data
* WorldClim v2.1 GeoTIFF files (bio variables 1, 4-6, 12-15)

Setup Steps
~~~~~~~~~~~

1. **Clone the Repository**::

    git clone https://github.com/Mc141/PRJ381-data-preprocessing.git
    cd PRJ381-data-preprocessing

2. **Install Dependencies**::

    pip install -r requirements.txt

3. **Download WorldClim Data**::

    # Download from: https://worldclim.org/data/worldclim21.html
    # Place files in: data/worldclim/
    # Required: wc2.1_10m_bio_1.tif, bio_4.tif, bio_5.tif, bio_6.tif,
    #           bio_12.tif, bio_13.tif, bio_14.tif, bio_15.tif

4. **Run the Application**::

    uvicorn app.main:app --reload --port 8000

5. **Access Documentation**:
   
   * Swagger UI: http://localhost:8000/docs
   * ReDoc: http://localhost:8000/redoc

ML Pipeline Workflow
-------------------

The API follows a structured workflow for creating machine learning datasets:

Step 1: System Health Check
~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify system status and dependencies::

    GET /api/v1/status/health

Step 2: Collect Species Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch global species occurrence data from GBIF::

    GET /api/v1/gbif/occurrences?store_in_db=true&max_results=2000

This collects ~1,700+ global occurrence records for training.

Step 3: Download Environmental Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download WorldClim v2.1 climate data::

    POST /api/v1/worldclim/ensure-data

Downloads ~900MB of real bioclimate data to your local system.

Step 4: Generate ML-Ready Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate training and validation CSV files::

    POST /api/v1/datasets/generate-ml-ready
    Body: {"max_global": 2000, "max_local": 500}

Creates enriched datasets with 13 features (location, climate, temporal, elevation).

Step 5: Generate Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use trained XGBoost model for invasion risk mapping::

    GET /api/v1/predictions/heatmap?model_name=xgboost

Generates probability heatmap for invasion risk assessment.

Data sources
------------

This project uses WorldClim v2.1 climate data, SRTM elevation via Open-Topo-Data, and GBIF occurrence records. See the Architecture and Overview sections for details.

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

Check system status::

    GET /api/v1/status/health
    GET /api/v1/status/service-info

Extract environmental data::

    POST /api/v1/environmental/extract-batch
    Body: {"coordinates": [
        {"latitude": -33.925, "longitude": 18.424},
        {"latitude": -34.056, "longitude": 18.472}
    ]}

Generate datasets::

    POST /api/v1/datasets/generate-ml-ready
    Body: {"max_global": 1000, "max_local": 250, "verbose": true}

Generate prediction heatmap::

    GET /api/v1/datasets/export

Invasion Risk Heatmap (Interactive)
-----------------------------------

The project includes an interactive, grid-based invasion risk heatmap powered by the XGBoost model and real environmental data (WorldClim v2.1 + SRTM elevation).

How to Generate
~~~~~~~~~~~~~~~

From the project root, run::

    python -m models.xgboost.generate_heatmap_api --grid_size 20 --month 3

Options:

* ``--grid_size``: Number of grid points per dimension (higher = more detail)
* ``--month``: Month of year (1-12) for seasonality features
* ``--western_cape_extended`` or ``--specific_area`` with ``lat/lon`` bounds

Interactive UI Features
~~~~~~~~~~~~~~~~~~~~~~~

Hover Tooltip:

* Category (Very Low → Critical)
* Probability (%)
* Latitude / Longitude

Click Popup (per grid cell):

* Risk level and probability
* Assessment text explaining the risk category
* Exact coordinates
* 🗺️ "Navigate with Google Maps" button that opens directions to the cell location

Risk Hotspots:

* Top 5 highest-risk points marked with color-coded markers
* Popup includes rank, risk %, and coordinates

Information Panels:

* Top Center: Title with species, month, data sources (WorldClim + SRTM)
* Top Right: Risk Category Guide with management suggestions
* Bottom Left: Statistics (distribution by category, mean/max/min, median, std)
* Bottom Right: Data Sources & Methodology (provenance and disclaimer)

Legend:

* Continuous color scale (0% → 100%)
* Cyan/Green for low risk → Yellow/Orange → Red for high to critical risk

Layers:

* Invasion Risk Grid (default)
* Satellite base layer
* Risk Hotspots (Top 5)

Notes:

* Popups open on click; hover displays tooltips
* Environmental values are pulled from the listed sources; some grid cells may have missing data
* Use a larger ``--grid_size`` for publication-quality maps (longer runtime)

Advanced Usage
--------------

Batch Processing
~~~~~~~~~~~~~~~~

For large datasets, use the async processing capabilities::

    # Process multiple years of data
    GET /api/v1/datasets/merge?start_year=2020&start_month=1&start_day=1&end_year=2024&end_month=12&end_day=31&years_back=10

    GET /api/v1/predictions/heatmap?model_name=xgboost

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The application supports the following environment variables:

* ``LOG_LEVEL``: Logging level (default: INFO)
* ``API_TIMEOUT``: API request timeout in seconds (default: 30)
* ``WORLDCLIM_PATH``: Path to WorldClim GeoTIFF files (default: data/worldclim/)

Data Files Required
~~~~~~~~~~~~~~~~~~~

WorldClim v2.1 GeoTIFF files needed in ``data/worldclim/``:

* ``wc2.1_10m_bio_1.tif``: Annual mean temperature
* ``wc2.1_10m_bio_4.tif``: Temperature seasonality
* ``wc2.1_10m_bio_5.tif``: Max temperature of warmest month
* ``wc2.1_10m_bio_6.tif``: Min temperature of coldest month
* ``wc2.1_10m_bio_12.tif``: Annual precipitation
* ``wc2.1_10m_bio_13.tif``: Precipitation of wettest month
* ``wc2.1_10m_bio_14.tif``: Precipitation of driest month
* ``wc2.1_10m_bio_15.tif``: Precipitation seasonality

Error Handling
--------------

The API provides comprehensive error handling:

* **400 Bad Request**: Invalid parameters or coordinates
* **404 Not Found**: No data found or missing WorldClim files
* **500 Internal Server Error**: GeoTIFF extraction or API communication errors

Common Issues
~~~~~~~~~~~~~

**WorldClim File Missing**::

    # Check if files exist
    ls -lh data/worldclim/*.tif
    
    # Verify file permissions
    chmod 644 data/worldclim/*.tif

**API Timeout Issues**::

    # Open-Topo-Data has 1-second rate limit
    # Use batch processing with delays
    # Check internet connection

**Memory Issues**::

    # Process smaller batches
    # Increase system memory
    # Use streaming for large coordinate lists

Performance Tips
----------------

* Use batch processing for multiple coordinates (100 at a time)
* Respect Open-Topo-Data rate limits (1-second delays)
* Cache extracted environmental data
* Monitor disk space for WorldClim files (~500MB total)
* Use smaller sample datasets during development
