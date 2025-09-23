"""
Prediction router for invasive species spread modeling.

This module implements a simple ecological niche model for Pyracantha angustifolia
based on recent observations and environmental suitability scoring across the
greater Cape Town metropolitan area.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import folium
from fastapi import APIRouter, HTTPException, Query
from pymongo import MongoClient
from app.services.database import get_database
from geopy.distance import geodesic
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Ecological preferences for Pyracantha angustifolia
OPTIMAL_TEMP = (10, 25)  # °C
OPTIMAL_PRECIP = (20, 100)  # mm/month
SPREAD_RADIUS_KM = 2.0  # km radius for distance decay

# Cape Town region bounding box (matches iNaturalist fetcher)
CAPE_TOWN_BOUNDS = {
    "min_lat": -34.43214105152007,
    "max_lat": -33.806848821450004,
    "min_lon": 18.252509856447368,
    "max_lon": 18.580726409181743
}


def suitability_score(temp: float, precip: float) -> float:
    """
    Calculate habitat suitability score based on temperature and precipitation.
    
    Args:
        temp (float): Average temperature in °C
        precip (float): Monthly precipitation in mm
        
    Returns:
        float: Suitability score between 0 and 1
    """
    # Temperature scoring
    if OPTIMAL_TEMP[0] <= temp <= OPTIMAL_TEMP[1]:
        score_temp = 1.0
    elif 5 <= temp <= 30:  # Tolerable range
        score_temp = 0.5
    else:
        score_temp = 0.0
    
    # Precipitation scoring
    if OPTIMAL_PRECIP[0] <= precip <= OPTIMAL_PRECIP[1]:
        score_precip = 1.0
    elif 10 <= precip <= 120:  # Tolerable range
        score_precip = 0.5
    else:
        score_precip = 0.0
    
    return (score_temp + score_precip) / 2


def distance_decay(distance_km: float) -> float:
    """
    Calculate distance decay factor for spread probability.
    
    Args:
        distance_km (float): Distance from known observation in km
        
    Returns:
        float: Decay factor between 0 and 1
    """
    return math.exp(-distance_km / SPREAD_RADIUS_KM)


def create_grid(bounds: Dict, resolution_km: float = 0.5) -> List[Dict]:
    """
    Create a grid of points covering the study area.
    
    Args:
        bounds (Dict): Bounding box with min/max lat/lon
        resolution_km (float): Grid resolution in kilometers
        
    Returns:
        List[Dict]: List of grid points with lat/lon
    """
    # Approximate conversion: 1 degree ≈ 111 km
    lat_step = resolution_km / 111.0
    lon_step = resolution_km / (111.0 * math.cos(math.radians(
        (bounds["min_lat"] + bounds["max_lat"]) / 2
    )))
    
    grid_points = []
    lat = bounds["min_lat"]
    
    while lat <= bounds["max_lat"]:
        lon = bounds["min_lon"]
        while lon <= bounds["max_lon"]:
            grid_points.append({
                "latitude": round(lat, 6),
                "longitude": round(lon, 6)
            })
            lon += lon_step
        lat += lat_step
    
    return grid_points


@router.get("/predictions/presence_baseline")
async def get_presence_baseline(
    days_back: int = Query(100, description="Number of days back to consider for presence", ge=1, le=365)
):
    """
    Get recent observations that form the presence baseline.
    
    Args:
        days_back (int): Number of days back to look for observations
        
    Returns:
        Dict: Presence baseline data with observation locations
    """
    try:
        db = get_database()
        inat_collection = db["inat_observations"]
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        # Query recent observations
        query = {
            "time_observed_at": {"$gte": cutoff_str},
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        }
        
        observations = list(inat_collection.find(query, {
            "_id": 0,
            "id": 1,
            "latitude": 1,
            "longitude": 1,
            "time_observed_at": 1
        }))
        
        if not observations:
            raise HTTPException(
                status_code=404, 
                detail=f"No observations found in the last {days_back} days"
            )
        
        logger.info(f"Found {len(observations)} recent observations for presence baseline")
        
        return {
            "presence_count": len(observations),
            "days_back": days_back,
            "cutoff_date": cutoff_str,
            "observations": observations,
            "bounds": CAPE_TOWN_BOUNDS
        }
        
    except Exception as e:
        logger.error(f"Error getting presence baseline: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve presence baseline")


@router.get("/predictions/suitability_map")
async def generate_suitability_map(
    days_back: int = Query(100, description="Days back for presence data", ge=1, le=365),
    grid_resolution: float = Query(0.5, description="Grid resolution in km", ge=0.1, le=2.0)
):
    """
    Generate habitat suitability map for the study area.
    
    Args:
        days_back (int): Days back to consider for presence baseline
        grid_resolution (float): Resolution of prediction grid in km
        
    Returns:
        Dict: Suitability map data with predictions for each grid cell
    """
    try:
        db = get_database()
        inat_collection = db["inat_observations"]
        weather_collection = db["weather_data"]
        
        # Step 1: Get recent observations (presence baseline)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        observations = list(inat_collection.find({
            "time_observed_at": {"$gte": cutoff_str},
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        }))
        
        if not observations:
            raise HTTPException(
                status_code=404,
                detail=f"No observations found in the last {days_back} days"
            )
        
        # Step 2: Create prediction grid
        grid_points = create_grid(CAPE_TOWN_BOUNDS, grid_resolution)
        logger.info(f"Created grid with {len(grid_points)} points")
        
        # Step 3: Calculate predictions for each grid point
        predictions = []
        
        for point in grid_points:
            lat, lon = point["latitude"], point["longitude"]
            
            # Get recent weather data for this location (approximate)
            # Use weather data from nearest observation point
            recent_weather = weather_collection.find({
                "date": {"$gte": cutoff_str}
            }).limit(30)  # Get sample of recent weather
            
            # Calculate average conditions
            temps = []
            precips = []
            
            for weather in recent_weather:
                if weather.get("T2M") is not None:
                    temps.append(weather["T2M"])
                if weather.get("PRECTOTCORR") is not None:
                    precips.append(weather["PRECTOTCORR"] * 30)  # Convert daily to monthly
            
            if not temps or not precips:
                # Use NaN values if no weather data to identify missing data
                avg_temp = float('nan')
                avg_precip = float('nan')
            else:
                avg_temp = float(np.mean(temps))
                avg_precip = float(np.mean(precips))
            
            # Calculate base suitability (will be 0 if NaN values)
            if not (np.isnan(avg_temp) or np.isnan(avg_precip)):
                base_suitability = suitability_score(avg_temp, avg_precip)
            else:
                base_suitability = 0.0
            
            # Calculate distance-weighted probability from observations
            max_probability = 0.0
            
            for obs in observations:
                obs_lat = float(obs["latitude"])
                obs_lon = float(obs["longitude"])
                
                # Calculate distance
                distance_km = geodesic((lat, lon), (obs_lat, obs_lon)).kilometers
                
                if distance_km == 0:
                    # Exact presence location
                    probability = 1.0
                else:
                    # Distance-weighted suitability
                    decay_factor = distance_decay(distance_km)
                    probability = decay_factor * base_suitability
                
                max_probability = max(max_probability, probability)
            
            predictions.append({
                "latitude": lat,
                "longitude": lon,
                "suitability": round(base_suitability, 3),
                "probability": round(max_probability, 3),
                "avg_temp": round(avg_temp, 1),
                "avg_precip": round(avg_precip, 1)
            })
        
        logger.info(f"Generated predictions for {len(predictions)} grid points")
        
        return {
            "grid_resolution_km": grid_resolution,
            "days_back": days_back,
            "total_grid_points": len(predictions),
            "presence_observations": len(observations),
            "bounds": CAPE_TOWN_BOUNDS,
            "predictions": predictions,
            "summary": {
                "max_probability": max(p["probability"] for p in predictions),
                "avg_suitability": round(np.mean([p["suitability"] for p in predictions]), 3),
                "high_risk_cells": len([p for p in predictions if p["probability"] > 0.7])
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating suitability map: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate suitability map")


@router.get("/predictions/visualize_map")
async def create_prediction_map(
    days_back: int = Query(100, description="Days back for presence data", ge=1, le=365),
    grid_resolution: float = Query(0.5, description="Grid resolution in km", ge=0.1, le=2.0),
    save_file: bool = Query(True, description="Save HTML file to disk")
):
    """
    Create an interactive Folium map showing spread predictions.
    
    Args:
        days_back (int): Days back to consider for presence baseline
        grid_resolution (float): Resolution of prediction grid in km
        save_file (bool): Whether to save HTML file to disk
        
    Returns:
        Dict: Map creation summary and HTML content
    """
    try:
        # Get suitability map data
        suitability_data = await generate_suitability_map(days_back, grid_resolution)
        predictions = suitability_data["predictions"]
        observations = suitability_data["presence_observations"]
        
        # Create Folium map centered on Cape Town region
        center_lat = (CAPE_TOWN_BOUNDS["min_lat"] + CAPE_TOWN_BOUNDS["max_lat"]) / 2
        center_lon = (CAPE_TOWN_BOUNDS["min_lon"] + CAPE_TOWN_BOUNDS["max_lon"]) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add prediction grid
        for cell in predictions:
            if cell["probability"] > 0.1:  # Only show cells with meaningful probability
                color = "red" if cell["probability"] > 0.7 else "orange" if cell["probability"] > 0.4 else "yellow"
                
                folium.CircleMarker(
                    location=(cell["latitude"], cell["longitude"]),
                    radius=3 + (cell["probability"] * 5),  # Size based on probability
                    color=color,
                    fill=True,
                    fillOpacity=cell["probability"] * 0.8,
                    popup=f"""
                    <b>Prediction Cell</b><br>
                    Probability: {cell['probability']:.3f}<br>
                    Suitability: {cell['suitability']:.3f}<br>
                    Temp: {cell['avg_temp']:.1f}°C<br>
                    Precip: {cell['avg_precip']:.1f}mm
                    """
                ).add_to(m)
        
        # Add actual observations as blue markers
        db = get_database()
        recent_obs = list(db["inat_observations"].find({
            "time_observed_at": {"$gte": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")},
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        }))
        
        for obs in recent_obs:
            folium.Marker(
                location=(float(obs["latitude"]), float(obs["longitude"])),
                popup=f"""
                <b>Confirmed Observation</b><br>
                ID: {obs['id']}<br>
                Date: {obs.get('time_observed_at', 'Unknown')}<br>
                Species: Pyracantha angustifolia
                """,
                icon=folium.Icon(color='blue', icon='leaf')
            ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Invasion Risk Legend</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> High Risk (>0.7)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk (0.4-0.7)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Low Risk (0.1-0.4)</p>
        <p><i class="fa fa-map-marker" style="color:blue"></i> Confirmed Sightings</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map if requested
        map_html = m._repr_html_()
        filepath = None
        
        if save_file:
            import os
            # Get absolute path to ensure file is saved in correct location
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            outputs_dir = os.path.join(project_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            filename = f"invasion_prediction_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(outputs_dir, filename)
            
            try:
                m.save(filepath)
                logger.info(f"Map saved to {filepath}")
            except Exception as save_error:
                logger.error(f"Error saving map file: {save_error}")
                filepath = None
        
        return {
            "map_created": True,
            "grid_points": len(predictions),
            "confirmed_observations": len(recent_obs),
            "high_risk_cells": len([p for p in predictions if p["probability"] > 0.7]),
            "medium_risk_cells": len([p for p in predictions if 0.4 <= p["probability"] <= 0.7]),
            "low_risk_cells": len([p for p in predictions if 0.1 <= p["probability"] < 0.4]),
            "file_saved": filepath if save_file else None,
            "map_html": map_html[:1000] + "..." if len(map_html) > 1000 else map_html
        }
        
    except Exception as e:
        logger.error(f"Error creating prediction map: {e}")
        raise HTTPException(status_code=500, detail="Failed to create prediction map")


@router.post("/predictions/generate-xgboost-heatmap")
async def generate_xgboost_heatmap(
    # Geographic area options
    region: str = Query("western_cape_core", description="Predefined region: western_cape_core, western_cape_extended, stellenbosch, garden_route, custom"),
    lat_min: Optional[float] = Query(None, description="Minimum latitude (required for custom region)"),
    lat_max: Optional[float] = Query(None, description="Maximum latitude (required for custom region)"),
    lon_min: Optional[float] = Query(None, description="Minimum longitude (required for custom region)"),
    lon_max: Optional[float] = Query(None, description="Maximum longitude (required for custom region)"),
    
    # Prediction parameters
    grid_size: int = Query(20, ge=5, le=100, description="Grid size (5-100). Higher = more detail but slower"),
    month: int = Query(3, ge=1, le=12, description="Month (1-12) for seasonal prediction"),
    
    # Performance options
    batch_size: int = Query(20, ge=5, le=100, description="API batch size (5-100)"),
    rate_limit_delay: float = Query(1.0, ge=0.1, le=10.0, description="Delay between batches in seconds"),
    
    # Output options
    include_stats: bool = Query(True, description="Include statistics panel on map"),
    return_html: bool = Query(True, description="Return HTML content in response"),
    save_file: bool = Query(False, description="Save HTML file to disk")
):
    """
    Generate a high-resolution invasion risk heatmap using the trained XGBoost model and real API data.
    
    This endpoint runs the heatmap generation script programmatically and returns the generated HTML map.
    It uses the same XGBoost model and real environmental data as the command-line script.
    
    **Geographic Regions:**
    - `western_cape_core`: Cape Town metropolitan area (default)
    - `western_cape_extended`: Larger Western Cape region
    - `stellenbosch`: Stellenbosch wine region
    - `garden_route`: Garden Route region
    - `custom`: Use custom lat/lon bounds (requires lat_min, lat_max, lon_min, lon_max)
    
    **Performance Notes:**
    - Higher grid_size = more detail but longer processing time
    - Estimated time: ~1-3 minutes for grid_size=20, ~5-15 minutes for grid_size=50
    - Uses real-time API calls to fetch environmental data for each grid point
    
    **Returns:**
    - HTML content of the interactive map (if return_html=True)
    - Generation statistics and metadata
    - Optional file save location (if save_file=True)
    """
    try:
        import asyncio
        import os
        import sys
        import tempfile
        from datetime import datetime
        
        # Add the experiments module to the path
        experiments_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'experiments', 'xgboost')
        sys.path.insert(0, experiments_path)
        
        # Import the heatmap generation functions
        try:
            from generate_heatmap_api_options import (
                load_model, create_environmental_grid_with_real_data, 
                predict_invasion_risk, create_heatmap
            )
        except ImportError as e:
            logger.error(f"Failed to import heatmap generation module: {e}")
            raise HTTPException(status_code=500, detail="Heatmap generation module not available")
        
        # Set coordinates based on region
        if region == "custom":
            if not all([lat_min, lat_max, lon_min, lon_max]):
                raise HTTPException(status_code=400, detail="Custom region requires lat_min, lat_max, lon_min, lon_max")
            if lat_min >= lat_max or lon_min >= lon_max:
                raise HTTPException(status_code=400, detail="Invalid coordinate bounds")
        elif region == "western_cape_extended":
            lat_min, lat_max, lon_min, lon_max = -34.5, -32.0, 18.0, 21.0
        elif region == "stellenbosch":
            lat_min, lat_max, lon_min, lon_max = -34.0, -33.7, 18.7, 19.1
        elif region == "garden_route":
            lat_min, lat_max, lon_min, lon_max = -34.5, -33.5, 19.5, 23.5
        else:  # western_cape_core (default)
            lat_min, lat_max, lon_min, lon_max = -34.2, -33.8, 18.2, 19.0
        
        logger.info(f"Generating XGBoost heatmap for {region} region: {lat_min}° to {lat_max}°S, {lon_min}° to {lon_max}°E")
        logger.info(f"Parameters: grid_size={grid_size}, month={month}, batch_size={batch_size}")
        
        # Load the trained model
        model_path = os.path.join(experiments_path, 'model.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Trained XGBoost model not found. Please train the model first.")
        
        model = load_model(model_path)
        
        # Generate environmental grid with real API data
        start_time = datetime.now()
        env_data, lat_grid, lon_grid = await create_environmental_grid_with_real_data(
            lat_min, lat_max, lon_min, lon_max, grid_size, month,
            api_url="http://localhost:8000",  # Use local API
            batch_size=batch_size,
            rate_limit_delay=rate_limit_delay,
            verbose=False  # Reduce logging for API usage
        )
        
        # Predict invasion risk
        risk_scores = predict_invasion_risk(model, env_data)
        
        # Create output file path
        if save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"xgboost_heatmap_{region}_{timestamp}.html"
            output_dir = os.path.join(experiments_path, "generated_maps")
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Use temporary file
            output_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"temp_xgboost_heatmap_{timestamp}.html"
        
        # Generate the heatmap
        output_path = create_heatmap(
            env_data['latitude'], env_data['longitude'],
            risk_scores, lat_grid, lon_grid,
            month, output_file,
            output_dir=output_dir,
            include_stats=include_stats,
            verbose=False
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Read the generated HTML content
        html_content = None
        if return_html:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read generated HTML file: {e}")
                html_content = None
        
        # Clean up temporary file if not saving
        if not save_file:
            try:
                os.remove(output_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
        
        # Calculate statistics
        month_name = datetime(2000, month, 1).strftime('%B')
        high_risk_count = int((risk_scores > 0.7).sum())
        medium_risk_count = int(((risk_scores > 0.4) & (risk_scores <= 0.7)).sum())
        low_risk_count = int((risk_scores <= 0.4).sum())
        
        logger.info(f"XGBoost heatmap generated successfully in {processing_time:.1f}s")
        
        return {
            "success": True,
            "region": region,
            "coordinates": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max
            },
            "parameters": {
                "grid_size": grid_size,
                "total_points": len(risk_scores),
                "month": month,
                "month_name": month_name,
                "batch_size": batch_size,
                "rate_limit_delay": rate_limit_delay
            },
            "statistics": {
                "mean_risk": float(np.mean(risk_scores)),
                "max_risk": float(np.max(risk_scores)),
                "min_risk": float(np.min(risk_scores)),
                "high_risk_points": high_risk_count,
                "medium_risk_points": medium_risk_count,
                "low_risk_points": low_risk_count
            },
            "processing": {
                "processing_time_seconds": processing_time,
                "data_points_filled": len(env_data),
                "model_type": "XGBoost",
                "data_source": "Real API Data (WorldClim + SRTM)"
            },
            "output": {
                "file_saved": output_path if save_file else None,
                "html_content": html_content if return_html else None,
                "content_size_kb": len(html_content) // 1024 if html_content else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating XGBoost heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")


@router.post("/predictions/train-xgboost-model")
async def train_xgboost_model_endpoint(
    save_artifacts: bool = Query(True, description="Save model and plots to disk"),
    return_metrics: bool = Query(True, description="Include metrics in response"),
    limit_rows: int = Query(0, ge=0, description="Optional: Subsample rows from training data for faster runs (0 = no limit)"),
):
    """
    Train the XGBoost model using the standardized training pipeline and return results.

    This endpoint imports the training script in `experiments/xgboost/train_model_api.py`,
    runs the full pipeline (load data, train with GridSearchCV, evaluate, save artifacts),
    and returns key metrics and artifact locations.

    Notes:
    - This operation can take several minutes depending on data size and environment.
    - Use `limit_rows` to run a quicker subsampled training for sanity checks.
    """
    try:
        import asyncio
        import os
        import sys
        import numpy as np

        # Ensure project root and experiments dirs are on path
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
        experiments_root = os.path.join(project_root, 'experiments')
        experiments_xgb_path = os.path.join(experiments_root, 'xgboost')
        sys.path.insert(0, project_root)
        sys.path.insert(0, experiments_root)
        sys.path.insert(0, experiments_xgb_path)

        # Import training module
        try:
            # Use headless backend for matplotlib to avoid display issues
            os.environ.setdefault("MPLBACKEND", "Agg")
            import train_model_api as tma
        except Exception as e:
            logger.error(f"Failed to import training module: {e}")
            raise HTTPException(status_code=500, detail="Training module not available")

        # Define a blocking function to run in a thread
        def _run_training():
            # Load data
            train_data, local_validation = tma.load_data()

            # Optional subsampling for speed
            if limit_rows and limit_rows > 0:
                try:
                    train_data = train_data.sample(n=min(limit_rows, len(train_data)), random_state=42)
                    if local_validation is not None and len(local_validation) > 0:
                        local_validation_sample = min(max(limit_rows // 4, 50), len(local_validation))
                        local_validation_sample = max(local_validation_sample, 1)
                        local_validation = local_validation.sample(n=local_validation_sample, random_state=42)
                except Exception as sub_e:
                    # Proceed without subsampling if anything goes wrong
                    logger.warning(f"Subsampling failed, proceeding with full data: {sub_e}")

            # Prepare features
            pf = tma.prepare_features(train_data, local_validation)
            if len(pf) == 4:
                X_train, y_train, X_local, y_local = pf
            else:
                X_train, y_train = pf
                X_local, y_local = None, None

            # Train model (GridSearchCV inside)
            model = tma.train_xgboost_model(X_train, y_train)

            # Evaluate if we have local validation
            metrics = None
            optimal_threshold = None
            if X_local is not None and y_local is not None:
                metrics, optimal_threshold = tma.evaluate_model(model, X_local, y_local)
            else:
                # Fallback: evaluate on training (warn in logs)
                logger.warning("No local validation set; evaluating on training data (optimistic)")
                try:
                    y_prob = model.predict_proba(X_train)[:, 1]
                    # Try to import standardized threshold finder
                    try:
                        import metrics_utils as mu  # from experiments root
                        optimal_threshold, metrics = mu.find_optimal_threshold(y_train, y_prob)
                    except Exception:
                        optimal_threshold = 0.5
                        metrics = {
                            "accuracy": float((y_train == (y_prob >= 0.5).astype(int)).mean()),
                            "auc": None,
                        }
                except Exception as eval_e:
                    logger.warning(f"Fallback evaluation failed: {eval_e}")

            # Plot feature importance
            feat_imp_df = tma.plot_feature_importance(model, X_train.columns)

            # Save artifacts
            model_path = None
            roc_path = os.path.join(experiments_xgb_path, 'roc_curve.png')
            fi_path = os.path.join(experiments_xgb_path, 'feature_importance.png')
            threshold_path = os.path.join(experiments_xgb_path, 'optimal_threshold.pkl')
            results_md = os.path.abspath(os.path.join(experiments_xgb_path, '..', 'MODEL_RESULTS.md'))

            if save_artifacts:
                tma.save_model(model)
                model_path = os.path.join(experiments_xgb_path, 'model.pkl')
                if metrics is not None and optimal_threshold is not None:
                    try:
                        tma.append_results_to_markdown(metrics, optimal_threshold, feat_imp_df)
                    except Exception as md_e:
                        logger.warning(f"Appending results to markdown failed: {md_e}")

            # Prepare JSON-serializable pieces
            metrics_out = None
            if return_metrics and metrics is not None:
                # Ensure native Python types
                metrics_out = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}

            feature_top = None
            try:
                feature_top = [
                    {"feature": str(r.Feature), "importance": float(r.Importance)}
                    for _, r in feat_imp_df.sort_values('Importance', ascending=False).head(10).iterrows()
                ]
            except Exception:
                feature_top = None

            return {
                "metrics": metrics_out,
                "optimal_threshold": float(optimal_threshold) if optimal_threshold is not None else None,
                "artifacts": {
                    "model_path": model_path,
                    "roc_curve_path": roc_path if os.path.exists(roc_path) else None,
                    "feature_importance_path": fi_path if os.path.exists(fi_path) else None,
                    "threshold_path": threshold_path if os.path.exists(threshold_path) else None,
                    "results_markdown": results_md if os.path.exists(results_md) else None,
                },
                "feature_importance_top10": feature_top,
                "training": {
                    "train_rows": int(len(X_train)),
                    "validation_rows": int(len(X_local)) if X_local is not None else 0,
                    "subsampled": bool(limit_rows and limit_rows > 0),
                },
            }

        # Run the blocking training in a background thread
        result = await asyncio.to_thread(_run_training)

        return {"success": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
