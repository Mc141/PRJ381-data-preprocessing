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
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import glob
from pymongo import MongoClient
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
    save_file: bool = Query(False, description="Save HTML file to disk"),
    download: bool = Query(False, description="Return the generated HTML as a downloadable file (attachment)")
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
        - If `download=True`: an HTML file response (attachment) for direct download
        - Else JSON containing:
            - HTML content of the interactive map (if return_html=True)
            - Generation statistics and metadata
            - Optional file save location (if save_file=True)
    """
    try:
        import asyncio  # noqa: F401 (reserved for potential future async offloading)
        import os
        import sys
        import tempfile
        from datetime import datetime
        import importlib

        # Use models/xgboost as the single source of truth
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
        models_xgb_path = os.path.join(project_root, 'models', 'xgboost')
        if not os.path.isdir(models_xgb_path):
            raise HTTPException(status_code=500, detail="Models path not found: models/xgboost")
        module_path = models_xgb_path
        sys.path.insert(0, module_path)

        # Import the heatmap generation module dynamically
        try:
            ghm = importlib.import_module('generate_heatmap_api')
        except ImportError as e:
            logger.error(f"Failed to import heatmap generation module: {e}")
            raise HTTPException(status_code=500, detail="Heatmap generation module not available")

        # Set coordinates based on region
        if region == "custom":
            required_vals = [lat_min, lat_max, lon_min, lon_max]
            if any(v is None for v in required_vals):
                raise HTTPException(status_code=400, detail="Custom region requires lat_min, lat_max, lon_min, lon_max")
            # Type narrowing for linters/type checkers
            assert lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None
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
        model_path = os.path.join(module_path, 'model.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Trained XGBoost model not found in models/xgboost. Please train the model first.")

        model = ghm.load_model(os.path.basename(model_path))

        # Generate environmental grid with real API data
        start_time = datetime.now()
        env_data, lat_grid, lon_grid = await ghm.create_environmental_grid_with_real_data(
            lat_min, lat_max, lon_min, lon_max, grid_size, month
        )

        # Predict invasion risk
        risk_scores = ghm.predict_invasion_risk(model, env_data)

        # Create output file path
        if save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_rel_file = os.path.join("generated_maps", f"xgboost_heatmap_{region}_{timestamp}.html")
            os.makedirs(os.path.join(module_path, "generated_maps"), exist_ok=True)
        else:
            # Use temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_rel_file = f"temp_xgboost_heatmap_{timestamp}.html"

        # Generate the heatmap
        output_path = ghm.create_heatmap(
            env_data['latitude'], env_data['longitude'],
            risk_scores, lat_grid, lon_grid,
            month, output_rel_file
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # If client requested a downloadable file, return it directly
        def clear_generated_maps():
            # Remove all files in models/xgboost/generated_maps
            try:
                gen_dir = os.path.join(module_path, "generated_maps")
                for f in glob.glob(os.path.join(gen_dir, "*.html")):
                    os.remove(f)
            except Exception as e:
                logger.warning(f"Failed to clear generated_maps: {e}")

        if download:
            background = None
            # If we didn't request persistent save, schedule cleanup after sending and clear directory
            if not save_file:
                background = BackgroundTask(lambda: (os.remove(output_path), clear_generated_maps()))
            else:
                background = BackgroundTask(clear_generated_maps)
            return FileResponse(
                path=output_path,
                media_type="text/html",
                filename=os.path.basename(output_path),
                background=background,
            )

        # Otherwise, inline content and/or save file info in JSON
        html_content = None
        if return_html:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read generated HTML file: {e}")
                html_content = None

        # Clean up temporary file if not saving and not downloading, and clear directory
        if not save_file:
            try:
                os.remove(output_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
            clear_generated_maps()

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
                "content_size_kb": (len(html_content) // 1024) if html_content else None
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

    This endpoint imports the training script in `models/xgboost/train_model_api.py`,
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
        import importlib

        # Ensure project root and model dir are on path (models/ only)
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
        models_xgb_path = os.path.join(project_root, 'models', 'xgboost')
        if not os.path.isdir(models_xgb_path):
            raise HTTPException(status_code=500, detail="Models path not found: models/xgboost")
        module_path = models_xgb_path
        sys.path.insert(0, project_root)
        sys.path.insert(0, module_path)

        # Import training module
        try:
            # Use headless backend for matplotlib to avoid display issues
            os.environ.setdefault("MPLBACKEND", "Agg")
            tma = importlib.import_module('train_model_api')
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
                        from app.utils import metrics_utils as mu
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
            roc_path = os.path.join(module_path, 'roc_curve.png')
            fi_path = os.path.join(module_path, 'feature_importance.png')
            threshold_path = os.path.join(module_path, 'optimal_threshold.pkl')
            # MODEL_RESULTS.md lives at models/MODEL_RESULTS.md
            results_md = os.path.abspath(os.path.join(module_path, '..', 'MODEL_RESULTS.md'))

            if save_artifacts:
                tma.save_model(model)
                model_path = os.path.join(module_path, 'model.pkl')
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
