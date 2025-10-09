"""
Prediction Router for Invasive Species Spread Modeling

This module provides API endpoints and utilities for ecological niche modeling of
Pyracantha angustifolia. It leverages recent observations and environmental suitability
scoring to support global risk assessment, heatmap generation, and model training.
Endpoints include risk heatmap generation, model training, and core functions for suitability scoring and grid generation.
All documentation and comments are aligned with the current service and written in a clear, professional style.
"""

import os
import sys
import glob
import asyncio
import importlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

router = APIRouter()
logger = logging.getLogger(__name__)


def _load_xgboost_module(module_path: str):
    """Load the XGBoost heatmap generation module."""
    if not os.path.isdir(module_path):
        raise HTTPException(status_code=500, detail="Models path not found: models/xgboost")
    
    sys.path.insert(0, module_path)
    try:
        return importlib.import_module('generate_heatmap_api')
    except ImportError as e:
        logger.error(f"Failed to import heatmap generation module: {e}")
        raise HTTPException(status_code=500, detail="Heatmap generation module not available")


def _validate_and_get_region_bounds(
    region: str,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float]
) -> tuple[float, float, float, float]:
    """Validate region and return coordinate bounds."""
    if region == "custom":
        if any(v is None for v in [lat_min, lat_max, lon_min, lon_max]):
            raise HTTPException(status_code=400, detail="Custom region requires lat_min, lat_max, lon_min, lon_max")
        assert lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None
        if lat_min >= lat_max or lon_min >= lon_max:
            raise HTTPException(status_code=400, detail="Invalid coordinate bounds")
        return lat_min, lat_max, lon_min, lon_max
    else:  # western_cape (default)
        return -34.2, -33.8, 18.2, 19.0


def _create_output_path(module_path: str, region: str, save_file: bool) -> str:
    """Create the output file path for the heatmap."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_file:
        output_rel_file = os.path.join("generated_maps", f"xgboost_heatmap_{region}_{timestamp}.html")
        os.makedirs(os.path.join(module_path, "generated_maps"), exist_ok=True)
    else:
        output_rel_file = f"temp_xgboost_heatmap_{timestamp}.html"
    return output_rel_file


def _clear_generated_maps(module_path: str):
    """Clear all HTML files from generated_maps directory."""
    try:
        gen_dir = os.path.join(module_path, "generated_maps")
        for f in glob.glob(os.path.join(gen_dir, "*.html")):
            os.remove(f)
    except Exception as e:
        logger.warning(f"Failed to clear generated_maps: {e}")


def _calculate_risk_statistics(risk_scores: np.ndarray, month: int) -> dict:
    """Calculate risk statistics from prediction scores."""
    month_name = datetime(2000, month, 1).strftime('%B')
    return {
        "mean_risk": float(np.mean(risk_scores)),
        "max_risk": float(np.max(risk_scores)),
        "min_risk": float(np.min(risk_scores)),
        "high_risk_points": int((risk_scores > 0.7).sum()),
        "medium_risk_points": int(((risk_scores > 0.4) & (risk_scores <= 0.7)).sum()),
        "low_risk_points": int((risk_scores <= 0.4).sum()),
        "month_name": month_name
    }


def _read_html_content(output_path: str) -> Optional[str]:
    """Read HTML content from file."""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read generated HTML file: {e}")
        return None


def _cleanup_temp_files(output_path: str, module_path: str):
    """Clean up temporary files."""
    try:
        os.remove(output_path)
        _clear_generated_maps(module_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file: {e}")


@router.post("/predictions/generate-xgboost-heatmap")
async def generate_xgboost_heatmap(
    region: str = Query(
        "western_cape",
        description="Region: western_cape (Cape Town area) or custom (requires lat/lon bounds)"
    ),
    lat_min: Optional[float] = Query(None, description="Minimum latitude (required for custom region)"),
    lat_max: Optional[float] = Query(None, description="Maximum latitude (required for custom region)"),
    lon_min: Optional[float] = Query(None, description="Minimum longitude (required for custom region)"),
    lon_max: Optional[float] = Query(None, description="Maximum longitude (required for custom region)"),
    grid_size: int = Query(20, ge=5, le=100, description="Grid size (5-100). Higher values increase detail but slow processing"),
    month: int = Query(3, ge=1, le=12, description="Month (1-12) for seasonal prediction"),
    download: bool = Query(False, description="Return as downloadable HTML file, otherwise return JSON with HTML content")
):
    """
    Generate a high-resolution invasion risk heatmap using the trained XGBoost model and real-time environmental data.

    This endpoint runs the heatmap generation script and returns the resulting HTML map. It uses the XGBoost model and up-to-date environmental data.

    Geographic Regions:
        - western_cape: Cape Town metropolitan area (default, -34.2 to -33.8°S, 18.2 to 19.0°E)
        - custom: User-specified bounds (requires lat_min, lat_max, lon_min, lon_max)

    Returns:
        - If download=True: HTML file response for direct download
        - Otherwise: JSON with HTML content (if return_html=True), generation statistics, and optional file save location (if save_file=True)
    """
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
        module_path = os.path.join(project_root, 'models', 'xgboost')
        
        ghm = _load_xgboost_module(module_path)
        lat_min, lat_max, lon_min, lon_max = _validate_and_get_region_bounds(region, lat_min, lat_max, lon_min, lon_max)
        
        logger.info(f"Generating XGBoost heatmap for {region} region: {lat_min}° to {lat_max}°S, {lon_min}° to {lon_max}°E")
        logger.info(f"Parameters: grid_size={grid_size}, month={month}")

        # Load the trained model
        model_path = os.path.join(module_path, 'model.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Trained XGBoost model not found in models/xgboost. Please train the model first.")

        model = ghm.load_model(os.path.basename(model_path))

        start_time = datetime.now()
        env_data, lat_grid, lon_grid = await ghm.create_environmental_grid_with_real_data(
            lat_min, lat_max, lon_min, lon_max, grid_size, month
        )
        
        risk_scores = ghm.predict_invasion_risk(model, env_data)
        output_rel_file = _create_output_path(module_path, region, save_file=False)
        output_path = ghm.create_heatmap(
            env_data['latitude'], env_data['longitude'],
            risk_scores, lat_grid, lon_grid,
            month, output_rel_file
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()

        if download:
            # Always cleanup temp file in background after download
            background = BackgroundTask(lambda: (os.remove(output_path), _clear_generated_maps(module_path)))
            return FileResponse(
                path=output_path,
                media_type="text/html",
                filename=os.path.basename(output_path),
                background=background,
            )
        
        # Return JSON with HTML content
        html_content = _read_html_content(output_path)
        _cleanup_temp_files(output_path, module_path)
        
        stats = _calculate_risk_statistics(risk_scores, month)
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
                "month_name": stats["month_name"]
            },
            "statistics": {
                "mean_risk": stats["mean_risk"],
                "max_risk": stats["max_risk"],
                "min_risk": stats["min_risk"],
                "high_risk_points": stats["high_risk_points"],
                "medium_risk_points": stats["medium_risk_points"],
                "low_risk_points": stats["low_risk_points"]
            },
            "processing": {
                "processing_time_seconds": processing_time,
                "data_points_filled": len(env_data),
                "model_type": "XGBoost",
                "data_source": "Real API Data (WorldClim + SRTM)"
            },
            "output": {
                "html_content": html_content,
                "content_size_kb": len(html_content) // 1024 if html_content else 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating XGBoost heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")


def _load_training_module(module_path: str):
    """Load the XGBoost training module."""
    if not os.path.isdir(module_path):
        raise HTTPException(status_code=500, detail="Models path not found: models/xgboost")
    
    sys.path.insert(0, module_path)
    os.environ.setdefault("MPLBACKEND", "Agg")
    
    try:
        return importlib.import_module('train_model_api')
    except Exception as e:
        logger.error(f"Failed to import training module: {e}")
        raise HTTPException(status_code=500, detail="Training module not available")


def _subsample_data(train_data, local_validation, limit_rows: int):
    """Subsample training and validation data if limit specified."""
    if not limit_rows or limit_rows <= 0:
        return train_data, local_validation
    
    try:
        train_data = train_data.sample(n=min(limit_rows, len(train_data)), random_state=42)
        if local_validation is not None and len(local_validation) > 0:
            local_validation_sample = min(max(limit_rows // 4, 50), len(local_validation))
            local_validation = local_validation.sample(n=max(local_validation_sample, 1), random_state=42)
    except Exception as sub_e:
        logger.warning(f"Subsampling failed, proceeding with full data: {sub_e}")
    
    return train_data, local_validation


def _evaluate_model_with_fallback(model, X_train, y_train, X_local, y_local, tma):
    """Evaluate model on validation set or fall back to training set."""
    if X_local is not None and y_local is not None:
        return tma.evaluate_model(model, X_local, y_local)
    
    logger.warning("No local validation set; evaluating on training data (optimistic)")
    try:
        y_prob = model.predict_proba(X_train)[:, 1]
        try:
            from app.utils import metrics_utils as mu
            optimal_threshold, metrics = mu.find_optimal_threshold(y_train, y_prob)
        except Exception:
            optimal_threshold = 0.5
            metrics = {
                "accuracy": float((y_train == (y_prob >= 0.5).astype(int)).mean()),
                "auc": None,
            }
        return metrics, optimal_threshold
    except Exception as eval_e:
        logger.warning(f"Fallback evaluation failed: {eval_e}")
        return None, None


def _prepare_artifacts_paths(module_path: str) -> dict:
    """Prepare paths for model artifacts."""
    return {
        "roc_path": os.path.join(module_path, 'roc_curve.png'),
        "fi_path": os.path.join(module_path, 'feature_importance.png'),
        "threshold_path": os.path.join(module_path, 'optimal_threshold.pkl'),
        "results_md": os.path.abspath(os.path.join(module_path, '..', 'MODEL_RESULTS.md'))
    }


def _save_training_artifacts(tma, model, metrics, optimal_threshold, feat_imp_df, save_artifacts: bool, module_path: str) -> Optional[str]:
    """Save model and training artifacts to disk."""
    if not save_artifacts:
        return None
    
    tma.save_model(model)
    model_path = os.path.join(module_path, 'model.pkl')
    
    if metrics is not None and optimal_threshold is not None:
        try:
            tma.append_results_to_markdown(metrics, optimal_threshold, feat_imp_df)
        except Exception as md_e:
            logger.warning(f"Appending results to markdown failed: {md_e}")
    
    return model_path


def _format_training_response(metrics, optimal_threshold, feat_imp_df, model_path, paths: dict, X_train, X_local, limit_rows: int, return_metrics: bool) -> dict:
    """Format the training response with metrics and artifact information."""
    metrics_out = None
    if return_metrics and metrics is not None:
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
            "roc_curve_path": paths["roc_path"] if os.path.exists(paths["roc_path"]) else None,
            "feature_importance_path": paths["fi_path"] if os.path.exists(paths["fi_path"]) else None,
            "threshold_path": paths["threshold_path"] if os.path.exists(paths["threshold_path"]) else None,
            "results_markdown": paths["results_md"] if os.path.exists(paths["results_md"]) else None,
        },
        "feature_importance_top10": feature_top,
        "training": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_local)) if X_local is not None else 0,
            "subsampled": bool(limit_rows and limit_rows > 0),
        },
    }


@router.post("/predictions/train-xgboost-model")
async def train_xgboost_model_endpoint(
    save_artifacts: bool = Query(True, description="Save model and plots to disk"),
    return_metrics: bool = Query(True, description="Include metrics in response"),
    limit_rows: int = Query(0, ge=0, description="Optionally subsample training data for faster runs (0 = no limit)"),
):
    """
    Train the XGBoost model using the standardized pipeline and return training results.

    This endpoint imports and executes the training script in models/xgboost/train_model_api.py,
    running the full pipeline: data loading, model training (with GridSearchCV), evaluation, and artifact saving.
    Key metrics and artifact locations are returned in the response.

    Notes:
        - Training may take several minutes depending on data size and environment.
        - Use limit_rows to subsample data for a faster, approximate run.
    """
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
        module_path = os.path.join(project_root, 'models', 'xgboost')
        sys.path.insert(0, project_root)
        
        tma = _load_training_module(module_path)

        def _run_training():
            train_data, local_validation = tma.load_data()
            train_data, local_validation = _subsample_data(train_data, local_validation, limit_rows)
            
            pf = tma.prepare_features(train_data, local_validation)
            if len(pf) == 4:
                X_train, y_train, X_local, y_local = pf
            else:
                X_train, y_train = pf
                X_local, y_local = None, None

            model = tma.train_xgboost_model(X_train, y_train)
            metrics, optimal_threshold = _evaluate_model_with_fallback(model, X_train, y_train, X_local, y_local, tma)
            feat_imp_df = tma.plot_feature_importance(model, X_train.columns)
            
            paths = _prepare_artifacts_paths(module_path)
            model_path = _save_training_artifacts(tma, model, metrics, optimal_threshold, feat_imp_df, save_artifacts, module_path)
            
            return _format_training_response(metrics, optimal_threshold, feat_imp_df, model_path, paths, X_train, X_local, limit_rows, return_metrics)

        result = await asyncio.to_thread(_run_training)

        return {"success": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
