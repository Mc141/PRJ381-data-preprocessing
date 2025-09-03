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
