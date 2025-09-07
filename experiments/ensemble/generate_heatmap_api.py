"""
Generate grid-based heatmaps for Pyracantha invasion risk prediction using Ensemble model and real API data.

This script:
1. Creates a grid of coordinates for the specified geographic area
2. Fetches real environmental data from the local FastAPI endpoints for each grid point
3. Processes and formats the data for compatibility with the trained Ensemble model
4. Predicts invasion risk for each grid point
5. Creates an interactive grid-based choropleth map of invasion risk

Usage:
    python -m experiments.ensemble.generate_heatmap_api [options]

Requirements:
    - FastAPI server running on localhost:8000 (uvicorn app.main:app)
    - Trained Ensemble model (model.pkl) and optimal_threshold.pkl
    - Python packages: folium, numpy, pandas, requests, scikit-learn, xgboost
"""

import os
import sys
import argparse
import datetime
import pickle
import numpy as np
import pandas as pd
from math import pi
import folium
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster, MeasureControl, Fullscreen
import aiohttp
import asyncio
import json
import time
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Directory where model files are stored
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model_file='model.pkl', threshold_file='optimal_threshold.pkl'):
    """Load the trained ensemble model and optimal threshold from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, model_file)
        threshold_path = os.path.join(MODEL_DIR, threshold_file)
        
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("\nPlease train the model first using:")
            print("python -m experiments.ensemble.train_model")
            sys.exit(1)
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Load optimal threshold if available
        threshold = 0.5  # Default threshold
        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                threshold = pickle.load(f)
            print(f"Using optimal threshold: {threshold:.4f}")
        else:
            print("Optimal threshold file not found, using default (0.5)")
            
        print(f"Model loaded successfully: {type(model).__name__}")
        return model, threshold
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

async def fetch_environmental_data_batch(session, coordinates, bio_variables, max_retries=5):
    """
    Fetch real environmental data from the API endpoint in batches with exponential backoff retry.
    
    Args:
        session: aiohttp ClientSession
        coordinates: List of coordinate dicts
        bio_variables: List of bioclimate variables
        max_retries: Maximum number of retry attempts
    """
    # Convert variables to format expected by API
    variables_str = [str(v) for v in bio_variables]
    
    # Prepare the payload
    payload = {
        "coordinates": coordinates
    }
    
    # Build the URL with query parameters
    variables_param = "&".join([f"variables={v}" for v in variables_str])
    url = f"http://localhost:8000/api/v1/environmental/extract-batch?{variables_param}"
    
    # Exponential backoff parameters
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching environmental data batch for {len(coordinates)} coordinates...")
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Too Many Requests
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Rate limit hit (429). Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue  # Try again
                else:
                    error_text = await response.text()
                    print(f"Error fetching environmental data: {response.status}\n{error_text}")
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    return None
        except Exception as e:
            print(f"Error making API request: {e}")
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            return None
    
    print(f"Failed to fetch data after {max_retries} attempts")
    return None

async def fetch_elevation_batch(session, coordinates, max_retries=5):
    """
    Fetch real elevation data from the API endpoint.
    
    Args:
        session: aiohttp ClientSession
        coordinates: List of coordinate dicts
        max_retries: Maximum number of retry attempts
    """
    # Prepare the payload
    payload = {
        "coordinates": coordinates
    }
    
    # API endpoint for elevation
    url = "http://localhost:8000/api/v1/elevation/extract-batch"
    
    # Exponential backoff parameters
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching elevation data batch for {len(coordinates)} coordinates...")
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Too Many Requests
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit (429). Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    error_text = await response.text()
                    print(f"Error fetching elevation data: {response.status}\n{error_text}")
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    return None
        except Exception as e:
            print(f"Error making elevation API request: {e}")
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            return None
    
    print(f"Failed to fetch elevation data after {max_retries} attempts")
    return None

async def create_environmental_grid_with_real_data(lat_min, lat_max, lon_min, lon_max, grid_size, month):
    """
    Create a grid of points with REAL environmental data from API endpoints.
    
    This function:
    1. Creates a uniform grid of coordinates across the specified geographic area
    2. Calls the FastAPI environmental data batch endpoint to get real data
    3. Processes the API responses and formats them for model compatibility
    4. Fills any missing values with sensible defaults to prevent errors
    
    Args:
        lat_min (float): Minimum latitude (southern boundary)
        lat_max (float): Maximum latitude (northern boundary)
        lon_min (float): Minimum longitude (western boundary)
        lon_max (float): Maximum longitude (eastern boundary)
        grid_size (int): Number of points along each axis
        month (int): Month of year (1-12) for seasonal factors
        
    Returns:
        tuple: (DataFrame with environmental data, latitude grid, longitude grid)
    """
    print(f"Creating environmental grid with REAL data for area: Lat {lat_min} to {lat_max}, Lon {lon_min} to {lon_max}")
    
    # Create a grid of lat-lon points
    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    
    # Create meshgrid for visualization
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    
    # Flatten the grid for prediction
    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()
    
    # Create DataFrame with coordinates and month
    env_data = pd.DataFrame({
        'latitude': flat_lats,
        'longitude': flat_lons,
        'month': month
    })
    
    # Prepare coordinates for API call
    coordinates = []
    for lat, lon in zip(flat_lats, flat_lons):
        coordinates.append({
            "latitude": float(lat),
            "longitude": float(lon)
        })
    
    # Define bio variables needed for model
    bio_variables = [1, 4, 5, 6, 12, 13, 14, 15]
    
    # Split into manageable batches (API might have limits)
    BATCH_SIZE = 20  # Smaller batch size to reduce rate limiting issues
    all_results = []
    
    # Rate limiting parameters
    min_delay_between_batches = 1.0  # Minimum delay in seconds between batches
    last_request_time = 0
    
    # Create session with rate limiting headers
    headers = {
        "User-Agent": "PyracanthaRiskMapGenerator/1.0",  # Identify our application
        "X-Request-Source": "GridMapGenerator"  # Help API identify source of requests
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # First, get environmental data
        for i in range(0, len(coordinates), BATCH_SIZE):
            batch = coordinates[i:i+BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (len(coordinates) + BATCH_SIZE - 1)//BATCH_SIZE
            print(f"Processing environmental batch {batch_num}/{total_batches} ({len(batch)} coordinates)")
            
            # Implement rate limiting
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < min_delay_between_batches:
                delay = min_delay_between_batches - time_since_last + random.uniform(0.1, 0.5)  # Add jitter
                print(f"Rate limiting: waiting {delay:.2f}s between batches")
                await asyncio.sleep(delay)
            
            # Fetch data from API
            batch_results = await fetch_environmental_data_batch(session, batch, bio_variables)
            last_request_time = time.time()
            
            if batch_results:
                all_results.extend(batch_results.get('results', []))
            else:
                # If API call failed, fill with NaN
                for _ in batch:
                    all_results.append({})
        
        # Now, get elevation data
        elevation_results = []
        for i in range(0, len(coordinates), BATCH_SIZE):
            batch = coordinates[i:i+BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (len(coordinates) + BATCH_SIZE - 1)//BATCH_SIZE
            print(f"Processing elevation batch {batch_num}/{total_batches} ({len(batch)} coordinates)")
            
            # Implement rate limiting
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < min_delay_between_batches:
                delay = min_delay_between_batches - time_since_last + random.uniform(0.1, 0.5)
                print(f"Rate limiting: waiting {delay:.2f}s between batches")
                await asyncio.sleep(delay)
            
            # Fetch data from API
            batch_results = await fetch_elevation_batch(session, batch)
            last_request_time = time.time()
            
            if batch_results:
                elevation_results.extend(batch_results.get('results', []))
            else:
                # If API call failed, fill with NaN
                for _ in batch:
                    elevation_results.append({})
    
    # Process the environmental data results
    for i, result in enumerate(all_results):
        if not result or 'error' in result:
            continue
            
        env_data_point = result.get('data', {})
        
        # Add bio variables with proper prefixes
        for bio_var in bio_variables:
            var_name = f'bio{bio_var}'
            env_data.loc[i, var_name] = env_data_point.get(var_name)
    
    # Process the elevation results
    for i, result in enumerate(elevation_results):
        if not result or 'error' in result:
            continue
            
        env_data.loc[i, 'elevation'] = result.get('elevation')
    
    # Calculate cyclic month features
    env_data['sin_month'] = np.sin(2 * np.pi * env_data['month'] / 12)
    env_data['cos_month'] = np.cos(2 * np.pi * env_data['month'] / 12)
    
    # Add day of year for features that need it
    # Assuming mid-month for simplicity
    env_data['day_of_year'] = np.ceil((env_data['month'] - 1) * 30.5 + 15)
    
    # Generate derived features needed for the ensemble model
    try:
        # Distance from median presence (proxy feature)
        center_lat, center_lon = -33.5, 18.8  # Approximate center of presence points in South Africa
        env_data['dist_from_median'] = np.sqrt((env_data['latitude'] - center_lat)**2 + 
                                              (env_data['longitude'] - center_lon)**2)
        
        # Environmental interaction terms
        env_data['elev_temp'] = env_data['elevation'] * env_data['bio1'] / 1000  # Scaled interaction
        env_data['precip_seasonality_ratio'] = env_data['bio15'] / (env_data['bio12'] + 1)  # Avoid div by 0
        
        # Temperature and precipitation interactions
        bio_pairs = [(1, 12), (1, 13), (1, 14), (4, 5), (4, 12), (5, 6), (5, 12), 
                     (5, 13), (6, 13), (6, 14), (6, 15), (13, 15), (14, 15)]
                     
        for b1, b2 in bio_pairs:
            feature_name = f'bio{b1}_x_bio{b2}'
            env_data[feature_name] = env_data[f'bio{b1}'] * env_data[f'bio{b2}']
        
        # Elevation interactions
        for b in [1, 6, 13]:
            feature_name = f'elevation_x_bio{b}'
            env_data[feature_name] = env_data['elevation'] * env_data[f'bio{b}'] / 1000  # Scaled
    except Exception as e:
        print(f"Warning: Could not generate all derived features: {e}")
    
    # Replace NaNs with appropriate default values
    print("Checking data completeness...")
    for col in env_data.columns:
        nan_count = env_data[col].isna().sum()
        if nan_count > 0:
            print(f"  Column '{col}' has {nan_count} missing values")
            
            # Use column means for replacement
            if col in ['bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15', 'elevation']:
                col_mean = env_data[col].mean()
                if pd.isna(col_mean):  # If all values are NaN
                    # Use reasonable defaults for bioclimatic variables
                    defaults = {
                        'bio1': 150,    # 15.0°C in tenths
                        'bio4': 5000,   # Standard deviation
                        'bio5': 250,    # 25.0°C in tenths
                        'bio6': 50,     # 5.0°C in tenths
                        'bio12': 800,   # 800mm annual precipitation
                        'bio13': 100,   # 100mm wettest month
                        'bio14': 10,    # 10mm driest month
                        'bio15': 50,    # 50% seasonality
                        'elevation': 300 # 300m elevation
                    }
                    env_data[col].fillna(defaults.get(col, 0), inplace=True)
                    print(f"    All values missing for '{col}', using default: {defaults.get(col, 0)}")
                else:
                    env_data[col].fillna(col_mean, inplace=True)
                    print(f"    Replaced with column mean: {col_mean:.2f}")
    
    # Convert all numeric columns to float to avoid XGBoost dtype errors
    print("Converting all feature columns to float type...")
    for col in env_data.columns:
        if col not in ['latitude', 'longitude']:  # Keep coordinates as they are
            try:
                env_data[col] = env_data[col].astype(float)
            except Exception as e:
                print(f"Warning: Could not convert '{col}' to float: {e}")
    
    # Double check that all required columns exist
    required_columns = [
        'latitude', 'longitude', 'elevation', 
        'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15',
        'month', 'sin_month', 'cos_month', 'day_of_year',
        'dist_from_median', 'elev_temp', 'precip_seasonality_ratio'
    ]
    
    for col in required_columns:
        if col not in env_data.columns:
            print(f"Warning: Required column '{col}' is missing. Adding with default values.")
            env_data[col] = 0
    
    print("Environmental grid created successfully with real data")
    return env_data, lat_grid, lon_grid

def predict_invasion_risk(model, threshold, env_data):
    """Predict invasion risk using the trained ensemble model."""
    print("Predicting invasion risk...")
    
    # Recreate exactly the same features used in training (from train_model.py)
    # Create a copy of the dataframe to avoid modifying the original
    df = env_data.copy()
    
    # 1. Climate indices and ecological features (exactly as in train_model.py)
    # Temperature range
    df['temp_range'] = df['bio5'] - df['bio6']
    
    # Aridity index
    df['aridity_index'] = df['bio12'] / (df['bio1'] + 10)
    
    # Growing degree days approximation
    df['growing_degree_approx'] = np.maximum(0, df['bio1'] - 5) * (1 - 0.1 * df['bio4'] / 100)
    
    # 3. Interaction terms (exactly as in train_model.py)
    # Temperature-precipitation interactions
    df['temp_precip'] = df['bio1'] * df['bio12'] / 1000
    
    # Selected key interactions (using only a subset to avoid too many features)
    df['bio1_x_bio4'] = df['bio1'] * df['bio4'] / 100
    
    # Complete feature columns including the engineered ones - MUST match exactly what was used in training
    feature_columns = [
        'latitude', 'longitude', 'elevation', 
        'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15',
        'month', 'day_of_year', 'sin_month', 'cos_month', 
        'temp_range', 'aridity_index', 'precip_seasonality_ratio', 'growing_degree_approx',
        'dist_from_median', 'temp_precip', 'elev_temp', 
        'bio1_x_bio12', 'bio1_x_bio4', 'bio4_x_bio12', 'bio5_x_bio13', 
        'bio6_x_bio14', 'bio13_x_bio15'
    ]
    
    # Debug: check that all columns exist
    missing_cols = [col for col in feature_columns if col not in env_data.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {env_data.columns.tolist()}")
    
    # Make sure all columns exist (with empty values if needed)
    for col in feature_columns:
        if col not in env_data.columns:
            env_data[col] = np.nan
    
    X = env_data[feature_columns]
    
    # Get probability predictions (class 1 = presence)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Apply optimal threshold if available
    if threshold != 0.5:
        print(f"Applying optimal threshold: {threshold:.4f}")
    
    return y_prob

def create_heatmap(lats, lons, risk_scores, lat_grid, lon_grid, month, output_file):
    """Create an interactive choropleth grid map of invasion risk."""
    print(f"Creating grid-based visualization for month {month}...")
    
    # Get month name
    month_name = datetime.date(2000, month, 1).strftime('%B')
    
    # Calculate the center point of the map
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Calculate min/max for display
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Create a map
    risk_map = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                        tiles='CartoDB positron')
    
    # Add Folium plugins for better visualization
    try:
        # Add fullscreen button
        Fullscreen().add_to(risk_map)
        
        # Add measurement tool
        MeasureControl(position='topright', primary_length_unit='kilometers').add_to(risk_map)
    except ImportError:
        print("Some Folium plugins not available. Continuing with basic map.")
    
    # Add a base layer group
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(risk_map)
    
    # Add map title
    title_html = f'''
        <h3 align="center" style="font-size:18px; margin-top: 10px;">
            <b>Pyracantha angustifolia Invasion Risk Prediction</b><br>
            <span style="font-size:14px">{month_name} | Ensemble Model | REAL API DATA</span>
        </h3>
    '''
    
    # Safe way to add HTML to map
    try:
        folium.Element(title_html).add_to(risk_map)
    except Exception as e:
        print(f"Warning: Could not add title to map: {e}")
    
    # Define color function for risk levels
    def get_color(risk):
        if risk > 0.8:
            return '#FF0000'  # Red - Very high risk
        elif risk > 0.6:
            return '#FFA500'  # Orange - High risk
        elif risk > 0.4:
            return '#FFFF00'  # Yellow - Medium risk
        elif risk > 0.2:
            return '#00FF00'  # Green - Low risk
        else:
            return '#0000FF'  # Blue - Very low risk
    
    # Define the style function for grid cells
    def style_function(feature):
        risk = feature['properties']['risk']
        return {
            'fillColor': get_color(risk),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    # Create a GeoJSON feature collection for the grid cells
    features = []
    
    # Grid spacing in degrees
    dlat = (max(lats) - min(lats)) / (lat_grid.shape[0] - 1)
    dlon = (max(lons) - min(lons)) / (lon_grid.shape[1] - 1)
    
    # Create the grid cells as polygon features
    k = 0
    for i in range(lat_grid.shape[0] - 1):
        for j in range(lon_grid.shape[1] - 1):
            # Get the risk score for this grid cell
            risk = float(risk_scores[k])
            k += 1
            
            # Create a polygon for this grid cell
            polygon = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [lon_grid[i, j], lat_grid[i, j]],
                        [lon_grid[i, j+1], lat_grid[i, j+1]],
                        [lon_grid[i+1, j+1], lat_grid[i+1, j+1]],
                        [lon_grid[i+1, j], lat_grid[i+1, j]],
                        [lon_grid[i, j], lat_grid[i, j]]
                    ]]
                },
                'properties': {
                    'risk': risk,
                    'lat': lat_grid[i, j],
                    'lon': lon_grid[i, j]
                }
            }
            features.append(polygon)
    
    # Create a GeoJSON layer with the grid cells
    grid_layer = folium.GeoJson(
        {'type': 'FeatureCollection', 'features': features},
        name='Invasion Risk',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['risk', 'lat', 'lon'],
            aliases=['Risk Score', 'Latitude', 'Longitude'],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,
        )
    )
    grid_layer.add_to(risk_map)
    
    # Add a colormap legend
    colormap = LinearColormap(
        colors=['blue', 'green', 'yellow', 'orange', 'red'],
        vmin=0, vmax=1,
        caption='Invasion Risk Score'
    )
    colormap.add_to(risk_map)
    
    # Add a stats panel with summary info
    stats_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px;
                    border:2px solid grey; z-index:9999; 
                    background-color:white; padding: 10px;
                    font-size:14px;">
            <span style="font-size:16px; font-weight: bold;">Invasion Risk Summary</span>
            <table style="width:100%; table-layout:fixed; margin-top:5px;">
                <tr><td><b>Area:</b></td><td>{lat_min:.2f}°S to {lat_max:.2f}°S, {lon_min:.2f}°E to {lon_max:.2f}°E</td></tr>
                <tr><td><b>Month:</b></td><td>{month_name}</td></tr>
                <tr><td><b>Mean Risk:</b></td><td>{risk_scores.mean():.3f}</td></tr>
                <tr><td><b>Max Risk:</b></td><td>{risk_scores.max():.3f}</td></tr>
                <tr><td><b>High Risk (>0.7):</b></td><td>{(risk_scores > 0.7).sum()} locations</td></tr>
                <tr><td><b>Med Risk (0.4-0.7):</b></td><td>{((risk_scores > 0.4) & (risk_scores <= 0.7)).sum()} locations</td></tr>
                <tr><td><b>Model:</b></td><td>Ensemble with REAL DATA</td></tr>
            </table>
            <div style="font-size:10px; margin-top:5px; text-align:right;">
                Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>
    '''
    
    # Safe way to add stats panel
    try:
        folium.Element(stats_html).add_to(risk_map)
    except Exception as e:
        print(f"Warning: Could not add stats panel to map: {e}")
    
    # Add layer control
    folium.LayerControl().add_to(risk_map)
    
    # Save the map
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"invasion_prediction_map_{timestamp}.html"
    
    output_path = os.path.join(MODEL_DIR, output_file)
    risk_map.save(output_path)
    print(f"Grid map saved to {output_path}")
    
    return output_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Pyracantha invasion risk grid map using Ensemble Model with REAL API DATA.')
    
    parser.add_argument('--grid_size', type=int, default=20,
                        help='Size of the grid (number of points in each dimension). Higher values give more detail but take longer. Default is 20 for real API data since each point requires API calls.')
    parser.add_argument('--western_cape_extended', action='store_true',
                        help='Use extended Western Cape region boundaries instead of core area')
    parser.add_argument('--specific_area', action='store_true',
                        help='Define a specific area within Western Cape')
    parser.add_argument('--lat_min', type=float, default=None,
                        help='Minimum latitude (if specific_area is used)')
    parser.add_argument('--lat_max', type=float, default=None,
                        help='Maximum latitude (if specific_area is used)')
    parser.add_argument('--lon_min', type=float, default=None,
                        help='Minimum longitude (if specific_area is used)')
    parser.add_argument('--lon_max', type=float, default=None,
                        help='Maximum longitude (if specific_area is used)')
    parser.add_argument('--month', type=int, default=3,
                        help='Month (1-12) for prediction. Controls the seasonality factor (default: 3)')
    parser.add_argument('--output_file', type=str, default='',
                        help='Output HTML file name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set Western Cape boundaries
    if args.western_cape_extended:
        # Extended Western Cape region
        args.lat_min = -34.5
        args.lat_max = -32.0
        args.lon_min = 18.0
        args.lon_max = 21.0
    elif args.specific_area:
        # User defined specific area
        if args.lat_min is None or args.lat_max is None or args.lon_min is None or args.lon_max is None:
            parser.error('Specific area requires lat_min, lat_max, lon_min, and lon_max')
    else:
        # Core Western Cape region (default)
        args.lat_min = -34.2
        args.lat_max = -33.8
        args.lon_min = 18.2
        args.lon_max = 19.0
    
    return args

async def main_async():
    """Main async function to run the heatmap generation pipeline."""
    print("Starting invasion risk grid map generation pipeline with REAL API DATA using Ensemble Model...")
    print("NOTE: This script requires the FastAPI server to be running.")
    print("      Start it with: python -m uvicorn app.main:app --reload")
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the trained model and optimal threshold
    model, threshold = load_model()
    
    # Create a grid of environmental data with REAL data from APIs
    env_data, lat_grid, lon_grid = await create_environmental_grid_with_real_data(
        args.lat_min, args.lat_max, 
        args.lon_min, args.lon_max, 
        args.grid_size, 
        args.month
    )
    
    # Predict invasion risk
    risk_scores = predict_invasion_risk(model, threshold, env_data)
    
    # Create and save the heatmap
    output_path = create_heatmap(
        env_data['latitude'], env_data['longitude'], 
        risk_scores, lat_grid, lon_grid, 
        args.month, args.output_file
    )
    
    print(f"Process complete. Heatmap saved to {output_path}")
    return output_path

def main():
    """Wrapper for async main function"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(main_async())

if __name__ == "__main__":
    main()
