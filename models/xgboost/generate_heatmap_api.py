"""
Generate grid-based heatmaps for Pyracantha invasion risk prediction using XGBoost model and real API data.

This script:
1. Creates a grid of coordinates for the specified geographic area
2. Fetches real environmental data from the local API endpoints for each grid point
3. Processes and formats the data for compatibility with the trained XGBoost model
4. Predicts invasion risk for each grid point
5. Creates an interactive grid-based choropleth map of invasion risk

Usage:
    python -m experiments.xgboost.generate_heatmap_api [options]

Requirements:
    - FastAPI server running on localhost:8000 (uvicorn app.main:app)
    - Trained XGBoost model (model.pkl)
    - Python packages: folium, numpy, pandas, requests, xgboost
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

def load_model(model_file='model.pkl'):
    """Load the trained model from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("\nPlease train the model first using:")
            print("python -m experiments.xgboost.train_model")
            sys.exit(1)
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
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
    # Build the URL with query parameters
    variables_param = "&".join([f"variables={v}" for v in bio_variables])
    url = f"http://localhost:8000/api/v1/environmental/extract-batch?{variables_param}"
    
    payload = {"coordinates": coordinates}
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching environmental data batch for {len(coordinates)} coordinates...")
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                
                # Handle errors with retry
                if response.status == 429:
                    print(f"Rate limit hit (429). Retrying (attempt {attempt+1}/{max_retries})")
                else:
                    error_text = await response.text()
                    print(f"Error fetching environmental data: {response.status}\n{error_text}")
                
        except Exception as e:
            print(f"Error making API request: {e}")
        
        # Retry logic with exponential backoff
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(delay)
    
    print(f"Failed to fetch data after {max_retries} attempts")
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
        for i in range(0, len(coordinates), BATCH_SIZE):
            batch = coordinates[i:i+BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (len(coordinates) + BATCH_SIZE - 1)//BATCH_SIZE
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} coordinates)")
            
            # Implement rate limiting
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < min_delay_between_batches:
                delay = min_delay_between_batches - time_since_last + random.uniform(0.1, 0.5)  # Add jitter
                print(f"Rate limiting: waiting {delay:.2f}s before next batch")
                await asyncio.sleep(delay)
            
            last_request_time = time.time()
            
            # Fetch data with retry logic
            batch_result = await fetch_environmental_data_batch(session, batch, bio_variables)
            
            if batch_result and "results" in batch_result:
                all_results.extend(batch_result["results"])
                print(f"✓ Successfully processed batch {batch_num}/{total_batches}")
            else:
                print(f"✗ Warning: Failed to fetch data for batch {batch_num}/{total_batches}")
                # Create empty results to maintain grid structure
                for _ in range(len(batch)):
                    all_results.append({})
    
    # Process the results and add to dataframe
    print("Processing fetched environmental data...")
    
    # Create mapping of coordinates to row indexes
    coord_map = {(row['latitude'], row['longitude']): idx for idx, row in env_data.iterrows()}
    
    # Fill in real data
    real_data_count = 0
    for result in all_results:
        if not result or result.get('latitude') is None or result.get('longitude') is None:
            continue
            
        key = (result['latitude'], result['longitude'])
        idx = coord_map.get(key)
        if idx is None:
            continue
        
        # Add elevation
        if 'elevation' in result:
            env_data.loc[idx, 'elevation'] = result['elevation']
        
        # API returns just the numbers as keys (not bio1)
        for bio_var in bio_variables:
            bio_key = f"bio{bio_var}"
            value = result.get(str(bio_var)) or result.get(bio_key)
            if value is not None:
                env_data.loc[idx, bio_key] = value
                
        # Debug first result
        if real_data_count == 0:
            print(f"Sample API response keys: {list(result.keys())}")
            print(f"Converting keys like '{bio_variables[0]}' to 'bio{bio_variables[0]}'")
                
        real_data_count += 1
    
    print(f"Successfully filled {real_data_count}/{len(env_data)} grid points with real data")
    
    # Calculate sin/cos of month for cyclical representation
    env_data['sin_month'] = np.sin(2 * pi * (env_data['month'] - 1) / 12)
    env_data['cos_month'] = np.cos(2 * pi * (env_data['month'] - 1) / 12)
    
    # Fill any missing values with reasonable defaults (better than crashing)
    # Note: NaNs here reflect real data gaps (e.g., WorldClim nodata over water/tile edges or transient fetch errors).
    # We use conservative defaults so the grid can be rendered consistently.
    fill_values = {
        'elevation': 500,
        'bio1': 1.6,
        'bio4': 35,
        'bio5': 2.6,
        'bio6': 0.6,
        'bio12': 750,
        'bio13': 120,
        'bio14': 20,
        'bio15': 55
    }
    
    # Count and fill missing values
    missing_count = env_data.isnull().sum()
    if missing_count.sum() > 0:
        print(f"Missing values before filling:")
        for col, count in missing_count[missing_count > 0].items():
            print(f"  {col}: {count} missing values")
    
        # Fill missing values
        for col, fill_val in fill_values.items():
            if col in env_data.columns:
                missing_col = env_data[col].isnull().sum()
                if missing_col > 0:
                    print(f"Filling {missing_col} missing values in {col}")
                    env_data[col] = env_data[col].fillna(fill_val)
    
    print("Environmental grid with REAL data created successfully")
    return env_data, lat_grid, lon_grid

def predict_invasion_risk(model, env_data):
    """Predict invasion risk using the trained model."""
    print("Predicting invasion risk...")
    
    # Extract features in the same order as used for training
    feature_columns = [
        'latitude', 'longitude', 'elevation', 
        'bio1', 'bio4', 'bio5', 'bio6', 'bio12', 'bio13', 'bio14', 'bio15',
        'sin_month', 'cos_month'
    ]
    
    # Ensure all columns exist
    missing_cols = [col for col in feature_columns if col not in env_data.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {env_data.columns.tolist()}")
        for col in missing_cols:
            env_data[col] = np.nan
    
    X = env_data[feature_columns]
    
    # Get probability predictions (class 1 = presence)
    y_prob = model.predict_proba(X)[:, 1]
    
    return y_prob

def create_heatmap(lats, lons, risk_scores, lat_grid, lon_grid, month, output_file):
    """Create an interactive choropleth grid map of invasion risk."""
    print(f"Creating grid-based visualization for month {month}...")
    
    # Get month name
    month_name = datetime.date(2000, month, 1).strftime('%B')
    
    # Calculate the center point of the map
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Create a map
    risk_map = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                        tiles='CartoDB positron')
    
    # Add Folium plugins for better visualization
    Fullscreen().add_to(risk_map)
    MeasureControl(position='topright', primary_length_unit='kilometers').add_to(risk_map)
    
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
            <span style="font-size:14px">{month_name} | XGBoost Model | REAL API DATA</span>
        </h3>
    '''
    folium.Element(title_html).add_to(risk_map)
    
    # Define color function for risk levels with high granularity
    def get_color(risk):
        """
        Enhanced color mapping with fine-grained intervals for better visualization.
        
        Color Scheme:
        - 0-20%: Blue shades (very low risk)
        - 20-40%: Green to Lime (low risk)
        - 40-60%: Yellow shades (moderate risk)
        - 60-80%: Orange shades (high risk)
        - 80-100%: Red shades with 1% intervals (very high risk)
        """
        # CRITICAL RISK ZONE (80-100%): 1% intervals with distinct reds
        if risk >= 0.99:
            return '#660000'  # Darkest Red - 99-100%
        elif risk >= 0.98:
            return '#750000'  # Very Dark Red - 98-99%
        elif risk >= 0.97:
            return '#850000'  # Deep Dark Red - 97-98%
        elif risk >= 0.96:
            return '#940000'  # Dark Maroon - 96-97%
        elif risk >= 0.95:
            return '#A30000'  # Maroon - 95-96%
        elif risk >= 0.94:
            return '#B20000'  # Deep Red - 94-95%
        elif risk >= 0.93:
            return '#C10000'  # Blood Red - 93-94%
        elif risk >= 0.92:
            return '#D00000'  # Dark Red - 92-93%
        elif risk >= 0.91:
            return '#E00000'  # Red - 91-92%
        elif risk >= 0.90:
            return '#F00000'  # Pure Red - 90-91%
        elif risk >= 0.89:
            return '#FF0000'  # Bright Red - 89-90%
        elif risk >= 0.88:
            return '#FF0F0F'  # Light Red 1 - 88-89%
        elif risk >= 0.87:
            return '#FF1E1E'  # Light Red 2 - 87-88%
        elif risk >= 0.86:
            return '#FF2D2D'  # Light Red 3 - 86-87%
        elif risk >= 0.85:
            return '#FF3C3C'  # Pink Red - 85-86%
        elif risk >= 0.84:
            return '#FF4B00'  # Red-Orange 1 - 84-85%
        elif risk >= 0.83:
            return '#FF5500'  # Red-Orange 2 - 83-84%
        elif risk >= 0.82:
            return '#FF6000'  # Red-Orange 3 - 82-83%
        elif risk >= 0.81:
            return '#FF6A00'  # Red-Orange 4 - 81-82%
        elif risk >= 0.80:
            return '#FF7500'  # Orange-Red - 80-81%
        
        # HIGH RISK ZONE (60-80%): 2% intervals with orange spectrum
        elif risk >= 0.78:
            return '#FF8000'  # Dark Orange 1 - 78-80%
        elif risk >= 0.76:
            return '#FF8A00'  # Dark Orange 2 - 76-78%
        elif risk >= 0.74:
            return '#FF9500'  # Orange 1 - 74-76%
        elif risk >= 0.72:
            return '#FF9F00'  # Orange 2 - 72-74%
        elif risk >= 0.70:
            return '#FFA500'  # Pure Orange - 70-72%
        elif risk >= 0.68:
            return '#FFB000'  # Light Orange 1 - 68-70%
        elif risk >= 0.66:
            return '#FFBA00'  # Light Orange 2 - 66-68%
        elif risk >= 0.64:
            return '#FFC500'  # Orange-Yellow 1 - 64-66%
        elif risk >= 0.62:
            return '#FFCF00'  # Orange-Yellow 2 - 62-64%
        elif risk >= 0.60:
            return '#FFD700'  # Gold - 60-62%
        
        # MODERATE RISK ZONE (40-60%): 2% intervals with yellow spectrum
        elif risk >= 0.58:
            return '#FFE000'  # Deep Yellow 1 - 58-60%
        elif risk >= 0.56:
            return '#FFE800'  # Deep Yellow 2 - 56-58%
        elif risk >= 0.54:
            return '#FFF000'  # Yellow 1 - 54-56%
        elif risk >= 0.52:
            return '#FFF800'  # Yellow 2 - 52-54%
        elif risk >= 0.50:
            return '#FFFF00'  # Pure Yellow - 50-52%
        elif risk >= 0.48:
            return '#F8FF00'  # Light Yellow 1 - 48-50%
        elif risk >= 0.46:
            return '#F0FF00'  # Light Yellow 2 - 46-48%
        elif risk >= 0.44:
            return '#E8FF00'  # Yellow-Lime 1 - 44-46%
        elif risk >= 0.42:
            return '#E0FF00'  # Yellow-Lime 2 - 42-44%
        elif risk >= 0.40:
            return '#D8FF00'  # Lime-Yellow - 40-42%
        
        # LOW RISK ZONE (20-40%): 2% intervals with lime to green
        elif risk >= 0.38:
            return '#CCFF00'  # Lime 1 - 38-40%
        elif risk >= 0.36:
            return '#BFFF00'  # Lime 2 - 36-38%
        elif risk >= 0.34:
            return '#B3FF00'  # Lime 3 - 34-36%
        elif risk >= 0.32:
            return '#A6FF00'  # Lime 4 - 32-34%
        elif risk >= 0.30:
            return '#99FF00'  # Yellow-Green - 30-32%
        elif risk >= 0.28:
            return '#80FF00'  # Light Green 1 - 28-30%
        elif risk >= 0.26:
            return '#66FF00'  # Light Green 2 - 26-28%
        elif risk >= 0.24:
            return '#4DFF00'  # Green-Lime 1 - 24-26%
        elif risk >= 0.22:
            return '#33FF00'  # Green-Lime 2 - 22-24%
        elif risk >= 0.20:
            return '#1AFF1A'  # Bright Green - 20-22%
        
        # VERY LOW RISK ZONE (0-20%): 2% intervals with green to blue
        elif risk >= 0.18:
            return '#00FF33'  # Green 1 - 18-20%
        elif risk >= 0.16:
            return '#00FF4D'  # Green 2 - 16-18%
        elif risk >= 0.14:
            return '#00FF66'  # Green 3 - 14-16%
        elif risk >= 0.12:
            return '#00FF80'  # Green-Cyan 1 - 12-14%
        elif risk >= 0.10:
            return '#00FF99'  # Green-Cyan 2 - 10-12%
        elif risk >= 0.08:
            return '#00FFB3'  # Cyan-Green 1 - 8-10%
        elif risk >= 0.06:
            return '#00FFCC'  # Cyan-Green 2 - 6-8%
        elif risk >= 0.04:
            return '#00FFE6'  # Light Cyan 1 - 4-6%
        elif risk >= 0.02:
            return '#00FFFF'  # Pure Cyan - 2-4%
        else:
            return '#00E6FF'  # Bright Cyan - 0-2%
    
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
    
    # We need the original 2D grids to create proper grid cells
    risk_2d = risk_scores.reshape(lat_grid.shape)
    
    # Grid dimensions
    grid_height = lat_grid.shape[0]
    grid_width = lat_grid.shape[1]
    
    # Create grid cells as GeoJSON features
    for i in range(grid_height - 1):
        for j in range(grid_width - 1):
            # Create a polygon for each grid cell
            top_left = [lat_grid[i, j], lon_grid[i, j]]
            top_right = [lat_grid[i, j+1], lon_grid[i, j+1]]
            bottom_right = [lat_grid[i+1, j+1], lon_grid[i+1, j+1]]
            bottom_left = [lat_grid[i+1, j], lon_grid[i+1, j]]
            
            risk_value = risk_2d[i, j]
            
            # Create the polygon coordinates
            polygon = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [top_left[1], top_left[0]],  # GeoJSON uses [lon, lat]
                        [top_right[1], top_right[0]],
                        [bottom_right[1], bottom_right[0]],
                        [bottom_left[1], bottom_left[0]],
                        [top_left[1], top_left[0]]  # Close the polygon
                    ]]
                },
                'properties': {
                    'risk': float(risk_value),
                    'risk_percent': f"{float(risk_value) * 100:.1f}%",
                    'lat': float(top_left[0]),
                    'lon': float(top_left[1])
                }
            }
            features.append(polygon)
    
    # Create the GeoJSON data
    geojson_data = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Add the choropleth layer
    choropleth = folium.GeoJson(
        geojson_data,
        style_function=style_function,
        name='Invasion Risk Grid',
        tooltip=folium.GeoJsonTooltip(
            fields=['risk_percent', 'lat', 'lon'],
            aliases=['Risk:', 'Lat:', 'Lon:'],
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
        ),
    )
    
    # Add the choropleth to the map
    choropleth.add_to(risk_map)
    
    # Add enhanced color scale legend with more granular colors
    colormap = LinearColormap(
        colors=[
            '#00E6FF',  # Bright Cyan (0%)
            '#00FF99',  # Green-Cyan (10%)
            '#1AFF1A',  # Bright Green (20%)
            '#99FF00',  # Yellow-Green (30%)
            '#D8FF00',  # Lime-Yellow (40%)
            '#FFFF00',  # Pure Yellow (50%)
            '#FFD700',  # Gold (60%)
            '#FFA500',  # Pure Orange (70%)
            '#FF7500',  # Orange-Red (80%)
            '#FF0000',  # Bright Red (90%)
            '#660000'   # Darkest Red (100%)
        ],
        vmin=0, vmax=1,
        caption='Invasion Risk Probability (0% - 100%)',
        index=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    risk_map.add_child(colormap)
    
    # Create marker cluster for observation points (unused but available for extension)
    marker_cluster = MarkerCluster(name='Observation Points').add_to(risk_map)
    
    # Create a statistics panel
    stats_html = f'''
        <div style="position: fixed; 
                    bottom: 10px; left: 10px; width: 250px;
                    background-color: white; z-index:9999; font-size:12px;
                    padding: 10px; border-radius: 5px; border: 1px solid grey;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0;">Invasion Risk Statistics</h4>
            <table style="width:100%">
                <tr><td><b>Region:</b></td><td>{min(lats):.1f}° to {max(lats):.1f}°S, {min(lons):.1f}° to {max(lons):.1f}°E</td></tr>
                <tr><td><b>Month:</b></td><td>{month_name}</td></tr>
                <tr><td><b>Grid Points:</b></td><td>{len(risk_scores)}</td></tr>
                <tr><td><b>Mean Risk:</b></td><td>{np.mean(risk_scores):.3f}</td></tr>
                <tr><td><b>Maximum Risk:</b></td><td>{np.max(risk_scores):.3f}</td></tr>
                <tr><td><b>High Risk (>0.7):</b></td><td>{(risk_scores > 0.7).sum()} locations</td></tr>
                <tr><td><b>Med Risk (0.4-0.7):</b></td><td>{((risk_scores > 0.4) & (risk_scores <= 0.7)).sum()} locations</td></tr>
                <tr><td><b>Model:</b></td><td>XGBoost with REAL DATA</td></tr>
            </table>
            <div style="font-size:10px; margin-top:5px; text-align:right;">
                Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>
    '''
    folium.Element(stats_html).add_to(risk_map)
    
    # Add layer control
    folium.LayerControl().add_to(risk_map)
    
    # Save the map
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"xgboost_invasion_prediction_REAL_API_DATA_{timestamp}.html"
    
    output_path = os.path.join(MODEL_DIR, output_file)
    risk_map.save(output_path)
    print(f"Grid map saved to {output_path}")
    
    return output_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Pyracantha invasion risk grid map using XGBoost with REAL API DATA.')
    
    parser.add_argument('--grid_size', type=int, default=20,
                        help='Size of the grid (number of points in each dimension). Higher values give more detail but take longer. Default is 20.')
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
    print("Starting invasion risk grid map generation pipeline with REAL API DATA and XGBoost model...")
    print("NOTE: This script requires the FastAPI server to be running.")
    print("      Start it with: python -m uvicorn app.main:app --reload")
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the trained model
    model = load_model()
    
    # Create a grid of environmental data with REAL data from APIs
    env_data, lat_grid, lon_grid = await create_environmental_grid_with_real_data(
        args.lat_min, args.lat_max, 
        args.lon_min, args.lon_max, 
        args.grid_size, 
        args.month
    )
    
    # Predict invasion risk
    risk_scores = predict_invasion_risk(model, env_data)
    
    # Create and save the heatmap
    output_path = create_heatmap(
        env_data['latitude'], env_data['longitude'], 
        risk_scores, lat_grid, lon_grid,
        args.month, args.output_file
    )
    
    print(f"Grid map generation complete! View the result at: {output_path}")

def main():
    """Entry point for the script."""
    asyncio.run(main_async())

if __name__ == '__main__':
    main()
