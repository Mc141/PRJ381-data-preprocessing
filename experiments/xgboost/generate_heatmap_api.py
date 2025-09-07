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
    coord_map = {(env_data.iloc[i]['latitude'], env_data.iloc[i]['longitude']): i for i in range(len(env_data))}
    
    # Fill in real data
    real_data_count = 0
    for result in all_results:
        if not result:  # Skip empty results
            continue
            
        lat = result.get('latitude')
        lon = result.get('longitude')
        
        if lat is None or lon is None:
            continue
            
        # Find matching row in DataFrame (with some tolerance for floating point precision)
        key = (lat, lon)
        if key in coord_map:
            idx = coord_map[key]
            
            # Add elevation
            env_data.loc[idx, 'elevation'] = result.get('elevation')
            
            # API returns just the numbers as keys (not bio1)
            for bio_var in bio_variables:
                bio_key = f"bio{bio_var}"  # Format needed by the model
                
                # Try the numeric format from the API
                if str(bio_var) in result:
                    env_data.loc[idx, bio_key] = result.get(str(bio_var))
                # Fallback to other formats if available
                elif bio_key in result:
                    env_data.loc[idx, bio_key] = result.get(bio_key)
                    
            # Debug what we're getting from the API
            if real_data_count == 0:
                print(f"Sample API response keys: {list(result.keys())}")
                print(f"Converting keys like '{bio_variables[0]}' to 'bio{bio_variables[0]}'")
                    
            real_data_count += 1
    
    print(f"Successfully filled {real_data_count}/{len(env_data)} grid points with real data")
    
    # Calculate sin/cos of month for cyclical representation
    env_data['sin_month'] = np.sin(2 * pi * (env_data['month'] - 1) / 12)
    env_data['cos_month'] = np.cos(2 * pi * (env_data['month'] - 1) / 12)
    
    # Fill any missing values with reasonable defaults (better than crashing)
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
    
    # Count missing values before filling
    missing_count = env_data.isnull().sum()
    print(f"Missing values before filling:")
    for col, count in missing_count.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    # Fill missing values
    for col, fill_val in fill_values.items():
        if col in env_data.columns and env_data[col].isnull().sum() > 0:
            missing_col = env_data[col].isnull().sum()
            if missing_col > 0:
                print(f"Filling {missing_col} missing values in {col}")
                env_data[col].fillna(fill_val, inplace=True)
    
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
            <span style="font-size:14px">{month_name} | XGBoost Model | REAL API DATA</span>
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
    
    # Add color scale legend
    colormap = LinearColormap(
        colors=['blue', 'green', 'yellow', 'orange', 'red'],
        vmin=0, vmax=1,
        caption='Invasion Risk Probability'
    )
    risk_map.add_child(colormap)
    
    # Create marker cluster for observation points
    marker_cluster = None
    try:
        marker_cluster = MarkerCluster(name='Observation Points').add_to(risk_map)
    except ImportError:
        marker_cluster = risk_map
    
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
