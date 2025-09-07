"""
Enhanced Grid-Based Heatmap Generator for Pyracantha Invasion Risk Prediction

This enhanced script extends the original generate_heatmap_api.py with:
1. Higher resolution grid support for more detailed mapping
2. Multi-month seasonal comparisons with side-by-side visualizations
3. Known invasion site overlay capabilities 
4. Interactive controls and layered visualization

Usage:
    python -m experiments.ensemble.generate_heatmap_enhanced [options]

Requirements:
    - FastAPI server running on localhost:8000 (uvicorn app.main:app)
    - Trained Ensemble model (model.pkl) and optimal_threshold.pkl
    - Python packages: folium, numpy, pandas, requests, scikit-learn, xgboost
    - Known invasion sites CSV (optional)
"""

import os
import sys
import argparse
import datetime
import pickle
import numpy as np
import pandas as pd
# Set pandas option to avoid downcasting warnings
pd.set_option('future.no_silent_downcasting', True)
from math import pi
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster, MeasureControl, Fullscreen, Draw
from folium.plugins import FloatImage, TimestampedGeoJson
from branca.colormap import LinearColormap
from branca.element import Figure, MacroElement
from jinja2 import Template
import aiohttp
import asyncio
import json
import time
import random
import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import product

# Set pandas option to disable silent downcasting warning
pd.set_option('future.no_silent_downcasting', True)

# Import the dynamic known sites generator
from experiments.ensemble.generate_known_sites import generate_known_sites

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Directory where model files are stored
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Default known invasion sites file location
DEFAULT_KNOWN_SITES = os.path.join(os.path.dirname(__file__), 'known_invasion_sites.csv')

def load_model(model_file='model.pkl', threshold_file='optimal_threshold.pkl'):
    """Load the trained ensemble model and optimal threshold from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, model_file)
        threshold_path = os.path.join(MODEL_DIR, threshold_file)
        
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        
        print(f"Using optimal threshold: {threshold:.4f}")
        print(f"Model loaded successfully: {type(model).__name__}")
        
        return model, threshold
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

async def fetch_environmental_data_batch(session, coordinates, variables, max_retries=5):
    """
    Fetch real environmental data from the API endpoint in batches with exponential backoff retry.
    
    Args:
        session: aiohttp ClientSession
        coordinates: List of coordinate dicts
        variables: List of bioclimate variables
        max_retries: Maximum number of retry attempts
    """
    # Convert variables to format expected by API
    variables_str = [str(v) for v in variables]
    
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
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue  # Try again
        
        except Exception as e:
            print(f"Error making environmental API request: {e}")
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(delay)
            continue
    
    print(f"Failed to fetch environmental data after {max_retries} attempts")
    return None

async def fetch_elevation_batch(session, coordinates, max_retries=5):
    """
    Fetch real elevation data from the API endpoint.
    
    Args:
        session: aiohttp ClientSession
        coordinates: List of coordinate dicts
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dictionary with elevation results or None if failed
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
                else:
                    error_text = await response.text()
                    print(f"Error fetching elevation data: {response.status}\n{error_text}")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
        
        except Exception as e:
            print(f"Error making elevation API request: {e}")
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(delay)
            continue
    
    print(f"Failed to fetch elevation data after {max_retries} attempts")
    return None

def create_grid(lat_min, lat_max, lon_min, lon_max, grid_size, month):
    """
    Create a grid of points for environmental data extraction.
    
    Args:
        lat_min, lat_max: Latitude range
        lon_min, lon_max: Longitude range
        grid_size: Number of points along each dimension
        month: Month of the year (1-12)
        
    Returns:
        DataFrame with grid coordinates and temporal features
    """
    # Create latitude and longitude arrays
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    
    # Create all combinations
    points = []
    for lat in lat_grid:
        for lon in lon_grid:
            points.append({'latitude': lat, 'longitude': lon})
    
    # Create DataFrame
    grid_df = pd.DataFrame(points)
    
    # Add temporal features
    day_of_year = int(datetime.date(2020, month, 15).strftime('%j'))  # Use middle of month
    grid_df['month'] = month
    grid_df['day_of_year'] = day_of_year
    
    # Calculate circular features for month to avoid December-January discontinuity
    grid_df['sin_month'] = np.sin(2 * pi * month / 12)
    grid_df['cos_month'] = np.cos(2 * pi * month / 12)
    
    return grid_df, lat_grid, lon_grid

async def create_environmental_grid(lat_min, lat_max, lon_min, lon_max, grid_size, month, batch_size=20):
    """
    Create an environmental data grid by fetching real data from API endpoints.
    
    Args:
        lat_min, lat_max: Latitude range
        lon_min, lon_max: Longitude range
        grid_size: Number of points along each dimension
        month: Month of the year (1-12)
        batch_size: Number of coordinates to process in each API call
        
    Returns:
        DataFrame with grid coordinates and environmental data
    """
    print(f"Creating environmental grid with REAL data for area: Lat {lat_min} to {lat_max}, Lon {lon_min} to {lon_max}")
    
    # Create initial grid with coordinates and time features
    env_data, lat_grid, lon_grid = create_grid(lat_min, lat_max, lon_min, lon_max, grid_size, month)
    
    # Prepare coordinates in the format expected by the API
    coordinates = []
    for _, row in env_data.iterrows():
        coordinates.append({
            "latitude": row['latitude'],
            "longitude": row['longitude']
        })
    
    # WorldClim bioclimatic variables to fetch
    bio_variables = [1, 4, 5, 6, 12, 13, 14, 15]
    
    # Split coordinates into batches to avoid overwhelming the API
    coordinate_batches = [coordinates[i:i+batch_size] for i in range(0, len(coordinates), batch_size)]
    
    # Rate limiting settings
    min_delay_between_batches = 1.0  # Minimum seconds between API calls
    
    # Create an aiohttp session for all API calls
    async with aiohttp.ClientSession() as session:
        # First, get bioclimatic variables
        all_results = []
        total_batches = len(coordinate_batches)
        last_request_time = time.time() - min_delay_between_batches  # Allow first request immediately
        
        for batch_num, batch in enumerate(coordinate_batches, 1):
            # Rate limiting
            time_since_last = time.time() - last_request_time
            if time_since_last < min_delay_between_batches:
                delay = min_delay_between_batches - time_since_last + random.uniform(0.1, 0.5)  # Add jitter
                print(f"Processing environmental batch {batch_num}/{total_batches} ({len(batch)} coordinates)")
                print(f"Rate limiting: waiting {delay:.2f}s between batches")
                await asyncio.sleep(delay)
            
            last_request_time = time.time()
            
            # Fetch data with retry logic
            batch_result = await fetch_environmental_data_batch(session, batch, bio_variables)
            
            if batch_result and "results" in batch_result:
                all_results.extend(batch_result["results"])
            else:
                print(f"✗ Warning: Failed to fetch data for batch {batch_num}/{total_batches}")
                # Create empty results to maintain grid structure
                for _ in range(len(batch)):
                    all_results.append({})
        
        # Process the environmental results
        for i, result in enumerate(all_results):
            for bio_var in bio_variables:
                # API returns just the numbers as keys (not bio1)
                if str(bio_var) in result:
                    env_data.loc[i, f'bio{bio_var}'] = result.get(str(bio_var))
        
        # Now, get elevation data
        elevation_results = []
        last_request_time = time.time() - min_delay_between_batches  # Allow first request immediately
        
        for batch_num, batch in enumerate(coordinate_batches, 1):
            # Rate limiting
            time_since_last = time.time() - last_request_time
            if time_since_last < min_delay_between_batches:
                delay = min_delay_between_batches - time_since_last + random.uniform(0.1, 0.5)  # Add jitter
                print(f"Processing elevation batch {batch_num}/{total_batches} ({len(batch)} coordinates)")
                print(f"Rate limiting: waiting {delay:.2f}s between batches")
                await asyncio.sleep(delay)
            
            last_request_time = time.time()
            
            # Fetch elevation data with retry logic
            batch_results = await fetch_elevation_batch(session, batch)
            
            if batch_results and "results" in batch_results:
                elevation_results.extend(batch_results.get('results', []))
            else:
                print(f"✗ Warning: Failed to fetch elevation for batch {batch_num}/{total_batches}")
                # Add empty results to maintain grid structure
                for _ in range(len(batch)):
                    elevation_results.append({})
    
    # Process the elevation results
    for i, result in enumerate(elevation_results):
        if result and 'elevation' in result:
            env_data.loc[i, 'elevation'] = result.get('elevation')
    
    # Create derived features needed by the model
    try:
        # Distance from geographic center (South Africa - Western Cape approx)
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
                    fill_value = defaults.get(col, 0)
                    env_data = env_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                    # Fill NA values using the recommended approach
                    env_data[col] = env_data[col].fillna(fill_value)
                    print(f"    All values missing for '{col}', using default: {fill_value}")
                else:
                    env_data = env_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                    # Fill NA values using the recommended approach
                    env_data[col] = env_data[col].fillna(col_mean)
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
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Create missing columns with default values
        for col in missing_cols:
            print(f"Creating missing column: {col}")
            df[col] = 0
    
    X = df[feature_columns]
    
    # Get probability predictions (class 1 = presence)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Apply optimal threshold if available
    if threshold != 0.5:
        print(f"Applying optimal threshold: {threshold:.4f}")
    
    return y_prob

async def load_known_sites(file_path=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """
    Load known invasion sites from a CSV file or generate dynamically.
    Expected format: latitude,longitude,name,discovery_date,severity
    
    Args:
        file_path: Path to the CSV file with known sites
        lat_min, lat_max, lon_min, lon_max: Area boundaries for dynamic generation
        
    Returns:
        DataFrame with known sites or empty DataFrame if file not found
    """
    # If a specific file is provided, use it
    if file_path and os.path.exists(file_path):
        try:
            print(f"Loading known invasion sites from {file_path}...")
            sites = pd.read_csv(file_path)
            print(f"Loaded {len(sites)} known invasion sites")
            return sites
        except Exception as e:
            print(f"Error loading known sites: {e}")
            return pd.DataFrame(columns=['latitude', 'longitude', 'name', 'discovery_date', 'severity'])
    
    # If no file or file doesn't exist, generate dynamically if we have coordinates
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        try:
            print("No known sites file specified or file not found. Generating real sites dynamically...")
            # Generate a temp file path for the CSV
            temp_csv_path = os.path.join(os.path.dirname(__file__), 
                                        f"temp_known_sites_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Generate the sites using our dynamic generator (async)
            csv_path = await generate_known_sites(
                lat_min=lat_min, lat_max=lat_max, 
                lon_min=lon_min, lon_max=lon_max,
                output_path=temp_csv_path,
                max_sites=15,
                use_training_data=True,
                use_gbif_api=True
            )
            
            # Now load the generated CSV
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                sites = pd.read_csv(csv_path)
                print(f"Dynamically generated {len(sites)} real invasion sites")
                return sites
        except Exception as e:
            print(f"Error generating dynamic known sites: {e}")
    
    # Default fallback
    print("No known sites available. Continuing without known sites.")
    return pd.DataFrame(columns=['latitude', 'longitude', 'name', 'discovery_date', 'severity'])

class SeasonalMapControl(MacroElement):
    """Custom Folium plugin for seasonal comparison control"""
    
    def __init__(self, month_maps, month_names):
        super(SeasonalMapControl, self).__init__()
        self._month_maps = month_maps
        self._month_names = month_names
        
        self._template = Template("""
            {% macro script(this, kwargs) %}
            var seasonalControl = L.control({position: 'topright'});
            seasonalControl.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'info seasonal-control');
                div.innerHTML = '<h4>Seasonal View</h4>';
                div.innerHTML += '<select id="season-selector" style="width:100%;">';
                
                {% for month_name in this._month_names %}
                    div.innerHTML += '<option value="{{ month_name }}">{{ month_name }}</option>';
                {% endfor %}
                
                div.innerHTML += '</select>';
                
                // Add change event listener
                L.DomEvent.addListener(div, 'change', function(e) {
                    var selectedMonth = document.getElementById('season-selector').value;
                    
                    // Hide all month layers
                    {% for month_name in this._month_names %}
                        map.removeLayer({{ this._month_maps[loop.index0] }});
                    {% endfor %}
                    
                    // Show selected month layer
                    var monthIndex = {{ this._month_names }}.indexOf(selectedMonth);
                    if (monthIndex >= 0) {
                        map.addLayer({{ this._month_maps[0] }}[monthIndex]);
                    }
                });
                
                return div;
            };
            seasonalControl.addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)

def create_enhanced_heatmap(lats, lons, risk_scores_dict, lat_grid, lon_grid, months, known_sites, output_file):
    """
    Create an enhanced interactive heatmap with seasonal comparison and known invasion sites.
    
    Args:
        lats, lons: Coordinate arrays
        risk_scores_dict: Dictionary mapping months to risk scores arrays
        lat_grid, lon_grid: Grid coordinate arrays
        months: List of months included in the visualization
        known_sites: DataFrame with known invasion sites
        output_file: Output HTML file path
    """
    print(f"Creating enhanced visualization for {len(months)} months...")
    
    # Calculate map center
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Calculate bounds for display
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Create the figure to hold the map
    fig = Figure(width='100%', height='100%')
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB positron',
        control_scale=True
    )
    fig.add_child(m)
    
    # Add Folium plugins for better visualization
    Fullscreen().add_to(m)
    MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    Draw(export=True).add_to(m)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Add terrain layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Terrain',
        overlay=False
    ).add_to(m)
    
    # Create color map for risk scores
    risk_colormap = LinearColormap(
        colors=['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred'],
        vmin=0, vmax=1,
        caption='Invasion Risk Score'
    )
    
    # Create month layer groups
    month_layers = []
    month_names = []
    month_feature_groups = []
    
    # Process each month's data
    for month in months:
        month_name = datetime.date(2020, month, 1).strftime('%B')
        month_names.append(month_name)
        
        # Create a feature group for this month's data
        fg = folium.FeatureGroup(name=f"Risk - {month_name}", show=month == months[0])
        month_feature_groups.append(fg)
        
        # Get risk scores for this month
        risk_scores = risk_scores_dict.get(month, [])
        if len(risk_scores) == 0:
            continue
            
        # Normalize risk scores for better visualization
        max_risk = max(risk_scores)
        if max_risk > 0:
            normalized_scores = risk_scores / max_risk
        else:
            normalized_scores = risk_scores
        
        # Generate grid cells for this month
        grid_data = []
        index = 0
        
        # Create grid with colored cells
        for i in range(len(lat_grid)):
            for j in range(len(lon_grid)):
                lat = lat_grid[i]
                lon = lon_grid[j]
                risk = risk_scores[index]
                
                # Only include significant risk values
                if risk > 0.01:
                    # Create polygon for grid cell
                    if i < len(lat_grid) - 1 and j < len(lon_grid) - 1:
                        # Define cell corners
                        cell_bounds = [
                            [lat_grid[i], lon_grid[j]],
                            [lat_grid[i], lon_grid[j+1]],
                            [lat_grid[i+1], lon_grid[j+1]],
                            [lat_grid[i+1], lon_grid[j]],
                        ]
                        
                        # Add cell polygon
                        folium.Polygon(
                            locations=cell_bounds,
                            color='gray',
                            weight=1,
                            fill=True,
                            fill_color=risk_colormap(risk),
                            fill_opacity=0.7,
                            popup=f"Risk: {risk:.2f}",
                            tooltip=f"Risk: {risk:.2f}"
                        ).add_to(fg)
                    
                    # For high-risk areas, add marker
                    if risk > 0.7:
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            color='black',
                            fill=True,
                            fill_color=risk_colormap(risk),
                            fill_opacity=0.9,
                            popup=f"<b>High Risk:</b> {risk:.2f}",
                        ).add_to(fg)
                
                index += 1
        
        # Add this month's feature group to the map
        fg.add_to(m)
        month_layers.append(fg)
    
    # Add known invasion sites if available
    if not known_sites.empty:
        # Create a feature group for known sites
        known_sites_fg = folium.FeatureGroup(name="Known Invasion Sites")
        
        # Create a marker cluster for the sites
        marker_cluster = MarkerCluster(name="Known Invasion Sites Cluster")
        
        for _, site in known_sites.iterrows():
            # Define marker color based on severity (if available)
            severity = site.get('severity', 'medium')
            if severity in ['high', 'severe', 'critical']:
                icon_color = 'red'
                icon_type = 'exclamation-circle'
            elif severity in ['medium', 'moderate']:
                icon_color = 'orange'
                icon_type = 'exclamation'
            else:
                icon_color = 'green'
                icon_type = 'info'
            
            # Create popup content
            popup_content = f"""
                <h4>{site.get('name', 'Invasion Site')}</h4>
                <table>
                    <tr><td>Location:</td><td>{site['latitude']:.4f}, {site['longitude']:.4f}</td></tr>
                    <tr><td>Discovered:</td><td>{site.get('discovery_date', 'Unknown')}</td></tr>
                    <tr><td>Severity:</td><td>{severity.title()}</td></tr>
                </table>
            """
            
            # Add marker to cluster - ensure coordinates are not NaN
            if not pd.isna(site['latitude']) and not pd.isna(site['longitude']):
                folium.Marker(
                    location=[float(site['latitude']), float(site['longitude'])],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa'),
                    tooltip=site.get('name', 'Invasion Site')
                ).add_to(marker_cluster)
        
        # Add markers to the feature group
        marker_cluster.add_to(known_sites_fg)
        known_sites_fg.add_to(m)
    
    # Add the colormap to the map
    risk_colormap.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a title and information box
    title_html = f"""
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 520px; height: auto; 
                    background-color: white; border:2px solid black; z-index:9999; 
                    font-size:14px; padding: 8px;
                    border-radius: 5px; opacity: 0.9">
            <h3 style="margin-top: 0; margin-bottom: 5px;">Enhanced Pyracantha Invasion Risk Map</h3>
            <table style="width: 100%;">
                <tr><td><b>Area:</b></td><td>{lat_min:.4f}°S to {lat_max:.4f}°S, {lon_min:.4f}°E to {lon_max:.4f}°E</td></tr>
                <tr><td><b>Months:</b></td><td>{', '.join(month_names)}</td></tr>
                <tr><td><b>Model:</b></td><td>Ensemble (XGBoost + Random Forest)</td></tr>
                <tr><td><b>Grid Size:</b></td><td>{len(lat_grid)}x{len(lon_grid)} ({len(lat_grid)*len(lon_grid)} points)</td></tr>
                <tr><td><b>Known Sites:</b></td><td>{len(known_sites) if not known_sites.empty else 0}</td></tr>
                <tr><td><b>Date:</b></td><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</td></tr>
            </table>
            <div style="font-size:11px; margin-top:5px;">
                <b>Instructions:</b> Use the layer control (top right) to toggle between months and add/remove known sites.
            </div>
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add custom seasonal control
    # SeasonalMapControl(month_layers, month_names).add_to(m)
    
    # Add a legend for known sites
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px;
                border: 2px solid grey; z-index: 9999; 
                background-color: white; padding: 10px;
                font-size: 14px; border-radius: 5px;">
        <h4 style="margin-top: 0;">Legend</h4>
        <div><i class="fa fa-exclamation-circle" style="color: red"></i> High Severity</div>
        <div><i class="fa fa-exclamation" style="color: orange"></i> Medium Severity</div>
        <div><i class="fa fa-info" style="color: green"></i> Low Severity</div>
        <div style="margin-top: 5px; border-top: 1px solid #ddd; padding-top: 5px;">
            <i style="background: red; width: 15px; height: 15px; display: inline-block;"></i> High Risk
        </div>
        <div>
            <i style="background: yellow; width: 15px; height: 15px; display: inline-block;"></i> Medium Risk
        </div>
        <div>
            <i style="background: green; width: 15px; height: 15px; display: inline-block;"></i> Low Risk
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not output_file:
        output_file = os.path.join(os.path.dirname(__file__), 
                                  f"enhanced_risk_map_{timestamp}.html")
    
    m.save(output_file)
    print(f"Enhanced visualization saved to {output_file}")
    
    return output_file

async def process_month(lat_min, lat_max, lon_min, lon_max, grid_size, month, batch_size, model, threshold):
    """Process a single month's data for multi-month comparison."""
    print(f"\nProcessing month {month} ({datetime.date(2020, month, 1).strftime('%B')})...")
    
    # Create environmental grid for this month
    env_data, lat_grid, lon_grid = await create_environmental_grid(
        lat_min, lat_max, lon_min, lon_max, grid_size, month, batch_size
    )
    
    # Predict invasion risk
    risk_scores = predict_invasion_risk(model, threshold, env_data)
    
    return month, risk_scores, lat_grid, lon_grid, env_data['latitude'].values, env_data['longitude'].values

async def main_async():
    """Main async function with command line arguments support."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate enhanced Pyracantha invasion risk heatmap using real API data.')
    parser.add_argument('--lat_min', type=float, default=-34.2, help='Minimum latitude')
    parser.add_argument('--lat_max', type=float, default=-33.8, help='Maximum latitude')
    parser.add_argument('--lon_min', type=float, default=18.2, help='Minimum longitude')
    parser.add_argument('--lon_max', type=float, default=19.0, help='Maximum longitude')
    parser.add_argument('--grid_size', type=int, default=20, help='Grid size (points per dimension)')
    parser.add_argument('--months', type=str, default='3,6,9,12', help='Months to model (comma-separated, 1-12)')
    parser.add_argument('--batch_size', type=int, default=20, help='API batch size')
    parser.add_argument('--known_sites', type=str, default=None, help='CSV file with known invasion sites')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Parse months
    try:
        months = [int(m.strip()) for m in args.months.split(',')]
        # Validate months
        for m in months:
            if m < 1 or m > 12:
                print(f"Warning: Invalid month {m}, must be 1-12. Removing from list.")
                months.remove(m)
        
        if not months:
            print("No valid months specified, defaulting to March (3).")
            months = [3]
    except:
        print("Error parsing months, defaulting to March (3).")
        months = [3]
    
    # Print header
    print("Starting ENHANCED invasion risk grid map generation with REAL API DATA...")
    print(f"Area: Lat {args.lat_min} to {args.lat_max}, Lon {args.lon_min} to {args.lon_max}")
    print(f"Grid size: {args.grid_size}x{args.grid_size} ({args.grid_size**2} points)")
    print(f"Months: {', '.join([datetime.date(2020, m, 1).strftime('%B') for m in months])}")
    print("NOTE: This script requires the FastAPI server to be running.")
    print("      Start it with: python -m uvicorn app.main:app --reload\n")
    
    # Load model and threshold
    model, threshold = load_model()
    
    # Load known invasion sites if available, or generate dynamically (async)
    known_sites = await load_known_sites(
        args.known_sites,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max
    )
    
    # Process each month
    tasks = []
    for month in months:
        tasks.append(process_month(
            args.lat_min, args.lat_max, args.lon_min, args.lon_max,
            args.grid_size, month, args.batch_size, model, threshold
        ))
    
    # Run all month processing concurrently
    results = await asyncio.gather(*tasks)
    
    # Organize results by month
    risk_scores_dict = {}
    lats = None
    lons = None
    lat_grid = None
    lon_grid = None
    
    for month, scores, month_lat_grid, month_lon_grid, month_lats, month_lons in results:
        risk_scores_dict[month] = scores
        
        # Use first month's coordinates for visualization
        if lats is None:
            lats = month_lats
            lons = month_lons
            lat_grid = month_lat_grid
            lon_grid = month_lon_grid
    
    # Create enhanced visualization with all months and known sites
    output_file = create_enhanced_heatmap(
        lats, lons, risk_scores_dict, lat_grid, lon_grid, months, known_sites, args.output
    )
    
    print("Enhanced invasion risk map generation complete.")
    print(f"Heatmap saved to {output_file}")
    return output_file

def main():
    """Main entry point with event loop setup."""
    try:
        loop = asyncio.get_event_loop()
    except:
        # For newer Python versions that don't have a current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(main_async())

if __name__ == "__main__":
    main()
