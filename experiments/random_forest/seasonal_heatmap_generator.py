#!/usr/bin/env python3
"""
Generate Seasonal-Aware Heatmap for Pyracantha angustifolia

This script uses the enhanced seasonal model to generate predictions that
capture flowering/observation patterns and temporal dynamics. The model
understands that Pyracantha is most likely to be observed during autumn
(March-May) when it's flowering/fruiting.

Key Seasonal Features:
- Peak season: Autumn (66% of observations)
- Peak month: April (40% of sightings)
- Secondary peak: May (23% of sightings)
- Weather patterns from actual observation periods
- Recent observations weighted more heavily
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import folium
from matplotlib.colors import LinearSegmentedColormap
import asyncio
import aiohttp
from typing import List, Dict, Tuple
import time

def load_seasonal_model():
    """Load the trained seasonal Random Forest model."""
    print("Loading seasonal model...")
    
    # Load seasonal model
    model_path = Path("outputs/seasonal_pyracantha_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Seasonal model not found. Please run seasonal_predictor.py first.")
    
    model_data = joblib.load(model_path)
    
    # Load original data
    data_path = "../../data/dataset.csv"
    df = pd.read_csv(data_path)
    
    return model_data, df

async def fetch_weather_for_coordinates(session: aiohttp.ClientSession, 
                                       lat: float, lon: float, 
                                       target_date: date = None,
                                       base_url: str = "http://127.0.0.1:8000") -> Dict:
    """Fetch weather data for coordinates with seasonal awareness."""
    if target_date is None:
        # Use peak season date (April 15th) for best predictions
        target_date = date(2024, 4, 15)  # Peak observation month
    
    # Get a full year of data for temporal aggregates
    end_date = target_date
    start_date = date(target_date.year - 1, target_date.month, target_date.day)
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_year": start_date.year,
        "start_month": start_date.month,
        "start_day": start_date.day,
        "end_year": end_date.year,
        "end_month": end_date.month,
        "end_day": end_date.day,
        "store_in_db": "false"
    }
    
    try:
        async with session.get(f"{base_url}/api/v1/weather", params=params, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"API error for ({lat:.3f}, {lon:.3f}): Status {response.status}")
                return None
    except Exception as e:
        print(f"Request failed for ({lat:.3f}, {lon:.3f}): {e}")
        return None

def process_seasonal_weather_data(weather_data: Dict, lat: float, lon: float, 
                                 target_date: date, feature_columns: List[str]) -> Dict:
    """Process weather data into seasonal model features."""
    if not weather_data or 'data' not in weather_data:
        return get_default_seasonal_features(lat, lon, target_date, feature_columns)
    
    daily_data = weather_data['data']
    df_weather = pd.DataFrame(daily_data)
    
    if df_weather.empty:
        return get_default_seasonal_features(lat, lon, target_date, feature_columns)
    
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather = df_weather.sort_values('date')
    
    # Get most recent data point
    target_row = df_weather.iloc[-1]
    
    features = {}
    
    # Basic coordinates
    features['latitude_x'] = lat
    features['longitude_x'] = lon
    features['elevation'] = target_row.get('elevation', 200)
    
    # Daily weather parameters
    features['T2M'] = target_row.get('T2M', 18)
    features['T2M_MAX'] = target_row.get('T2M_MAX', 22)
    features['T2M_MIN'] = target_row.get('T2M_MIN', 14)
    features['PRECTOTCORR'] = target_row.get('PRECTOTCORR', 1.5)
    features['RH2M'] = target_row.get('RH2M', 65)
    features['WS2M'] = target_row.get('WS2M', 3.5)
    features['ALLSKY_SFC_SW_DWN'] = target_row.get('ALLSKY_SFC_SW_DWN', 5)
    features['CLRSKY_SFC_SW_DWN'] = target_row.get('CLRSKY_SFC_SW_DWN', 7)
    features['TQV'] = target_row.get('TQV', 25)
    features['TS'] = target_row.get('TS', 18)
    
    # Temporal aggregates
    last_7_days = df_weather.tail(7)
    last_30_days = df_weather.tail(30)
    last_90_days = df_weather.tail(90)
    
    features['rain_sum_7'] = last_7_days['PRECTOTCORR'].sum()
    features['rain_sum_30'] = last_30_days['PRECTOTCORR'].sum()
    features['rain_sum_90'] = last_90_days['PRECTOTCORR'].sum()
    features['rain_sum_365'] = df_weather['PRECTOTCORR'].sum()
    
    features['t2m_mean_7'] = last_7_days['T2M'].mean()
    features['t2m_mean_30'] = last_30_days['T2M'].mean()
    features['t2m_mean_90'] = last_90_days['T2M'].mean()
    features['t2m_mean_365'] = df_weather['T2M'].mean()
    
    features['rh2m_mean_7'] = last_7_days['RH2M'].mean()
    features['rh2m_mean_30'] = last_30_days['RH2M'].mean()
    features['rh2m_mean_90'] = last_90_days['RH2M'].mean()
    features['rh2m_mean_365'] = df_weather['RH2M'].mean()
    
    features['wind_mean_7'] = last_7_days['WS2M'].mean()
    features['wind_mean_30'] = last_30_days['WS2M'].mean()
    features['wind_mean_90'] = last_90_days['WS2M'].mean()
    features['wind_mean_365'] = df_weather['WS2M'].mean()
    
    # Cloud index
    if 'ALLSKY_SFC_SW_DWN' in target_row and 'CLRSKY_SFC_SW_DWN' in target_row:
        clear_sky = target_row['CLRSKY_SFC_SW_DWN']
        all_sky = target_row['ALLSKY_SFC_SW_DWN']
        cloud_index = 1 - (all_sky / clear_sky) if clear_sky > 0 else 0.5
    else:
        cloud_index = 0.5
    
    features['cloud_index_mean_7'] = cloud_index
    features['cloud_index_mean_30'] = cloud_index
    features['cloud_index_mean_90'] = cloud_index
    features['cloud_index_mean_365'] = cloud_index
    
    # Growing degree days
    gdd_daily = np.maximum(df_weather['T2M'] - 10, 0)
    features['gdd_base10_sum_7'] = gdd_daily.tail(7).sum()
    features['gdd_base10_sum_30'] = gdd_daily.tail(30).sum()
    features['gdd_base10_sum_90'] = gdd_daily.tail(90).sum()
    features['gdd_base10_sum_365'] = gdd_daily.sum()
    
    # Heat and frost days
    heat_days = (df_weather['T2M_MAX'] > 30).astype(int)
    frost_days = (df_weather['T2M_MIN'] < 0).astype(int)
    
    features['heat_days_gt30_7'] = heat_days.tail(7).sum()
    features['heat_days_gt30_30'] = heat_days.tail(30).sum()
    features['heat_days_gt30_90'] = heat_days.tail(90).sum()
    features['heat_days_gt30_365'] = heat_days.sum()
    
    features['frost_days_lt0_7'] = frost_days.tail(7).sum()
    features['frost_days_lt0_30'] = frost_days.tail(30).sum()
    features['frost_days_lt0_90'] = frost_days.tail(90).sum()
    features['frost_days_lt0_365'] = frost_days.sum()
    
    # SEASONAL FEATURES - Key enhancement!
    features['year'] = target_date.year
    features['month'] = target_date.month
    features['day_of_year'] = target_date.timetuple().tm_yday
    features['week_of_year'] = target_date.isocalendar()[1]
    
    # Peak season indicators (Autumn: Mar-May)
    features['is_peak_season'] = 1 if target_date.month in [3, 4, 5] else 0
    features['is_peak_month'] = 1 if target_date.month == 4 else 0  # April strongest
    features['is_secondary_peak'] = 1 if target_date.month == 5 else 0  # May secondary
    
    # Flowering intensity based on observation patterns
    flowering_map = {
        1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0, 5: 0.8, 6: 0.3,
        7: 0.1, 8: 0.1, 9: 0.2, 10: 0.3, 11: 0.4, 12: 0.2
    }
    features['flowering_intensity'] = flowering_map[target_date.month]
    
    # Distance from peak month (April)
    features['distance_from_peak'] = min(
        abs(target_date.month - 4), 
        abs(target_date.month - 4 + 12), 
        abs(target_date.month - 4 - 12)
    )
    
    # Optimal season weight
    season_weight_map = {
        1: 0.2, 2: 0.3, 3: 0.8, 4: 1.0, 5: 0.9, 6: 0.4,
        7: 0.2, 8: 0.2, 9: 0.3, 10: 0.4, 11: 0.5, 12: 0.3
    }
    features['optimal_season_weight'] = season_weight_map[target_date.month]
    
    # Recent observation emphasis
    current_year = datetime.now().year
    features['observation_recency'] = max(0, 1 - (current_year - target_date.year) / 10)
    
    # Days to observation (0 for prediction)
    features['days_to_observation'] = 0
    
    # Fill any missing features with defaults
    for feature in feature_columns:
        if feature not in features:
            if 'elevation' in feature:
                features[feature] = 200
            elif any(x in feature.lower() for x in ['t2m', 'temp']):
                features[feature] = 18
            elif any(x in feature.lower() for x in ['rain', 'prec']):
                features[feature] = 1.5
            elif any(x in feature.lower() for x in ['rh2m', 'humid']):
                features[feature] = 65
            elif any(x in feature.lower() for x in ['wind', 'ws2m']):
                features[feature] = 3.5
            else:
                features[feature] = 0
    
    return features

def get_default_seasonal_features(lat: float, lon: float, target_date: date, 
                                 feature_columns: List[str]) -> Dict:
    """Get default features when API fails, with seasonal awareness."""
    features = {
        'latitude_x': lat,
        'longitude_x': lon,
        'elevation': 200,
        
        # Default weather (Cape Town-like)
        'T2M': 18, 'T2M_MAX': 22, 'T2M_MIN': 14,
        'PRECTOTCORR': 1.5, 'RH2M': 65, 'WS2M': 3.5,
        'ALLSKY_SFC_SW_DWN': 5, 'CLRSKY_SFC_SW_DWN': 7,
        'TQV': 25, 'TS': 18,
        
        # Temporal aggregates
        'rain_sum_7': 10.5, 'rain_sum_30': 45, 'rain_sum_90': 135, 'rain_sum_365': 550,
        't2m_mean_7': 18, 't2m_mean_30': 18, 't2m_mean_90': 18, 't2m_mean_365': 18,
        'rh2m_mean_7': 65, 'rh2m_mean_30': 65, 'rh2m_mean_90': 65, 'rh2m_mean_365': 65,
        'wind_mean_7': 3.5, 'wind_mean_30': 3.5, 'wind_mean_90': 3.5, 'wind_mean_365': 3.5,
        'cloud_index_mean_7': 0.3, 'cloud_index_mean_30': 0.3, 'cloud_index_mean_90': 0.3, 'cloud_index_mean_365': 0.3,
        'gdd_base10_sum_7': 56, 'gdd_base10_sum_30': 240, 'gdd_base10_sum_90': 720, 'gdd_base10_sum_365': 2920,
        'heat_days_gt30_7': 0, 'heat_days_gt30_30': 0, 'heat_days_gt30_90': 0, 'heat_days_gt30_365': 5,
        'frost_days_lt0_7': 0, 'frost_days_lt0_30': 0, 'frost_days_lt0_90': 0, 'frost_days_lt0_365': 0,
        
        # SEASONAL FEATURES
        'year': target_date.year,
        'month': target_date.month,
        'day_of_year': target_date.timetuple().tm_yday,
        'week_of_year': target_date.isocalendar()[1],
        'is_peak_season': 1 if target_date.month in [3, 4, 5] else 0,
        'is_peak_month': 1 if target_date.month == 4 else 0,
        'is_secondary_peak': 1 if target_date.month == 5 else 0,
        'flowering_intensity': {1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0, 5: 0.8, 6: 0.3, 7: 0.1, 8: 0.1, 9: 0.2, 10: 0.3, 11: 0.4, 12: 0.2}[target_date.month],
        'distance_from_peak': min(abs(target_date.month - 4), abs(target_date.month - 4 + 12), abs(target_date.month - 4 - 12)),
        'optimal_season_weight': {1: 0.2, 2: 0.3, 3: 0.8, 4: 1.0, 5: 0.9, 6: 0.4, 7: 0.2, 8: 0.2, 9: 0.3, 10: 0.4, 11: 0.5, 12: 0.3}[target_date.month],
        'observation_recency': max(0, 1 - (datetime.now().year - target_date.year) / 10),
        'days_to_observation': 0
    }
    
    # Fill any missing features
    for feature in feature_columns:
        if feature not in features:
            features[feature] = 0
    
    return features

async def create_seasonal_heatmap(model_data, df, output_dir=Path("outputs"), 
                                 grid_resolution=0.02, api_base_url="http://127.0.0.1:8000",
                                 prediction_month=4):  # April = peak month
    """Create heatmap using seasonal model with peak season predictions."""
    
    print(f"Creating seasonal invasion risk heatmap for month {prediction_month}...")
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    seasonal_patterns = model_data.get('seasonal_patterns', {})
    
    print(f"Seasonal model loaded:")
    print(f"  Peak season: {seasonal_patterns.get('peak_season', 'Unknown')}")
    print(f"  Peak coverage: {seasonal_patterns.get('peak_season_percentage', 0):.1f}%")
    print(f"  Model features: {len(feature_columns)}")
    
    # Get bounds from real data
    lat_min = df['latitude_x'].min() - 0.1
    lat_max = df['latitude_x'].max() + 0.1
    lon_min = df['longitude_x'].min() - 0.1
    lon_max = df['longitude_x'].max() + 0.1
    
    print(f"Grid bounds: Lat [{lat_min:.3f}, {lat_max:.3f}], Lon [{lon_min:.3f}, {lon_max:.3f}]")
    
    # Create grid
    lat_grid = np.arange(lat_min, lat_max, grid_resolution)
    lon_grid = np.arange(lon_min, lon_max, grid_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid_coords = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    
    print(f"Grid size: {len(lat_grid)} x {len(lon_grid)} = {len(grid_coords):,} cells")
    
    # Target date in specified month
    target_date = date(2024, prediction_month, 15)
    
    # Fetch weather data
    print("Fetching weather data for seasonal predictions...")
    
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, (lat, lon) in enumerate(grid_coords):
            if i % 50 == 0:
                print(f"Queuing coordinate {i+1}/{len(grid_coords)}: ({lat:.3f}, {lon:.3f})")
            
            task = fetch_weather_for_coordinates(session, lat, lon, target_date, api_base_url)
            tasks.append(task)
        
        # Process in batches
        batch_size = 15
        weather_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            weather_results.extend(batch_results)
            
            await asyncio.sleep(1)
    
    # Process weather into features
    print("Processing weather data into seasonal features...")
    
    grid_features = []
    api_success_count = 0
    
    for i, (coord, weather_data) in enumerate(zip(grid_coords, weather_results)):
        lat, lon = coord
        
        if isinstance(weather_data, Exception) or weather_data is None:
            features = get_default_seasonal_features(lat, lon, target_date, feature_columns)
        else:
            features = process_seasonal_weather_data(weather_data, lat, lon, target_date, feature_columns)
            api_success_count += 1
        
        grid_features.append(features)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(grid_coords)} coordinates")
    
    print(f"API success rate: {api_success_count}/{len(grid_coords)} ({100*api_success_count/len(grid_coords):.1f}%)")
    
    # Create DataFrame and make predictions
    grid_df = pd.DataFrame(grid_features)
    
    # Ensure all features are present
    for feature in feature_columns:
        if feature not in grid_df.columns:
            grid_df[feature] = 0
    
    print("Making seasonal predictions...")
    X_grid = grid_df[feature_columns]
    X_grid_scaled = scaler.transform(X_grid)
    invasion_risk = model.predict_proba(X_grid_scaled)[:, 1]
    
    print(f"Risk range: {invasion_risk.min():.3f} - {invasion_risk.max():.3f}")
    
    # Reshape to grid
    risk_grid = invasion_risk.reshape(lat_mesh.shape)
    
    # Create visualization
    print("Creating seasonal heatmap visualization...")
    
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Title with seasonal info
    season_names = {3: 'Autumn', 4: 'Peak Autumn', 5: 'Late Autumn', 6: 'Winter', 
                   7: 'Winter', 8: 'Winter', 9: 'Spring', 10: 'Spring', 
                   11: 'Spring', 12: 'Summer', 1: 'Summer', 2: 'Summer'}
    
    title_html = f'''
    <h3 align="center" style="font-size:20px; margin: 10px;"><b>Pyracantha Seasonal Invasion Risk</b></h3>
    <p align="center" style="font-size:14px; margin: 5px;">Month: {prediction_month} ({season_names[prediction_month]}) - Real Weather Data</p>
    <p align="center" style="font-size:12px; margin: 5px;">Peak Season: Autumn (66% of observations)</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Enhanced colormap for seasonal predictions
    colors = ['#1a472a', '#2d5a3d', '#4a7c59', '#7fb069', '#b2df8a', '#ffee65', '#ffc107', '#ff8f00', '#ff6f00', '#e65100']
    cmap = LinearSegmentedColormap.from_list('seasonal_risk', colors, N=256)
    
    # Add grid cells
    cell_count = 0
    for i in range(len(lat_grid) - 1):
        for j in range(len(lon_grid) - 1):
            lat_start, lat_end = lat_grid[i], lat_grid[i + 1]
            lon_start, lon_end = lon_grid[j], lon_grid[j + 1]
            
            risk = risk_grid[i, j]
            risk_normalized = (risk - invasion_risk.min()) / (invasion_risk.max() - invasion_risk.min()) if invasion_risk.max() > invasion_risk.min() else 0.5
            
            color_rgba = cmap(risk_normalized)
            color_hex = f"#{int(color_rgba[0]*255):02x}{int(color_rgba[1]*255):02x}{int(color_rgba[2]*255):02x}"
            
            folium.Rectangle(
                bounds=[[lat_start, lon_start], [lat_end, lon_end]],
                color=color_hex,
                fill=True,
                fillColor=color_hex,
                fillOpacity=0.7,
                weight=0,
                popup=f"Seasonal Risk: {risk:.3f}<br>Month: {prediction_month}<br>Coords: ({(lat_start+lat_end)/2:.3f}, {(lon_start+lon_end)/2:.3f})"
            ).add_to(m)
            
            cell_count += 1
    
    # Add original observations
    pyracantha_df = df[df['scientific_name'] == 'Pyracantha angustifolia'].copy()
    
    if not pyracantha_df.empty:
        print(f"Adding {len(pyracantha_df)} observations...")
        
        for _, row in pyracantha_df.iterrows():
            # Color code by observation month
            try:
                obs_date = pd.to_datetime(row['time_observed_at'])
                if pd.isna(obs_date):
                    continue
                obs_month = obs_date.month
                
                if obs_month in [4]:  # Peak month
                    marker_color = 'red'
                    marker_size = 5
                elif obs_month in [3, 5]:  # Peak season
                    marker_color = 'orange'
                    marker_size = 4
                else:  # Off season
                    marker_color = 'blue'
                    marker_size = 3
                
                folium.CircleMarker(
                    location=[row['latitude_x'], row['longitude_x']],
                    radius=marker_size,
                    popup=f"Observation<br>Month: {obs_month}<br>Date: {obs_date.strftime('%Y-%m-%d')}<br>iNat ID: {row['inat_id']}",
                    color='black',
                    fill=True,
                    fillColor=marker_color,
                    fillOpacity=0.8,
                    weight=1
                ).add_to(m)
            except:
                # Skip observations with invalid dates
                continue
    
    # Enhanced legend with seasonal info
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 15px">
    <p><b>Seasonal Invasion Risk</b></p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#1a472a"></i> Low ({invasion_risk.min():.3f})</p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#ff8f00"></i> High ({invasion_risk.max():.3f})</p>
    <p style="margin: 10px 0 5px 0;"><b>Observations:</b></p>
    <p style="margin: 2px 0;"><i class="fa fa-circle" style="color:red"></i> Peak Month (April)</p>
    <p style="margin: 2px 0;"><i class="fa fa-circle" style="color:orange"></i> Peak Season (Mar-May)</p>
    <p style="margin: 2px 0;"><i class="fa fa-circle" style="color:blue"></i> Off Season</p>
    <p style="margin: 10px 0 0 0; font-size: 12px;">Prediction Month: {prediction_month}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Statistics with seasonal insights
    seasonal_stats = seasonal_patterns.get('obs_by_season', {})
    stats_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 300px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px">
    <p><b>Seasonal Model Statistics</b></p>
    <p>Prediction Month: {prediction_month} ({season_names[prediction_month]})</p>
    <p>Grid Resolution: {grid_resolution:.3f}°</p>
    <p>Total Cells: {len(grid_coords):,}</p>
    <p>API Success: {100*api_success_count/len(grid_coords):.1f}%</p>
    <p>Risk Range: {invasion_risk.min():.3f} - {invasion_risk.max():.3f}</p>
    <p>Mean Risk: {invasion_risk.mean():.3f}</p>
    <p><b>Seasonal Patterns:</b></p>
    <p>Autumn: {seasonal_stats.get('Autumn', 0)} obs (66%)</p>
    <p>Peak Month (Apr): 40% of sightings</p>
    <p>Target Date: {target_date}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_filename = f"seasonal_invasion_heatmap_month{prediction_month:02d}_{timestamp}.html"
    map_path = output_dir / map_filename
    
    output_dir.mkdir(exist_ok=True)
    m.save(str(map_path))
    
    print(f"Seasonal heatmap saved: {map_path}")
    print(f"Seasonal statistics:")
    print(f"  Prediction month: {prediction_month} ({season_names[prediction_month]})")
    print(f"  API success rate: {100*api_success_count/len(grid_coords):.1f}%")
    print(f"  Risk range: {invasion_risk.min():.3f} - {invasion_risk.max():.3f}")
    print(f"  Mean risk: {invasion_risk.mean():.3f}")
    
    return map_path, {
        'risk_grid': risk_grid,
        'invasion_risk': invasion_risk,
        'grid_coords': grid_coords,
        'grid_df': grid_df,
        'api_success_rate': api_success_count / len(grid_coords),
        'prediction_month': prediction_month,
        'seasonal_patterns': seasonal_patterns,
        'map_path': map_path
    }

def main():
    """Generate seasonal heatmap for peak observation month."""
    print("=" * 70)
    print("SEASONAL PYRACANTHA INVASION RISK HEATMAP")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load seasonal model
        model_data, df = load_seasonal_model()
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate heatmap for peak month (April)
        print("Generating heatmap for PEAK SEASON (April)...")
        peak_map_path, peak_results = asyncio.run(
            create_seasonal_heatmap(
                model_data, df, output_dir,
                grid_resolution=0.02,
                api_base_url="http://127.0.0.1:8000",
                prediction_month=4  # Peak month
            )
        )
        
        print()
        print("=" * 70)
        print("SEASONAL HEATMAP GENERATION COMPLETED!")
        print(f"Peak Season Map: {peak_map_path}")
        print(f"Peak Risk Range: {peak_results['invasion_risk'].min():.3f} - {peak_results['invasion_risk'].max():.3f}")
        print(f"Mean Risk: {peak_results['invasion_risk'].mean():.3f}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
