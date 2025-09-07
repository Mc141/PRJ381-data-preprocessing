"""
Dynamic Known Invasion Sites Generator

This script generates a CSV file of known Pyracantha invasion sites for any geographic area
using real observation data from either:
1. The training dataset
2. GBIF API for recent observations
3. A combination of both sources

The generated CSV is used by the enhanced heatmap generator to show known invasion sites
on the interactive map.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import random
import json
import requests
import asyncio
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import from app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import the GBIF fetcher class
from app.services.gbif_fetcher import GBIFFetcher

# Constants
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "known_invasion_sites.csv")
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "global_training_ml_ready.csv")
SPECIES_NAME = "Pyracantha angustifolia"
SPECIES_KEY = 3024580  # GBIF taxon key for Pyracantha angustifolia

def load_training_data(filepath: str = TRAINING_DATA_PATH) -> pd.DataFrame:
    """
    Load the global training dataset which contains real observations
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from training data")
        return df
    except Exception as e:
        logger.warning(f"Could not load training data: {e}")
        return pd.DataFrame()

async def fetch_gbif_observations(lat_min: float, lat_max: float, 
                          lon_min: float, lon_max: float,
                          limit: int = 50) -> pd.DataFrame:
    """
    Fetch recent observations from GBIF API for the specified area using the GBIFFetcher class
    """
    try:
        # Convert to the format expected by the GBIF API
        params = {
            'hasCoordinate': True,
            'limit': limit
        }
        
        # Try to use our app's GBIF fetcher service
        try:
            # Create a GBIF fetcher instance and use it to get occurrences
            async with GBIFFetcher() as fetcher:
                # Get occurrence batch with coordinates in the area
                batch_results = await fetcher.fetch_occurrences_batch(
                    species_key=SPECIES_KEY, 
                    limit=limit,
                    hasCoordinate="true",  # API expects string "true"/"false", not boolean
                    decimalLatitude=f"{lat_min},{lat_max}",
                    decimalLongitude=f"{lon_min},{lon_max}"
                )
                
                if batch_results and 'results' in batch_results:
                    occurrences = batch_results.get('results', [])
                    if occurrences and len(occurrences) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(occurrences)
                        logger.info(f"Fetched {len(df)} records from GBIF using app service")
                        return df
        except Exception as e:
            logger.warning(f"Could not use app GBIF fetcher: {e}")
        
        # Fall back to direct GBIF API call
        api_url = "https://api.gbif.org/v1/occurrence/search"
        params.update({
            'scientificName': SPECIES_NAME,
            'taxonKey': SPECIES_KEY,
            'decimalLatitude': f"{lat_min},{lat_max}",
            'decimalLongitude': f"{lon_min},{lon_max}"
        })
        
        response = requests.get(api_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                df = pd.DataFrame(data['results'])
                logger.info(f"Fetched {len(df)} records directly from GBIF API")
                return df
        
        logger.warning("No records found from GBIF API")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching from GBIF: {e}")
        return pd.DataFrame()

def filter_observations_by_area(df: pd.DataFrame, 
                               lat_min: float, lat_max: float, 
                               lon_min: float, lon_max: float) -> pd.DataFrame:
    """
    Filter observations to only include those in the specified geographic area
    """
    if df.empty:
        return df
    
    # Check if we have decimalLatitude or latitude column
    lat_col = 'decimalLatitude' if 'decimalLatitude' in df.columns else 'latitude'
    lon_col = 'decimalLongitude' if 'decimalLongitude' in df.columns else 'longitude'
    
    # Filter by coordinates
    mask = (
        (df[lat_col] >= lat_min) & 
        (df[lat_col] <= lat_max) & 
        (df[lon_col] >= lon_min) & 
        (df[lon_col] <= lon_max)
    )
    
    filtered_df = df[mask].copy()
    logger.info(f"Filtered to {len(filtered_df)} observations in specified area")
    return filtered_df

def estimate_severity(record: pd.Series) -> str:
    """
    Estimate invasion severity based on available data
    """
    # Try different approaches depending on available data
    
    # If we have individualCount, use that
    if 'individualCount' in record and pd.notna(record['individualCount']):
        count = record['individualCount']
        if count > 50:
            return 'critical'
        elif count > 20:
            return 'high'
        elif count > 5:
            return 'medium'
        else:
            return 'low'
    
    # If we have occurrence status that indicates establishment
    if 'establishmentMeans' in record and pd.notna(record['establishmentMeans']):
        status = str(record['establishmentMeans']).lower()
        if 'invasive' in status:
            return 'critical'
        elif 'established' in status:
            return 'high'
        elif 'naturalised' in status or 'naturalized' in status:
            return 'medium'
        
    # If we have recordedBy (multiple observers may indicate larger population)
    if 'recordedBy' in record and pd.notna(record['recordedBy']):
        if ',' in record['recordedBy'] or ';' in record['recordedBy']:
            return 'medium'
    
    # Default to low
    return 'low'

def format_known_sites_csv(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Format the observations into the required CSV structure
    """
    if observations.empty:
        logger.warning("No observations to format")
        return pd.DataFrame(columns=['latitude', 'longitude', 'name', 'discovery_date', 'severity'])
    
    # Initialize new dataframe with required columns
    sites_df = pd.DataFrame(columns=['latitude', 'longitude', 'name', 'discovery_date', 'severity'])
    
    # Determine which columns to use based on what's available
    lat_col = 'decimalLatitude' if 'decimalLatitude' in observations.columns else 'latitude'
    lon_col = 'decimalLongitude' if 'decimalLongitude' in observations.columns else 'longitude'
    date_col = next((col for col in ['eventDate', 'dateIdentified', 'lastInterpreted', 'modified'] 
                     if col in observations.columns and observations[col].notna().any()), None)
    name_col = next((col for col in ['locality', 'stateProvince', 'county', 'verbatimLocality'] 
                    if col in observations.columns and observations[col].notna().any()), None)
    
    # Process each observation
    for idx, row in observations.iterrows():
        site = {}
        
        # Coordinates
        site['latitude'] = row[lat_col]
        site['longitude'] = row[lon_col]
        
        # Location name
        if name_col and pd.notna(row[name_col]):
            site['name'] = row[name_col]
        else:
            # Generate a generic name based on coordinates
            site['name'] = f"Pyracantha site at {row[lat_col]:.4f}, {row[lon_col]:.4f}"
        
        # Discovery date
        if date_col and pd.notna(row[date_col]):
            # Try to parse the date
            try:
                # Handle various date formats
                date_str = str(row[date_col])
                if 'T' in date_str:  # ISO format with time
                    date_str = date_str.split('T')[0]
                
                # Try different date formats
                for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y'):
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        site['discovery_date'] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
                
                if 'discovery_date' not in site:
                    # If we couldn't parse the date, use a generic recent date
                    site['discovery_date'] = datetime.now().strftime('%Y-%m-%d')
            except:
                site['discovery_date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            # Use current date if no date available
            site['discovery_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Estimate severity
        site['severity'] = estimate_severity(row)
        
        # Add to dataframe
        sites_df = pd.concat([sites_df, pd.DataFrame([site])], ignore_index=True)
    
    return sites_df

async def generate_known_sites(lat_min: float, lat_max: float, 
                         lon_min: float, lon_max: float,
                         output_path: str = DEFAULT_CSV_PATH,
                         max_sites: int = 15,
                         use_training_data: bool = True,
                         use_gbif_api: bool = True) -> str:
    """
    Main function to generate known invasion sites CSV
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        output_path: Where to save the CSV
        max_sites: Maximum number of sites to include
        use_training_data: Whether to use training data
        use_gbif_api: Whether to fetch from GBIF API
    
    Returns:
        Path to the generated CSV file
    """
    observations_list = []
    
    # 1. Try to load from training data
    if use_training_data:
        training_data = load_training_data()
        if not training_data.empty:
            # Filter to the area of interest
            area_observations = filter_observations_by_area(
                training_data, lat_min, lat_max, lon_min, lon_max)
            if not area_observations.empty:
                observations_list.append(area_observations)
                logger.info(f"Found {len(area_observations)} observations in training data")
    
    # 2. Try to fetch from GBIF API
    if use_gbif_api:
        try:
            # Await the async GBIF fetch
            gbif_observations = await fetch_gbif_observations(
                lat_min, lat_max, lon_min, lon_max, limit=50)
            if not gbif_observations.empty:
                observations_list.append(gbif_observations)
                logger.info(f"Found {len(gbif_observations)} observations from GBIF API")
        except Exception as e:
            logger.error(f"Error fetching from GBIF API: {e}")
    
    # 3. Combine observations
    if observations_list:
        # Prioritize more recent or more detailed observations
        # For now, we'll just take up to max_sites, with preference to GBIF (more recent)
        all_observations = pd.concat(observations_list, ignore_index=True)
        # Remove duplicates based on coordinates
        lat_col = 'decimalLatitude' if 'decimalLatitude' in all_observations.columns else 'latitude'
        lon_col = 'decimalLongitude' if 'decimalLongitude' in all_observations.columns else 'longitude'
        
        if not all_observations.empty:
            # Round coordinates to 5 decimal places to find "almost duplicates"
            all_observations['lat_round'] = all_observations[lat_col].round(5)
            all_observations['lon_round'] = all_observations[lon_col].round(5)
            all_observations = all_observations.drop_duplicates(subset=['lat_round', 'lon_round'])
            
            # Format as known sites
            known_sites = format_known_sites_csv(all_observations)
            
            # Limit to max_sites
            if len(known_sites) > max_sites:
                known_sites = known_sites.sample(max_sites)
            
            # Save to CSV
            known_sites.to_csv(output_path, index=False)
            logger.info(f"Created known sites CSV with {len(known_sites)} real observations at {output_path}")
            return output_path
    
    # 4. If we couldn't find any observations, create a minimal CSV
    logger.warning("No observations found for the area. Creating a minimal CSV.")
    pd.DataFrame(columns=['latitude', 'longitude', 'name', 'discovery_date', 'severity']).to_csv(output_path, index=False)
    return output_path

async def main_async():
    """Async command-line interface"""
    parser = argparse.ArgumentParser(description='Generate known invasion sites CSV from real observations')
    parser.add_argument('--lat_min', type=float, required=True, help='Minimum latitude')
    parser.add_argument('--lat_max', type=float, required=True, help='Maximum latitude')
    parser.add_argument('--lon_min', type=float, required=True, help='Minimum longitude')
    parser.add_argument('--lon_max', type=float, required=True, help='Maximum longitude')
    parser.add_argument('--output', type=str, default=DEFAULT_CSV_PATH, help='Output CSV path')
    parser.add_argument('--max_sites', type=int, default=15, help='Maximum number of sites to include')
    parser.add_argument('--no_training_data', action='store_true', help='Do not use training data')
    parser.add_argument('--no_gbif_api', action='store_true', help='Do not use GBIF API')
    
    args = parser.parse_args()
    
    # Generate the CSV
    output_path = await generate_known_sites(
        args.lat_min, args.lat_max, args.lon_min, args.lon_max,
        output_path=args.output,
        max_sites=args.max_sites,
        use_training_data=not args.no_training_data,
        use_gbif_api=not args.no_gbif_api
    )
    
    print(f"Generated known invasion sites CSV at: {output_path}")
    return output_path

def main():
    """Synchronous entry point that runs the async function"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(main_async())

if __name__ == "__main__":
    main()
