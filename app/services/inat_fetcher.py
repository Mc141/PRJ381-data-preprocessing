"""
iNaturalist API Client Module
=============================

This module provides functions for interacting with the iNaturalist API to retrieve
species observation data for biodiversity research and ecological analysis.

The module handles:
    - Asynchronous pagination through large datasets
    - Geographic filtering for specific study areas
    - Data validation and quality filtering
    - Coordinate extraction from various geographic formats
    - Observation metadata extraction and processing

Key Features:
    - Automatic pagination for complete data retrieval
    - Robust error handling for network issues
    - Flexible date range filtering
    - Geographic bounding box support
    - Quality grade and verification filtering

Geographic Focus:
    The module is configured for the Cape Town region with predefined bounding box:
    - Southwest: -34.43214, 18.25251 (latitude, longitude)
    - Northeast: -33.80685, 18.58073 (latitude, longitude)
    - Target taxon: Plants (taxon_id: 54053)

Usage Example:
    Basic observation retrieval::
    
        import asyncio
        from app.services.inat_fetcher import get_pages, get_observations
        
        # Fetch recent observations
        pages = await get_pages()
        observations = get_observations(pages)
        
        # Process observations
        for obs in observations:
            print(f"{obs['common_name']} at {obs['latitude']}, {obs['longitude']}")

API Documentation:
    iNaturalist API documentation: https://www.inaturalist.org/pages/api+reference

Author: MC141
"""

import json
import datetime
import math
import httpx
import asyncio
import requests


async def get_pages(start_date=datetime.datetime(2000, 1, 1), logger=None):
    """
    Asynchronously fetch all pages of observation data from the iNaturalist API.

    Retrieves observation data within a date range from start_date to today,
    automatically handling pagination to ensure complete data retrieval.
    The function is optimized for the Cape Town region and plant observations.

    Args:
        start_date (datetime.datetime, optional): Starting date for observation filtering.
            Defaults to January 1, 2000, which captures most available data.
        logger (logging.Logger, optional): Logger instance for recording operation
            details and errors. If None, no logging is performed.

    Returns:
        List[dict]: List of JSON response pages from the iNaturalist API.
            Each page contains up to 200 observations and metadata about
            the total results available.

    Raises:
        httpx.HTTPError: If API request fails due to network or server issues
        httpx.TimeoutException: If request exceeds timeout limits
        json.JSONDecodeError: If API response is not valid JSON

    Geographic Constraints:
        - Southwest corner: -34.43214105152007, 18.252509856447368
        - Northeast corner: -33.806848821450004, 18.580726409181743
        - Covers greater Cape Town metropolitan area

    Filtering Parameters:
        - quality_grade: "any" (includes research and casual grade observations)
        - identifications: "any" (includes observations with and without IDs)
        - taxon_id: 54053 (plants)
        - verifiable: "true" (excludes observations that cannot be verified)
        - spam: "false" (excludes flagged spam content)

    Example:
        Fetch observations from the last year::
        
            import asyncio
            from datetime import datetime, timedelta
            
            async def main():
                last_year = datetime.now() - timedelta(days=365)
                pages = await get_pages(start_date=last_year, logger=my_logger)
                print(f"Retrieved {len(pages)} pages of data")
            
            asyncio.run(main())

    Note:
        This function performs network I/O and should be called from an async context.
        Large date ranges may result in many API calls and longer execution times.
    """
    pages = []

    try:
        formatted_start_date = start_date.strftime('%Y-%m-%d')
        formatted_end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        base_url = "https://api.inaturalist.org/v1/observations"
        common_params = {
            "d1": formatted_start_date,
            "d2": formatted_end_date,
            "quality_grade": "any",
            "identifications": "any",
            "swlat": -34.43214105152007,
            "swlng": 18.252509856447368,
            "nelat": -33.806848821450004,
            "nelng": 18.580726409181743,
            "taxon_id": 54053,
            "verifiable": "true",
            "spam": "false",
            "fields": "id,uuid,observed_on,time_observed_at,user_id,created_at,quality_grade,image_url,"
                      "place_guess,latitude,longitude,positional_accuracy,private_place_guess,scientific_name,"
                      "common_name"
        }

        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch first page to get total results and pagination info
            response = await client.get(base_url, params=common_params)
            response.raise_for_status()
            first_page = response.json()
            pages.append(first_page)

            total_results = first_page.get("total_results", 0)
            page_count = math.ceil(total_results / 30)  # 30 results per page assumed

            # If multiple pages, fetch them concurrently
            if page_count > 1:
                tasks = []
                for page_num in range(2, page_count + 1):
                    params = common_params.copy()
                    params["page"] = page_num
                    tasks.append(fetch_page(client, base_url, params, page_num, logger))

                additional_pages = await asyncio.gather(*tasks)
                pages.extend(p for p in additional_pages if p is not None)

    except httpx.TimeoutException as e:
        _log(logger, f"Timeout while requesting first page: {e}")
    except httpx.RequestError as e:
        _log(logger, f"Request error during first page: {e}")
    except httpx.HTTPStatusError as e:
        _log(logger, f"HTTP error on first page: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        _log(logger, f"Unexpected error during pagination: {e}")

    return pages


async def fetch_page(client, url, params, page_num, logger=None):
    """
    Asynchronously fetch a single page of observations with error handling.

    Args:
        client (httpx.AsyncClient): HTTP client to use for the request.
        url (str): Base URL of the API endpoint.
        params (dict): Query parameters for the API call.
        page_num (int): Page number being fetched (used for logging).
        logger (callable, optional): Optional logging function.

    Returns:
        dict or None: JSON response dict if successful, None on failure.
    """
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        _log(logger, f"Timeout fetching page {page_num}")
    except httpx.RequestError as e:
        _log(logger, f"Request error on page {page_num}: {e}")
    except httpx.HTTPStatusError as e:
        _log(logger, f"HTTP error on page {page_num}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        _log(logger, f"Unexpected error on page {page_num}: {e}")
    return None


def _log(logger, message):
    """
    Log a message using the provided logger or fallback to console output.

    This utility function provides consistent logging across the module,
    allowing functions to optionally use a logger while gracefully falling
    back to print statements when no logger is available.

    Args:
        logger (logging.Logger or callable, optional): Logger instance or callable
            that accepts a message string. If None, message is printed to console.
        message (str): Message to log or print.

    Example:
        Usage with different logger types::
        
            import logging
            logger = logging.getLogger(__name__)
            
            # With logger
            _log(logger.info, "Processing started")
            
            # Without logger (prints to console)
            _log(None, "Fallback message")
    """
    if logger:
        logger(message)
    else:
        print(message)


def is_positional_accuracy_valid(observation, max_accuracy=100):
    """
    Validate the positional accuracy of an observation.

    Checks whether an observation's GPS coordinates meet the required accuracy
    standards for scientific analysis. This is crucial for ecological studies
    where precise location data is essential.

    Args:
        observation (dict): iNaturalist observation data containing positional metadata.
            Expected to have 'positional_accuracy' key with accuracy value in meters.
        max_accuracy (int, optional): Maximum allowed positional accuracy in meters.
            Defaults to 100 meters, which is suitable for most ecological studies.

    Returns:
        bool: True if the observation meets accuracy requirements, False otherwise.
            Returns False for missing accuracy data or non-integer values.

    Quality Criteria:
        - Positional accuracy must be present and not None
        - Value must be an integer (not string or float)
        - Value must be less than or equal to max_accuracy threshold
        - Negative values are considered valid (GPS uncertainty estimates)

    Example:
        Filter observations by accuracy::
        
            valid_observations = []
            for obs in observations:
                if is_positional_accuracy_valid(obs, max_accuracy=50):
                    valid_observations.append(obs)

    Note:
        iNaturalist positional accuracy represents the radius of uncertainty
        in meters around the reported coordinates. Lower values indicate
        higher precision.
    """
    accuracy = observation.get('positional_accuracy')
    return isinstance(accuracy, int) and accuracy <= max_accuracy


def extract_coordinates(observation):
    """
    Extract geographic coordinates from iNaturalist observation data.

    Processes observation data to extract latitude and longitude coordinates,
    handling multiple data formats that may be present in iNaturalist records.
    The function prioritizes GeoJSON format but falls back to location strings.

    Args:
        observation (dict): iNaturalist observation record containing geographic data.
            May include 'geojson' field with Point geometry or 'location' field
            with comma-separated coordinates.

    Returns:
        tuple[float | None, float | None]: A tuple containing (latitude, longitude).
            Returns (None, None) if coordinates cannot be extracted or parsed.

    Data Sources:
        1. **GeoJSON Format** (preferred): Uses 'geojson.coordinates' field
           - Format: {"type": "Point", "coordinates": [longitude, latitude]}
           - Note: GeoJSON uses [longitude, latitude] order
        
        2. **Location String** (fallback): Uses 'location' field  
           - Format: "latitude,longitude" as comma-separated string
           - Requires valid float conversion for both values

    Example:
        Extract coordinates from various formats::
        
            # GeoJSON format
            obs1 = {
                'geojson': {
                    'type': 'Point', 
                    'coordinates': [18.4241, -33.9249]
                }
            }
            lat, lon = extract_coordinates(obs1)  # (-33.9249, 18.4241)
            
            # Location string format  
            obs2 = {'location': '-33.9249,18.4241'}
            lat, lon = extract_coordinates(obs2)  # (-33.9249, 18.4241)
            
            # Invalid/missing data
            obs3 = {'location': 'invalid'}
            lat, lon = extract_coordinates(obs3)  # (None, None)

    Note:
        The function performs robust error handling and will not raise exceptions
        for malformed coordinate data. Invalid formats return None values.
    """
    latitude = None
    longitude = None

    geojson = observation.get('geojson')
    if geojson and geojson.get('type') == 'Point' and 'coordinates' in geojson:
        coords = geojson['coordinates']
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            longitude, latitude = coords[0], coords[1]
    elif observation.get('location'):
        try:
            lat_str, lon_str = observation['location'].split(',')
            latitude = float(lat_str)
            longitude = float(lon_str)
        except ValueError:
            # Ignore invalid location format
            pass

    return latitude, longitude


def extract_observation_data(observation):
    """
    Extract and standardize relevant data fields from an iNaturalist observation.

    Processes raw observation data from the iNaturalist API and extracts key fields
    needed for ecological analysis. The function handles missing data gracefully
    and standardizes field names for consistent downstream processing.

    Args:
        observation (dict): Raw observation record from iNaturalist API containing
            observation metadata, taxonomic information, geographic data, and
            user-contributed content.

    Returns:
        dict: Standardized observation data with the following fields:
            - id (int): Unique iNaturalist observation identifier
            - uuid (str): Universal unique identifier for the observation
            - observed_on (str): Date observation was made (YYYY-MM-DD format)
            - created_at (str): ISO timestamp when observation was created
            - latitude (float|None): Decimal latitude coordinate
            - longitude (float|None): Decimal longitude coordinate
            - positional_accuracy (int|None): GPS accuracy in meters
            - scientific_name (str|None): Taxonomic scientific name
            - common_name (str|None): Common/vernacular species name
            - image_url (str|None): URL of first observation photo
            - user_id (int|None): iNaturalist user identifier
            - quality_grade (str|None): Data quality assessment

    Data Processing:
        - **Coordinates**: Extracted using extract_coordinates() function
        - **Taxonomy**: Prioritizes taxon object data over root-level fields
        - **Images**: Uses first photo from photos array if available
        - **Dates**: Preserves original API format for compatibility
        - **Missing Data**: Gracefully handles None/missing values

    Example:
        Process observation data::
        
            raw_obs = {
                'id': 12345,
                'observed_on': '2023-07-15',
                'taxon': {'name': 'Protea cynaroides'},
                'photos': [{'url': 'https://example.com/photo.jpg'}],
                'geojson': {'type': 'Point', 'coordinates': [18.4, -33.9]}
            }
            
            clean_obs = extract_observation_data(raw_obs)
            print(clean_obs['scientific_name'])  # 'Protea cynaroides'
            print(clean_obs['latitude'])         # -33.9

    Note:
        This function is designed to work with the specific structure of
        iNaturalist API v1 observation objects. Field availability may
        vary based on observation privacy settings and user permissions.
    """
    latitude, longitude = extract_coordinates(observation)

    taxon = observation.get('taxon', {})
    scientific_name = taxon.get('name')
    common_name = taxon.get('preferred_common_name')

    image_url = None
    photos = observation.get('photos', [])
    if photos:
        image_url = photos[0].get('url')

    user = observation.get('user', {})
    user_id = user.get('id')

    return {
        "id": observation.get('id'),
        "uuid": observation.get('uuid'),
        "time_observed_at": observation.get('time_observed_at'),
        "created_at": observation.get('created_at'),
        "latitude": latitude,
        "longitude": longitude,
        "positional_accuracy": observation.get('positional_accuracy'),
        "place_guess": observation.get('place_guess'),
        "scientific_name": scientific_name,
        "common_name": common_name,
        "quality_grade": observation.get('quality_grade'),
        "image_url": image_url,
        "user_id": user_id
    }


def get_observations(pages):
    """
    Extract valid observations from multiple pages of API responses.

    Args:
        pages (list[dict]): List of API response pages.

    Returns:
        list[dict]: List of cleaned observation data dicts, filtering out
        observations with invalid positional accuracy.
    """
    observations = []
    for page in pages:
        for observation in page.get('results', []):
            if not is_positional_accuracy_valid(observation):
                continue
            observations.append(extract_observation_data(observation))
    return observations


def get_observation_by_id(id, logger=None):
    """
    Fetch a single observation by its ID from the iNaturalist API.

    Args:
        id (int): The observation ID to fetch.
        logger (callable, optional): Optional logging function.

    Returns:
        dict or None: Cleaned observation dict if found and valid, else None.
    """
    url = f"https://api.inaturalist.org/v1/observations?id={id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return None
        observation = results[0]
        if not is_positional_accuracy_valid(observation):
            return None
        return extract_observation_data(observation)
    except requests.exceptions.Timeout:
        _log(logger, f"Timeout error fetching observation ID {id}")
    except requests.exceptions.RequestException as e:
        _log(logger, f"Request error fetching observation ID {id}: {e}")
    except Exception as e:
        _log(logger, f"Unexpected error fetching observation ID {id}: {e}")
    return None
