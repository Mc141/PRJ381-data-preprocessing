import json
import datetime
import math
import httpx
import asyncio
import requests


async def get_pages(start_date=datetime.datetime(2000, 1, 1), logger=None):
    """
    Asynchronously fetch all pages of observation data from the iNaturalist API 
    within the date range from `start_date` to today.

    Pagination is handled automatically to retrieve all available data.

    Args:
        start_date (datetime.datetime): Starting date to filter observations. Defaults to 2000-01-01.
        logger (callable, optional): Optional logging function to log errors or info messages.

    Returns:
        list[dict]: List of JSON response pages (dicts) from the API, each containing observation data.

    Notes:
        - Uses fixed geographic bounding box and taxon_id for filtering.
        - Handles HTTP and network errors, logging them if a logger is provided.
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
    Logs a message using the provided logger or prints to console if no logger.

    Args:
        logger (callable or None): Logging function (e.g., logger.info). If None, print is used.
        message (str): Message to log or print.
    """
    if logger:
        logger(message)
    else:
        print(message)


def is_positional_accuracy_valid(observation, max_accuracy=100):
    """
    Checks if the observation's positional accuracy is within acceptable limits.

    Args:
        observation (dict): Single observation data.
        max_accuracy (int): Maximum allowed positional accuracy in meters.

    Returns:
        bool: True if positional_accuracy is an integer and <= max_accuracy, False otherwise.
    """
    accuracy = observation.get('positional_accuracy')
    return isinstance(accuracy, int) and accuracy <= max_accuracy


def extract_coordinates(observation):
    """
    Extract latitude and longitude coordinates from an observation.

    Args:
        observation (dict): Single observation data.

    Returns:
        tuple(float or None, float or None): Latitude and longitude if found, else None for each.
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
    Extract and clean relevant data fields from an observation record.

    Args:
        observation (dict): Raw observation data from the API.

    Returns:
        dict: Cleaned observation data with selected fields.
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
