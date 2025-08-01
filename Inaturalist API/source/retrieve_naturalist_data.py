import json
import datetime
import math
import httpx
import asyncio


async def get_pages():
    """
    Fetch all pages of observation data from the iNaturalist API asynchronously.

    This function:
    - Calculates a date range.
    - Sends a request to the iNaturalist API.
    - Retrieves the total number of results and fetches all pages.
    - Returns the data as a list of page dictionaries.

    Returns:
        list[dict]: A list of result pages returned from the API.

    Raises:
        httpx.TimeoutException: If the request times out.
        httpx.RequestError: For network-related errors.
        httpx.HTTPError: For HTTP errors from the server.
    """
    # Sometimes multiple api requests are needed, this stores them
    pages = []

    try:
        # Api needs a date range
        # Will read date from persistent storage of when the last time the data was refreshed.
        start_date = datetime.datetime(2000, 5, 1)
        formatted_start_date = start_date.strftime('%Y-%m-%d')
        # So everytime the this script is ran, the date needs to be logged and saved for future reference.
        end_date = datetime.datetime.now()
        formatted_end_date = end_date.strftime('%Y-%m-%d')

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

        async with httpx.AsyncClient() as client:
            # JSON response
            response = await client.get(base_url, params=common_params)
            response.raise_for_status()  # Raise exception if error occurred during request
            request_result_dict = response.json()

            # append first page results to list
            pages.append(request_result_dict)

            # Calculate page count
            observation_count = request_result_dict['total_results']
            page_count = math.ceil(observation_count / 30)

            # If there is more than 1 page we need an extra request per page
            if page_count > 1:
                tasks = []
                for page in range(2, page_count + 1):
                    # Add page parameter
                    params = common_params.copy()
                    params["page"] = page
                    tasks.append(client.get(base_url, params=params))

                responses = await asyncio.gather(*tasks)

                for r in responses:
                    r.raise_for_status()  # Raise exception if error occurred during request
                    pages.append(r.json())

    except httpx.TimeoutException as e:
        print(f"Timeout error: {e}")

    except httpx.RequestError as e:
        print(f"Request error: {e}")

    except httpx.HTTPError as e:
        print(f"HTTP exception for {e.request.url} - {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    return pages


def get_observations(pages):
    """
    Process API pages to extract and filter useful observations.

    This function:
    - Iterates through each page and result.
    - Skips observations with invalid or high positional accuracy (>100).
    - Extracts and returns relevant observation fields.

    Args:
        pages (list[dict]): A list of result pages from the API.

    Returns:
        list[dict]: A list of filtered and structured observations.
    """
    observations = []

    for page in pages:
        for observation in page['results']:
            # If inaccurate, ignore observation
            if (not isinstance(observation.get('positional_accuracy'), int)) or (observation.get('positional_accuracy') > 100):
                continue

            observation_dict = {
                "id": observation['id'],
                "uuid": observation.get('uuid'),
                "time_observed_at": observation.get('time_observed_at'),
                "created_at": observation.get('created_at'),
                "latitude": observation.get('latitude'),
                "longitude": observation.get('longitude'),
                "positional_accuracy": observation.get('positional_accuracy'),
                "place_guess": observation.get('place_guess'),
                "scientific_name": observation.get('scientific_name'),
                "common_name": observation.get('common_name'),
                "quality_grade": observation.get('quality_grade'),
                "image_url": observation.get('image_url'),
                "user_id": observation.get('user_id')
            }
            observations.append(observation_dict)

    return observations
