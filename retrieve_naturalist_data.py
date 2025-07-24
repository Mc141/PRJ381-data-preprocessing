import requests
import json
import datetime
import math


# Sometimes multiple api requests are needed, this stores them
pages = []


# Api needs a date range
# Will read date from persistent storage of when the last time the data was refreshed.
start_date = datetime.datetime(2000, 5, 1)
formatted_start_date = f"{start_date.year}-{start_date.month}-{start_date.day}"
# So everytime the this script is ran, the date needs to be logged and saved for future reference.


end_date = datetime.datetime.now()
formatted_end_date = f"{end_date.year}-{end_date.month}-{end_date.day}"




request_url = f"https://api.inaturalist.org/v1/observations?d1={start_date}&d2={end_date}&quality_grade=any&identifications=any&swlat=-34.43214105152007&swlng=18.252509856447368&nelat=-33.806848821450004&nelng=18.580726409181743&taxon_id=54053&verifiable=true&spam=false&fields=id,uuid,observed_on,time_observed_at,user_id,created_at,quality_grade,image_url,place_guess,latitude,longitude,positional_accuracy,private_place_guess,scientific_name,common_name"


request_result = requests.get(request_url)

request_result_dict = request_result.json()


# append firt page results to list
pages.append(request_result_dict)


observation_count = request_result_dict['total_results']

page_count = math.ceil(observation_count / 30)


# If there is more than 1 page we need an extra request per page
if page_count > 1:
    # from page 2, as 1 has already been added
    for page in range(2, page_count + 1):
        request_url = f"https://api.inaturalist.org/v1/observations?page={page}&d1={start_date}&d2={end_date}&quality_grade=any&identifications=any&swlat=-34.43214105152007&swlng=18.252509856447368&nelat=-33.806848821450004&nelng=18.580726409181743&taxon_id=54053&verifiable=true&spam=false&fields=id,uuid,observed_on,time_observed_at,user_id,created_at,quality_grade,image_url,place_guess,latitude,longitude,positional_accuracy,private_place_guess,scientific_name,common_name"
        request_result = requests.get(request_url)
        request_result_dict = request_result.json()
        pages.append(request_result_dict)


observations = []


for page in pages:
    for observation in page['results']:
        # If inaccurate, ignore observation
        if (not isinstance(observation.get('positional_accuracy'), int)) or (observation.get('positional_accuracy') > 100): continue
        
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




print(len(observations))
