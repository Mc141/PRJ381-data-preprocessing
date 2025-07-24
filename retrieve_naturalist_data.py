import requests
import json
import datetime



# Api needs a date range
# Will read date from persistent storage of when the last time the data was refreshed.
start_date = datetime.datetime(2025, 5, 1)
formatted_start_date = f"{start_date.year}-{start_date.month}-{start_date.day}"
# So everytime the this script is ran, the date needs to be logged and saved for future reference.


end_date = datetime.datetime.now()
formatted_end_date = f"{end_date.year}-{end_date.month}-{end_date.day}"





# Returns a single page. One page contains 30 results max. 
# So when retrieving in pipeline need to make sure there are not 30 sightings or more, otherwise need to retrieve page 2.
page = 1


# Border set to table mountain nature reserve
request_url = f"https://api.inaturalist.org/v1/observations?page={page}&d1={start_date}&d2={end_date}&quality_grade=any&identifications=any&swlat=-34.43214105152007&swlng=18.252509856447368&nelat=-33.806848821450004&nelng=18.580726409181743&taxon_id=54053&verifiable=true&spam=false&fields=id,uuid,observed_on,time_observed_at,user_id,created_at,quality_grade,image_url,place_guess,latitude,longitude,positional_accuracy,private_place_guess,scientific_name,common_name"


observations = requests.get(request_url)

observations_dict = observations.json()





# observations_dict['total_results'] returns the number of entries
# observations_dict['page'] returns the current page number
# observations_dict['per_page'] returns the number of entries per/current page


column_names = list(observations_dict['results'][0])




# print(observations_dict['results'][29]) returns entry 29



result_count = observations_dict['total_results']





