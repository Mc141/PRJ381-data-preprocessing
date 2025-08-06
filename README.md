# Invasive Plant Observation Collector

**Status:** Work in Progress

A standalone asynchronous data collection and enrichment service for invasive plant monitoring.
It supports **iNaturalist** species observation retrieval, **NASA POWER API** environmental enrichment, and persistent storage in **MongoDB**.
It forms part of a larger ecological monitoring and spread prediction platform.

---

## Project Overview

This microservice:

* Retrieves geospatial observation data from **iNaturalist** for *Pyracantha angustifolia* within a defined Western Cape, South Africa bounding box.
* Enriches observations with environmental variables (temperature, precipitation, humidity, wind speed, radiation, etc.) from **NASA POWER API**.
* Stores observations and weather data in **MongoDB** for later use in ML training and prediction.
* Provides an API for fetching, refreshing, merging, and exporting datasets.

This service is intended to:

* Run periodically (scheduled job or on-demand)
* Provide **clean, retrainable datasets** for ML models predicting seasonal spread
* Serve as a **data ingestion and enrichment layer** in a multi-service architecture

---

## Current Functionality

* **iNaturalist API Integration**

  * Asynchronous fetching of observation data
  * Filtering by date range, location bounding box, species taxon ID
  * Validation of positional accuracy
  * Cleaned observation structure ready for ML use
  * Optional storage in MongoDB

* **NASA POWER API Integration**

  * Fetch daily weather data for given coordinates & date ranges
  * Enrich iNat observations with environmental context
  * Optional storage in MongoDB

* **Dataset Building & Management**

  * Merge iNat and NASA data into a unified dataset
  * Refresh weather data for stored iNat observations
  * Export merged datasets as CSV
  * Retrieve and manage stored records

* **FastAPI REST Endpoints**

  * Modular routers for `/observations`, `/weather`, `/datasets`
  * MongoDB persistence layer
  * Full JSON API responses

---

## API Endpoints

### Observations Router (`/observations`)

| Method   | Endpoint                         | Description                                                           |
| -------- | -------------------------------- | --------------------------------------------------------------------- |
| `GET`    | `/observations`                  | Fetch all iNat observations (no date filter). Optionally store in DB. |
| `GET`    | `/observations/from`             | Fetch iNat observations from a specific date. Optionally store in DB. |
| `GET`    | `/observations/{observation_id}` | Retrieve a single observation by ID from iNat API.                    |
| `GET`    | `/observations/db`               | Retrieve stored iNat observations from MongoDB.                       |
| `DELETE` | `/observations/db`               | Delete all stored iNat observations.                                  |

---

### Weather Router (`/weather`)

| Method   | Endpoint          | Description                                                                           |
| -------- | ----------------- | ------------------------------------------------------------------------------------- |
| `GET`    | `/weather`        | Fetch NASA POWER weather data for coordinates and date range. Optionally store in DB. |
| `GET`    | `/weather/db`     | Retrieve stored weather data from MongoDB.                                            |
| `GET`    | `/weather/recent` | Retrieve most recent weather records for a given location.                            |
| `DELETE` | `/weather/db`     | Delete all stored weather data.                                                       |

---

### Datasets Router (`/datasets`)

| Method | Endpoint                    | Description                                                              |
| ------ | --------------------------- | ------------------------------------------------------------------------ |
| `GET`  | `/datasets/merge`           | Fetch iNat + NASA data concurrently, store in DB, return merged dataset. |
| `POST` | `/datasets/refresh-weather` | Re-fetch weather for all stored iNat observations and update DB.         |
| `GET`  | `/datasets/export`          | Export merged dataset as CSV.                                            |

---

## Tech Stack

| Component         | Technology          |
| ----------------- | ------------------- |
| Language          | Python 3.11+        |
| Web Framework     | FastAPI             |
| Async HTTP Client | `httpx`             |
| Data Handling     | `pandas`            |
| Database          | MongoDB (`pymongo`) |
| Environment       | `python-dotenv`     |
| Logging           | Python `logging`    |
| Concurrency       | `asyncio`           |

---

## Example Workflow

1. **Fetch & Store Observations**

```http
GET /observations?store_in_db=true
```

2. **Fetch & Store Weather Data**

```http
GET /weather?latitude=-33.9249&longitude=18.4073&start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10&store_in_db=true
```

3. **Merge & Enrich**

```http
GET /datasets/merge?start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10
```

4. **Refresh Weather for Stored Observations**

```http
POST /datasets/refresh-weather
```

5. **Export Dataset**

```http
GET /datasets/export
```

---

## Future Steps

* [ ] Add scheduled background jobs for periodic data sync
* [ ] Introduce update-skipping for already up-to-date weather records
* [ ] Enhance filtering and query parameters
* [ ] Dockerize and prepare for deployment
* [ ] Integration with ML training pipeline
* [ ] User authentication and API keys

---

## Maintainer

This is a component by **Martinus Christoffel Wolmarans** and will contribute to the larger Invasive Plant Monitoring System final year project.