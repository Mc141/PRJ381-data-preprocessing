# Invasive Plant Observation Collector

**Status:** Work in Progress 

A standalone asynchronous data collection service designed to support a larger ecological monitoring and spread prediction platform.

## Project Overview

This microservice retrieves geospatial observation data from the **iNaturalist API** for a specific invasive plant species (*Pyracantha angustifolia*) within a custom-defined boundary (focused on the Western Cape, South Africa).

The goal of this service is to:
- Gather **accurate and filtered** observation data
- Enrich it later with environmental variables (e.g., temperature)
- Provide a **clean, retrainable dataset** for ML models used in predicting seasonal spread

This service is intended to run periodically and store data locally or in a database (e.g., PostgreSQL), which will later be used by other components in the full system for modeling and prediction.

## Current Functionality

- Pulls **research-grade** sightings from iNaturalist via asynchronous HTTP requests
- Filters out low-accuracy or incomplete data points
- Structures the observation data into a clean list of dictionaries
- Includes **unit tests and fixtures** using `pytest`, following a **Test-Driven Development (TDD)** workflow
- Fully asynchronous and uses `httpx`, `asyncio` for scalable API interactions
- **FastAPI REST endpoints** for flexible data access and integration

## API Endpoints

The service exposes the following REST endpoints:

### `GET /observations`
Fetches all available research-grade observations for the target species.

### `GET /observations/from-date`
Retrieves observations from a specified date onwards.

**Query Parameters:**
- `year` (int, required): Year of the observation date
- `month` (int, required): Month of the observation date  
- `day` (int, required): Day of the observation date

**Example:**
```
GET /observations/from-date?year=2024&month=1&day=15
```

### `GET /observations/{observation_id}`
Fetches a single observation by its unique ID.

**Path Parameters:**
- `observation_id` (int, required): ID of the specified observation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Web Framework | FastAPI |
| HTTP Client | `httpx` (async) |
| Testing | `pytest`, `pytest-mock` |

| ORM/DB | Planned `Peewee` + PostgreSQL (future) |
| Containerization | Docker (planned) |

## Integration Plan

This service will be part of a **larger architecture** involving:
- A **frontend React SPA** for image classification and reporting
- A **FastAPI backend** for ML inference and user submissions
- A **forecast model** combining image classification with environmental data (from NASA POWER API)
- A **shared PostgreSQL DB** for central data access and periodic model retraining

This service will run on a schedule (e.g., cron job or async task runner) and can be deployed independently.



## Notes

- Only **research-grade** observations within a defined bounding box are collected
- Designed with **low frequency data loads** in mind (a few rows every few weeks)
- Comprehensive logging implemented for monitoring and debugging
- Input validation and error handling for robust API responses

## Future Steps

- [ ] Weather enrichment via NASA POWER API
- [ ] Database integration with PostgreSQL
- [ ] Dockerization and container deployment
- [ ] Scheduled jobs or background task processing
- [ ] Authentication and rate limiting
- [ ] Data export capabilities (CSV, JSON)
- [ ] Monitoring and metrics collection

## Maintainer

This is a portfolio component by **Martinus Christoffel Wolmarans** and will contribute to the larger Invasive Plant Monitoring System final year project.