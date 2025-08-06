## **1. Startup & Connection Test**

**Goal:** Ensure MongoDB connection is established and routers are loaded.

* **Command:**

  ```bash
  uvicorn main:app --reload
  ```
* **Expected:** Console prints:

  ```
  Connected to MongoDB: mongodb://localhost:27017/invasive_db
  ```
* **Verify:** Navigate to `http://127.0.0.1:8000/docs` → all endpoints (`observations`, `weather`, `datasets`) are visible.

---

## **2. iNaturalist Fetch & Store**

**Goal:** Fetch all iNat observations and store in DB.

* **Endpoint:**

  ```
  GET /observations?store_in_db=true
  ```
* **Expected:**

  * Returns list of observation dicts with keys:
    `id`, `uuid`, `time_observed_at`, `latitude`, `longitude`, `scientific_name`, etc.
  * Console log:

    ```
    Stored X observations in MongoDB
    ```

---

## **3. Verify iNat in DB**

**Goal:** Confirm that stored observations are retrievable.

* **Endpoint:**

  ```
  GET /observations/db?limit=5
  ```
* **Expected:** Returns a JSON array of 5 observation docs from DB.

---

## **4. Fetch Weather Data Only**

**Goal:** Check NASA POWER API retrieval without DB storage.

* **Endpoint:**

  ```
  GET /weather?latitude=-33.9249&longitude=18.4073&start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10
  ```
* **Expected:** Returns:

  ```json
  {
    "location": {...},
    "parameters_meta": {...},
    "data": [ { "date": "...", "TS": ..., "T2M": ... }, ... ]
  }
  ```

---

## **5. Fetch & Store Weather in DB**

**Goal:** Test DB storage flag.

* **Endpoint:**

  ```
  GET /weather?latitude=-33.9249&longitude=18.4073&start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10&store_in_db=true
  ```
* **Expected:** Console log:

  ```
  Stored X weather records in MongoDB
  ```

---

## **6. Verify Weather in DB**

**Goal:** Confirm weather records stored correctly.

* **Endpoint:**

  ```
  GET /weather/db?limit=5
  ```
* **Expected:** Returns JSON array of weather records with `latitude`, `longitude`, `date`, and parameter fields.

---

## **7. Merge Datasets**

**Goal:** Fetch iNat + Weather together and store both in DB.

* **Endpoint:**

  ```
  GET /datasets/merge?start_year=2023&start_month=1&start_day=1&end_year=2023&end_month=1&end_day=10
  ```
* **Expected:**

  * `"count": <number>` and `"dataset": [...]`
  * Each dataset entry contains:

    ```json
    {
      "observation": {...},
      "weather": [ {...}, {...} ]
    }
    ```

---

## **8. Refresh Weather for Stored Observations**

**Goal:** Re-fetch weather for all stored iNat obs.

* **Endpoint:**

  ```
  POST /datasets/refresh-weather
  ```
* **Expected:** Returns:

  ```json
  { "updated_weather_records": <number> }
  ```

---

## **9. Export Full Dataset**

**Goal:** Test CSV export.

* **Endpoint:**

  ```
  GET /datasets/export
  ```
* **Expected:**

  * Downloads `dataset.csv` with merged iNat + weather fields.
  * Open in Excel or pandas to confirm merge correctness.

---

## **10. Cleanup**

**Goal:** Test deletion endpoints.

* **Endpoints:**

  ```
  DELETE /observations/db
  DELETE /weather/db
  ```
* **Expected:** Returns:

  ```json
  { "deleted_count": <number> }
  ```
