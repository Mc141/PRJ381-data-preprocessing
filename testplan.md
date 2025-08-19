# **Manual API Testing Guide - Swagger UI**

This guide provides step-by-step instructions for manually testing all API endpoints using Swagger UI at `http://127.0.0.1:8000/docs`

## **🚀 Prerequisites**

1. **Start the server:**
   ```bash
   cd app
   uvicorn main:app --reload
   ```

2. **Open Swagger UI:** Navigate to `http://127.0.0.1:8000/docs`

3. **Verify MongoDB:** Ensure MongoDB is running and accessible

---

## **📋 Testing Order & Endpoints**

### **Phase 1: System Health & Setup**

#### **1.1 Health Check**
- **Endpoint:** `GET /status/health`
- **Parameters:** None
- **Expected Response:**
  ```json
  {"status": "healthy", "timestamp": "2025-08-18T..."}
  ```
- **Purpose:** Verify API is running

#### **1.2 Service Information**
- **Endpoint:** `GET /status/service_info`
- **Parameters:** None
- **Expected Response:**
  ```json
  {
    "service": "PRJ381 Data Preprocessing API",
    "version": "1.0.0",
    "endpoints": ["observations", "weather", "datasets", "status"]
  }
  ```
- **Purpose:** Confirm all services are loaded

---

### **Phase 2: Observations (iNaturalist Data)**

#### **2.1 Fetch Recent Observations (No Storage)**
- **Endpoint:** `GET /observations`
- **Parameters:**
  - `store_in_db`: `false` (default)
- **Expected Response:** Array of observation objects with keys:
  ```json
  [
    {
      "id": 123456789,
      "uuid": "abc-def-ghi",
      "time_observed_at": "2025-07-15T10:30:00Z",
      "latitude": -33.9249,
      "longitude": 18.4073,
      "scientific_name": "Pinus pinaster",
      "common_name": "Cluster Pine",
      "quality_grade": "research",
      "positional_accuracy": 25,
      "place_guess": "Cape Town, South Africa",
      "image_url": "https://...",
      "user_id": 567890
    }
  ]
  ```
- **Console Log:** `Fetched X observations (no date filter)`

#### **2.2 Fetch & Store Observations**
- **Endpoint:** `GET /observations`
- **Parameters:**
  - `store_in_db`: `true`
- **Expected Response:** Same as above
- **Console Log:** `Stored X observations in MongoDB`

#### **2.3 Fetch Observations from Specific Date**
- **Endpoint:** `GET /observations/from`
- **Parameters:**
  - `year`: `2025`
  - `month`: `7`
  - `day`: `15`
  - `store_in_db`: `true`
- **Expected Response:** Array of observations from July 15, 2025
- **Console Log:** `Fetched X observations from 2025-07-15`

#### **2.4 Get Individual Observation**
- **Endpoint:** `GET /observations/{id}`
- **Parameters:**
  - `id`: Use an ID from previous responses (e.g., `123456789`)
- **Expected Response:** Single observation object
- **Note:** May return 404 if observation doesn't exist

---

### **Phase 3: Weather Data (NASA POWER)**

#### **3.1 Fetch Weather Data (No Storage)**
- **Endpoint:** `GET /weather`
- **Parameters:**
  - `latitude`: `-33.9249` (Cape Town)
  - `longitude`: `18.4073`
  - `start_year`: `2024`
  - `start_month`: `7`
  - `start_day`: `1`
  - `end_year`: `2024`
  - `end_month`: `7`
  - `end_day`: `10`
  - `store_in_db`: `false`
- **Expected Response:**
  ```json
  {
    "location": {
      "latitude": -33.9249,
      "longitude": 18.4073,
      "elevation": 56.4
    },
    "parameters": {
      "T2M": {"longname": "Temperature at 2 Meters", "units": "C"},
      "PRECTOTCORR": {"longname": "Precipitation Corrected", "units": "mm/day"}
    },
    "data": [
      {
        "date": "2024-07-01",
        "T2M": 15.2,
        "T2M_MAX": 18.5,
        "T2M_MIN": 11.8,
        "PRECTOTCORR": 2.3,
        "RH2M": 75.2,
        "WS2M": 12.4,
        "latitude": -33.9249,
        "longitude": 18.4073,
        "elevation": 56.4
      }
    ]
  }
  ```
- **Console Log:** `Fetched weather data for (-33.9249, 18.4073) from 2024-07-01 to 2024-07-10`

#### **3.2 Fetch & Store Weather Data**
- **Endpoint:** `GET /weather`
- **Parameters:** Same as 3.1, but:
  - `store_in_db`: `true`
- **Expected Response:** Same as above
- **Console Log:** `Stored X weather records in MongoDB`

#### **3.3 Retrieve Stored Weather Data**
- **Endpoint:** `GET /weather/db`
- **Parameters:**
  - `limit`: `10`
- **Expected Response:** Array of stored weather records (without `_id` fields)

#### **3.4 Get Recent Weather for Location**
- **Endpoint:** `GET /weather/recent`
- **Parameters:**
  - `latitude`: `-33.9249`
  - `longitude`: `18.4073`
  - `days`: `5`
- **Expected Response:** Array of most recent 5 weather records for this location
- **Note:** Only returns data if coordinates match exactly

#### **3.5 Delete All Weather Data**
- **Endpoint:** `DELETE /weather/db`
- **Parameters:** None
- **Expected Response:**
  ```json
  {"message": "Deleted X weather records"}
  ```

---

### **Phase 4: Integrated Datasets**

#### **4.1 Merge Observations + Weather Data**
- **Endpoint:** `GET /datasets/merge`
- **Parameters:**
  - `start_year`: `2025`
  - `start_month`: `7`
  - `start_day`: `1`
  - `end_year`: `2025`
  - `end_month`: `7`
  - `end_day`: `15`
  - `years_back`: `3`
- **Expected Response:**
  ```json
  {
    "count": 5,
    "preview": [
      {
        "observation": {
          "id": 123456789,
          "latitude": -33.9249,
          "longitude": 18.4073,
          "scientific_name": "Pinus pinaster",
          "time_observed_at": "2025-07-10T12:00:00Z"
        },
        "weather_on_obs_date": {
          "date": "2025-07-10",
          "T2M": 16.8,
          "PRECTOTCORR": 0.5
        },
        "features": {
          "rain_7d_sum": 15.3,
          "t2m_30d_mean": 14.7,
          "gdd_365d_sum": 1245.6
        }
      }
    ]
  }
  ```
- **Console Log:** `Fetched iNaturalist observations from 2025-07-01 to 2025-07-15`
- **Note:** This may take 30-60 seconds for concurrent processing

#### **4.2 Refresh Weather for Stored Observations**
- **Endpoint:** `POST /datasets/refresh-weather`
- **Parameters:**
  - `years_back`: `5`
- **Expected Response:**
  ```json
  {"updated_weather_records": 8}
  ```
- **Purpose:** Updates weather data for all stored observations with 5-year history
- **Note:** Only works if observations exist in database

#### **4.3 Export Complete Dataset**
- **Endpoint:** `GET /datasets/export`
- **Parameters:**
  - `include_features`: `true`
- **Expected Response:** CSV file download (`dataset.csv`)
- **Content:** Merged observations + weather + computed features
- **Note:** File will contain columns for all weather parameters and engineered features

#### **4.4 Export Dataset Without Features**
- **Endpoint:** `GET /datasets/export`
- **Parameters:**
  - `include_features`: `false`
- **Expected Response:** CSV file with observations and weather data only

---

## **🔍 Testing Validation Checklist**

### **✅ Success Indicators:**

1. **Status Endpoints:**
   - Health check returns 200 with "healthy" status
   - Service info lists all router endpoints

2. **Observations:**
   - Returns valid JSON arrays with observation objects
   - Console logs show fetch and storage operations
   - Individual observation retrieval works

3. **Weather:**
   - NASA POWER data returns with location, parameters, and data arrays
   - Storage operations log successfully to console
   - Database retrieval returns stored records
   - Recent weather filtering works for exact coordinates

4. **Datasets:**
   - Merge operation creates integrated observation+weather records
   - Features are computed and included in response
   - CSV export downloads successfully
   - Refresh operation updates existing records

### **❌ Error Scenarios to Test:**

1. **Invalid Parameters:**
   - Invalid dates (e.g., `month: 13`)
   - Invalid coordinates (e.g., `latitude: 200`)
   - Future dates for observations

2. **Edge Cases:**
   - Empty date ranges
   - Coordinates with no observations
   - Large date ranges (test performance)

3. **Database States:**
   - Empty database (no stored data)
   - Partial data (observations but no weather)

---

## **📝 Expected Console Logs**

During testing, watch for these console messages:

```
Connected to MongoDB: mongodb://localhost:27017/invasive_db
[2025-08-18] INFO - Fetched 15 observations (no date filter)
[2025-08-18] INFO - Stored 15 observations in MongoDB
[2025-08-18] INFO - Fetched weather data for (-33.9249, 18.4073) from 2024-07-01 to 2024-07-10
[2025-08-18] INFO - Stored 10 weather records in MongoDB
[2025-08-18] INFO - Fetched iNaturalist observations from 2025-07-01 to 2025-07-15
[2025-08-18] INFO - Deleted 25 weather records from MongoDB
```

---

## **🎯 Complete Testing Workflow**

**Recommended order for comprehensive testing:**

1. **System Check:** Test status endpoints (1.1, 1.2)
2. **Data Collection:** Test observation endpoints (2.1-2.4)
3. **Weather Integration:** Test weather endpoints (3.1-3.5)
4. **Dataset Creation:** Test merge and export (4.1, 4.3)
5. **Data Management:** Test refresh and cleanup (4.2, 3.5)

**Total Testing Time:** ~15-20 minutes for complete workflow

This covers all functionality and ensures your data preprocessing pipeline works end-to-end! 🚀
