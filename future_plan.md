
# 🔹 Step 1. Gather and Store Data

### iNaturalist

* Use `/observations/from` endpoint to pull recent sightings (e.g., last 14 days).
* Store them in MongoDB with fields:

  * `id`, `latitude`, `longitude`, `time_observed_at`.

### NASA POWER

* Use `/weather` or your merged dataset endpoints to pull weather variables for the same coordinates and dates.
* Store per-observation enriched records:

  * `inat_id`, `date`, `temperature`, `precipitation`, etc.

This gives you a **joined dataset** of presence points + conditions.

---

# 🔹 Step 2. Define "Presence Baseline"

* **Rule:** “If reported in the last 14 days → probability = 1 (confirmed presence).”
* This forms your *core presence map*.
* Implementation: query Mongo for recent observations, project to GeoJSON or shapefile.

---

# 🔹 Step 3. Identify Ecological Preferences

From ecological literature:

* Pyracantha angustifolia prefers:

  * **Temperature range:** mild, frost-resistant but not tropical.
  * **Moisture:** survives drought but spreads faster in wetter winters (Cape winter rainfall).
  * **Seasonality:** grows actively in spring → seeds spread in late summer/autumn.

Translate into thresholds:

```python
optimal_temp = (10, 25)  # °C
optimal_precip = (20, 100)  # mm/month
```

---

# 🔹 Step 4. Build Suitability Score

For each **grid cell or nearby location** around Table Mountain:

1. Pull recent weather (NASA POWER daily).
2. Score suitability:

   * 1 if conditions in optimal range, 0 otherwise.
   * Or use a weighted score (e.g., Gaussian decay if slightly outside range).

Example pseudo-code:

```python
def suitability(temp, precip):
    score_temp = 1 if 10 <= temp <= 25 else 0.5 if 5 <= temp <= 30 else 0
    score_precip = 1 if 20 <= precip <= 100 else 0.5 if 10 <= precip <= 120 else 0
    return (score_temp + score_precip) / 2
```

---

# 🔹 Step 5. Spread Prediction Logic

Combine **recent sightings (Step 2)** with **suitability scores (Step 4):**

* **Presence sites:** probability = 1.
* **Nearby sites:** probability = `distance_decay * suitability`.

Distance decay function:

```python
import numpy as np

def distance_decay(distance_km):
    return np.exp(-distance_km / 2)  # decays with 2 km radius
```

Final probability for a site:

```python
probability = max(
    [distance_decay(d) * suitability(temp, precip) 
     for each nearby sighting]
)
```

---

# 🔹 Step 6. Map & Visualize

* Create a **grid** of points covering Table Mountain National Park (e.g., every 500m).
* For each grid cell:

  * Compute probability (from Step 5).
* Output:

  * GeoJSON for web map,
  * Or raster grid for GIS.

Example (with Folium for quick map):

```python
import folium

m = folium.Map(location=[-34.05, 18.35], zoom_start=12)
for cell in grid_cells:
    folium.CircleMarker(
        location=(cell.lat, cell.lon),
        radius=3,
        color="red",
        fill=True,
        fill_opacity=cell.probability
    ).add_to(m)

m.save("prediction_map.html")
```

---

# 🔹 Step 7. Iterate to ML Later

Once this pipeline works:

* Train ML models (logistic regression, RandomForest, or MaxEnt).
* Features = NASA weather, season, elevation, slope, distance to roads/trails.
* Labels = presence (recent sightings) vs absence (pseudo-absence generated).

---

