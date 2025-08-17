# app/routers/datasets.py
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from app.services.inat_fetcher import get_pages, get_observations
from app.services.nasa_fetcher import PowerAPI
from app.services.database import get_database

import pandas as pd
import numpy as np
import io
import asyncio

router = APIRouter()
logger = logging.getLogger("datasets")

# helpers

PARAM_REMAP = {
    "PRECTOTCORR": "rain",     # mm/day
    "T2M": "t2m",              # °C
    "T2M_MAX": "t2m_max",
    "T2M_MIN": "t2m_min",
    "RH2M": "rh2m",            # %
    "WS2M": "ws2m",            # m/s
    "ALLSKY_SFC_SW_DWN": "allsky",
    "CLRSKY_SFC_SW_DWN": "clrsky",
    "TQV": "tqv",
    "TS": "ts",
}

WINDOWS = [7, 30, 90, 365]  # compute features on these lookback windows when available

def _clean_weather_df(weather_rows: list[dict]) -> pd.DataFrame:
    """Create & clean a dataframe from weather rows (daily granularity)."""
    if not weather_rows:
        return pd.DataFrame()

    df = pd.DataFrame(weather_rows).copy()
    # normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    # Replace NASA fill values -999 with NaN for numeric columns
    for col in [
        "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "PRECTOTCORR", "RH2M", "T2M",
        "T2M_MAX", "T2M_MIN", "TQV", "TS", "WS2M"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -999, col] = np.nan

    # Compute cloud index (1 - ALLSKY/CLRSKY), clipped [0,1]
    if "ALLSKY_SFC_SW_DWN" in df.columns and "CLRSKY_SFC_SW_DWN" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]
        df["cloud_index"] = (1.0 - ratio).clip(lower=0, upper=1)
    else:
        df["cloud_index"] = np.nan

    # Remap to simpler names for feature engineering
    for src, dst in PARAM_REMAP.items():
        if src in df.columns:
            df[dst] = df[src]

    return df

def _gdd(series_tmin: pd.Series, series_tmax: pd.Series, base: float = 10.0) -> pd.Series:
    """
    Simple daily Growing Degree Days using (Tmin + Tmax)/2 - base, min 0.
    If Tmax/Tmin missing, fallback to T2M (mean).
    """
    avg = None
    if series_tmin is not None and series_tmax is not None:
        avg = (series_tmin + series_tmax) / 2.0
    else:
        avg = None
    gdd = (avg - base) if avg is not None else pd.Series(dtype=float)
    gdd = gdd.clip(lower=0)
    return gdd

def _roll_window(series: pd.Series, window: int, fn: str) -> float | None:
    """Return rolling aggregation as of the last day. If insufficient data, use available data."""
    s = series.dropna()
    if s.empty:
        return None
    # Select last `window` days if available
    s_tail = s.tail(window)
    if s_tail.empty:
        return None
    if fn == "sum":
        return float(s_tail.sum())
    if fn == "mean":
        return float(s_tail.mean())
    if fn == "max":
        return float(s_tail.max())
    if fn == "min":
        return float(s_tail.min())
    if fn == "count":
        return int(s_tail.count())
    return None

def _count_condition(series: pd.Series, window: int, op: str, threshold: float) -> int | None:
    """Count days meeting a condition in last `window` days."""
    s = series.dropna().tail(window)
    if s.empty:
        return None
    if op == "gt":
        return int((s > threshold).sum())
    if op == "lt":
        return int((s < threshold).sum())
    return None

def _compute_features(df: pd.DataFrame) -> dict:
    """
    Compute windowed features using the last date as the anchor (the obs date).
    Assumes df is daily and sorted by date ascending.
    """
    if df.empty:
        return {}

    features = {}

    # Daily GDD base 10 using T2M_MIN/T2M_MAX (fallback to T2M handled above if needed)
    if "t2m_min" in df.columns and "t2m_max" in df.columns:
        daily_gdd = _gdd(df["t2m_min"], df["t2m_max"], base=10.0)
    else:
        # fallback: approximate daily_gdd from mean T2M if min/max unavailable
        if "t2m" in df.columns:
            daily_gdd = (df["t2m"] - 10.0).clip(lower=0)
        else:
            daily_gdd = pd.Series(dtype=float)

    for w in WINDOWS:
        # rain sums
        if "rain" in df.columns:
            features[f"rain_sum_{w}"] = _roll_window(df["rain"], w, "sum")

        # temperature means/extremes
        if "t2m" in df.columns:
            features[f"t2m_mean_{w}"] = _roll_window(df["t2m"], w, "mean")
        if "t2m_max" in df.columns:
            features[f"t2m_max_{w}"] = _roll_window(df["t2m_max"], w, "mean")
        if "t2m_min" in df.columns:
            features[f"t2m_min_{w}"] = _roll_window(df["t2m_min"], w, "mean")

        # humidity, wind
        if "rh2m" in df.columns:
            features[f"rh2m_mean_{w}"] = _roll_window(df["rh2m"], w, "mean")
        if "ws2m" in df.columns:
            features[f"wind_mean_{w}"] = _roll_window(df["ws2m"], w, "mean")

        # cloudiness
        if "cloud_index" in df.columns:
            features[f"cloud_index_mean_{w}"] = _roll_window(df["cloud_index"], w, "mean")

        # GDD accumulation
        features[f"gdd_base10_sum_{w}"] = _roll_window(daily_gdd, w, "sum")

        # heat/frost day counts
        if "t2m_max" in df.columns:
            features[f"heat_days_gt30_{w}"] = _count_condition(df["t2m_max"], w, "gt", 30.0)
        if "t2m_min" in df.columns:
            features[f"frost_days_lt0_{w}"] = _count_condition(df["t2m_min"], w, "lt", 0.0)

    return features

# ---- endpoints ---------------------------------------------------------------

@router.get("/datasets/merge")
async def merge_datasets(
    start_year: int = Query(...),
    start_month: int = Query(...),
    start_day: int = Query(...),
    end_year: int = Query(...),
    end_month: int = Query(...),
    end_day: int = Query(...),
    years_back: int = Query(3, ge=1, le=10, description="How many years of daily weather to pull prior to each observation date")
):
    """
    Fetch iNaturalist + multi-year NASA POWER data concurrently, store in MongoDB,
    and return a compact merged preview (observation + last-day weather).
    Also stores per-observation feature aggregates in `weather_features`.
    """
    try:
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]
    features_collection = db["weather_features"]

    logger.info(f"Fetching iNaturalist observations from {start_date} to {end_date}")
    pages = await get_pages(start_date, logger=logger)
    observations = get_observations(pages)

    if not observations:
        raise HTTPException(status_code=404, detail="No observations found")

    async def fetch_and_store(obs: dict):
        """For one observation: fetch multi-year weather, store timeseries, compute + store features."""
        lat = obs.get("latitude")
        lon = obs.get("longitude")
        if lat is None or lon is None:
            return None

        obs_dt = pd.to_datetime(obs["time_observed_at"]).date()

        start_hist = (datetime.combine(obs_dt, datetime.min.time()) - timedelta(days=365 * years_back)).date()
        end_hist = obs_dt

        nasa_api = PowerAPI(
            start=start_hist,
            end=end_hist,
            lat=lat,
            long=lon,
        )
        weather_payload = nasa_api.get_weather()
        rows = weather_payload.get("data", [])

        # Upsert full daily time series for this observation window
        for r in rows:
            r["inat_id"] = obs["id"]
            # Normalize date back to string YYYY-MM-DD for Mongo queries
            if isinstance(r.get("date"), (datetime, pd.Timestamp)):
                r["date"] = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
            weather_collection.update_one(
                {"inat_id": obs["id"], "date": r["date"]},
                {"$set": r},
                upsert=True
            )

        # Compute features from the cleaned DF
        df = _clean_weather_df(rows)
        feats = _compute_features(df) if not df.empty else {}

        feature_doc = {
            "inat_id": obs["id"],
            "latitude": lat,
            "longitude": lon,
            "obs_date": obs_dt.strftime("%Y-%m-%d"),
            "years_back": years_back,
            "windows": WINDOWS,
            "features": feats,
        }
        features_collection.update_one(
            {"inat_id": obs["id"]},
            {"$set": feature_doc},
            upsert=True
        )

        # Return a compact preview row (obs + weather on obs_date if present)
        last_day = next((r for r in rows if r.get("date") == obs_dt.strftime("%Y-%m-%d")), None)
        # Strip Mongo ids later at response time if any
        return {
            "observation": obs,
            "weather_on_obs_date": last_day,
            "features": feature_doc["features"],
        }

    merged = await asyncio.gather(*[fetch_and_store(o) for o in observations])
    # Store iNat data (after weather & features)
    for obs in observations:
        inat_collection.update_one({"id": obs["id"]}, {"$set": obs}, upsert=True)

    merged = [m for m in merged if m is not None]

    # Remove _id fields if any
    for m in merged:
        if m is None:
            continue
        if "observation" in m and isinstance(m["observation"], dict):
            m["observation"].pop("_id", None)

    return {"count": len(merged), "preview": merged[:50]}  # cap preview size in response


@router.post("/datasets/refresh-weather")
async def refresh_weather(
    years_back: int = Query(3, ge=1, le=10, description="Recompute using this many years of history")
):
    """
    Refresh multi-year NASA POWER weather & features for all stored iNat observations concurrently.
    """
    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]
    features_collection = db["weather_features"]

    observations = list(inat_collection.find({}, {"_id": 0}))
    if not observations:
        raise HTTPException(status_code=404, detail="No observations found in DB")

    async def refresh_one(obs: dict):
        lat = obs.get("latitude")
        lon = obs.get("longitude")
        if lat is None or lon is None:
            return False
        obs_dt = pd.to_datetime(obs["time_observed_at"]).date()

        start_hist = (datetime.combine(obs_dt, datetime.min.time()) - timedelta(days=365 * years_back)).date()
        end_hist = obs_dt

        nasa_api = PowerAPI(
            start=start_hist,
            end=end_hist,
            lat=lat,
            long=lon
        )
        weather_payload = nasa_api.get_weather()
        rows = weather_payload.get("data", [])

        for r in rows:
            r["inat_id"] = obs["id"]
            if isinstance(r.get("date"), (datetime, pd.Timestamp)):
                r["date"] = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
            weather_collection.update_one(
                {"inat_id": obs["id"], "date": r["date"]},
                {"$set": r},
                upsert=True
            )

        df = _clean_weather_df(rows)
        feats = _compute_features(df) if not df.empty else {}
        features_collection.update_one(
            {"inat_id": obs["id"]},
            {"$set": {
                "inat_id": obs["id"],
                "latitude": lat,
                "longitude": lon,
                "obs_date": obs_dt.strftime("%Y-%m-%d"),
                "years_back": years_back,
                "windows": WINDOWS,
                "features": feats,
            }},
            upsert=True
        )
        return True

    results = await asyncio.gather(*[refresh_one(o) for o in observations])
    return {"updated_weather_records": int(sum(results))}


@router.get("/datasets/export")
def export_dataset(include_features: bool = Query(True, description="Include engineered features in the CSV")):
    """
    Export merged iNat + full weather timeseries (per day) + optional feature aggregates as CSV.
    Note: CSV will be a flattened left-join on `inat_id` with a row per daily weather record.
    """
    db = get_database()
    inat_collection = db["inat_observations"]
    weather_collection = db["weather_data"]
    features_collection = db["weather_features"]

    observations = list(inat_collection.find({}, {"_id": 0}))
    weather = list(weather_collection.find({}, {"_id": 0}))

    if not observations or not weather:
        raise HTTPException(status_code=404, detail="No dataset to export")

    df_inat = pd.DataFrame(observations)
    df_weather = pd.DataFrame(weather)

    # Merge: One row per (inat_id, date) with obs columns on the left
    merged_df = pd.merge(df_inat, df_weather, left_on="id", right_on="inat_id", how="left")

    if include_features:
        feats = list(features_collection.find({}, {"_id": 0}))
        if feats:
            df_feats = pd.DataFrame(feats)
            # Flatten features dict into columns
            if "features" in df_feats.columns:
                feat_expanded = pd.json_normalize(df_feats["features"])
                df_feats = pd.concat([df_feats.drop(columns=["features"]), feat_expanded], axis=1)
            # Join on inat_id (one row per observation’s feature set)
            merged_df = pd.merge(merged_df, df_feats, on="inat_id", how="left", suffixes=("", "_feat"))

    # Export to CSV
    stream = io.StringIO()
    merged_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"}
    )
