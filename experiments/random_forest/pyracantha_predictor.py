
"""
Pyracantha angustifolia Random Forest Predictor (Seasonality + Real-Weather Heatmap)

What’s new vs previous version:
- Absences now include *all non-sighting days* at the same observation locations
  (excluding a ±7-day buffer around the obs date to avoid label ambiguity).
- Map generation can fetch real daily weather via your FastAPI /weather endpoint
  for a given date window, then compute the *same* aggregates as training and predict.
- Seasonality encoded with cyclic features (sin/cos for month & day-of-year).

Assumptions:
- Dataset columns (from your header) include:
  id, uuid, time_observed_at, created_at, latitude_x, longitude_x, ...,
  date, obs_date, daily weather & aggregates like ALLSKY_SFC_SW_DWN, PRECTOTCORR, ...,
  rain_sum_7, t2m_mean_7, ..., rain_sum_365, ...
- Presence = rows where date == obs_date.
- Absence = rows where date != obs_date at the *same inat_id/location*, except within ±7 days of obs_date.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Optional deps for async HTTP; we gracefully fallback if not installed
try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None
import math


class PyracanthaRandomForestPredictor:
    """
    Random Forest predictor for Pyracantha angustifolia presence.

    Pipeline:
      1) Load & preprocess (adds month/day_of_year and cyclic encodings)
      2) Build presence/absence dataset (no leakage; all non-sighting days used as absences)
      3) Feature selection (schema-driven)
      4) Train RF + OOB/CV/Test evaluation
      5) Visualizations (feature importance, ROC, confusion, metrics)
      6) Spatial prediction heatmap (optionally uses your FastAPI /weather)
      7) Save artifacts
    """

    def __init__(
        self,
        random_state: int = 42,
        absence_buffer_days: int = 7,
        max_absence_multiplier: float = 10.0,  # cap absences to 10x positives (optional)
    ):
        self.random_state = random_state
        self.absence_buffer_days = absence_buffer_days
        self.max_absence_multiplier = max_absence_multiplier

        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns: Optional[List[str]] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.training_stats: Dict = {}
        self.lat_col: Optional[str] = None
        self.lon_col: Optional[str] = None

    # -------------------------
    # 1) Load & Preprocess Data
    # -------------------------
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # Parse dates
        if "date" not in df.columns or "obs_date" not in df.columns:
            raise ValueError("Expected 'date' and 'obs_date' columns in the dataset.")
        df["date"] = pd.to_datetime(df["date"])
        df["obs_date"] = pd.to_datetime(df["obs_date"])

        # Robust lat/lon detection
        self.lat_col = "latitude_x" if "latitude_x" in df.columns else ("latitude" if "latitude" in df.columns else None)
        self.lon_col = "longitude_x" if "longitude_x" in df.columns else ("longitude" if "longitude" in df.columns else None)
        if not self.lat_col or not self.lon_col:
            raise ValueError("Latitude/Longitude columns not found. Expected latitude_x/longitude_x or latitude/longitude.")

        # Temporal context
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        # Cyclic encodings for seasonality
        df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

        # Numeric NA imputation (median per column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        # Basic dataset stats
        self.training_stats["total_records"] = int(len(df))
        self.training_stats["unique_observations"] = int(df["inat_id"].nunique()) if "inat_id" in df.columns else None

        return df

    # -------------------------------------------------------
    # 2) Presence/Absence: include ALL non-sighting days as 0
    # -------------------------------------------------------
    def create_training_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Presence: rows where date == obs_date.
        Absence: rows where date != obs_date for the same inat_id/location,
                 excluding a ±buffer window around obs_date.
        Optionally cap the number of absences to max_absence_multiplier * positives.
        """
        print("Constructing presence/absence dataset (all non-sighting days as absences)...")

        if "inat_id" not in df.columns:
            raise ValueError("Expected 'inat_id' column to group by observations.")

        # Presence rows
        presence = df[df["date"] == df["obs_date"]].copy()
        presence["target"] = 1
        n_pos = len(presence)
        if n_pos == 0:
            raise ValueError("No presence rows found (date == obs_date). Check your data.")

        # Build absence pool: all other rows per inat_id excluding ±buffer around obs_date
        buffer = pd.Timedelta(days=self.absence_buffer_days)
        df["_is_absence_candidate"] = False

        # Efficient mask construction
        # For each inat_id, mark all rows not in [obs_date - buffer, obs_date + buffer] AND date != obs_date
        obs_dates = df[["inat_id", "obs_date"]].drop_duplicates()
        df = df.merge(obs_dates, on="inat_id", suffixes=("", "_per_id"))
        # mask for outside the buffer
        outside = (df["date"] < (df["obs_date_per_id"] - buffer)) | (df["date"] > (df["obs_date_per_id"] + buffer))
        not_obs_day = df["date"] != df["obs_date_per_id"]
        df["_is_absence_candidate"] = outside & not_obs_day

        absence_pool = df[df["_is_absence_candidate"]].copy()
        absence_pool = absence_pool.drop(columns=["_is_absence_candidate", "obs_date_per_id"])

        # If pool is huge, optionally downsample
        max_abs = int(self.max_absence_multiplier * n_pos) if self.max_absence_multiplier else len(absence_pool)
        if len(absence_pool) > max_abs:
            absence = absence_pool.sample(n=max_abs, random_state=self.random_state).copy()
        else:
            absence = absence_pool.copy()
        absence["target"] = 0

        # Combine and shuffle
        final_df = pd.concat([presence, absence], ignore_index=True)
        final_df = final_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        # Drop leakage/ID/date columns we do not want the model to use
        drop_cols = {
            "obs_date",
            "time_observed_at",
            "created_at",
            "uuid",
            "id",
            "user_id",
        }
        final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns], errors="ignore")

        # Store class distribution
        self.training_stats["final_samples"] = int(len(final_df))
        self.training_stats["presence_samples"] = int((final_df["target"] == 1).sum())
        self.training_stats["absence_samples"] = int((final_df["target"] == 0).sum())

        print(
            f"Final dataset: {len(final_df):,} rows | "
            f"Positives: {self.training_stats['presence_samples']:,} | "
            f"Negatives: {self.training_stats['absence_samples']:,}"
        )
        return final_df

    # -------------------
    # 3) Feature Selection
    # -------------------
    def select_features(self, df: pd.DataFrame) -> List[str]:
        location = [self.lat_col, self.lon_col, "elevation"]

        daily_weather = [
            "ALLSKY_SFC_SW_DWN",
            "CLRSKY_SFC_SW_DWN",
            "PRECTOTCORR",
            "RH2M",
            "T2M",
            "T2M_MAX",
            "T2M_MIN",
            "TQV",
            "TS",
            "WS2M",
        ]

        temporal_prefixes = [
            "rain_sum",
            "t2m_mean",
            "t2m_max",
            "t2m_min",
            "rh2m_mean",
            "wind_mean",
            "cloud_index_mean",
            "gdd_base10_sum",
            "heat_days_gt30",
            "frost_days_lt0",
        ]
        temporal_features = [c for c in df.columns if any(c.startswith(p) for p in temporal_prefixes)]

        temporal_context = [c for c in ["month", "day_of_year"] if c in df.columns]
        cyclic_seasonal = [c for c in ["sin_month", "cos_month", "sin_doy", "cos_doy"] if c in df.columns]

        all_features = location + daily_weather + temporal_features + temporal_context + cyclic_seasonal
        available = [c for c in all_features if c in df.columns]

        self.training_stats["total_features"] = int(len(available))
        self.training_stats["feature_categories"] = {
            "location": int(len([f for f in location if f in available])),
            "daily_weather": int(len([f for f in daily_weather if f in available])),
            "temporal_aggregates": int(len([f for f in temporal_features if f in available])),
            "temporal_context": int(len([f for f in temporal_context if f in available])),
            "cyclic_seasonal": int(len([f for f in cyclic_seasonal if f in available])),
        }

        print(f"Selected {len(available)} features (incl. cyclic seasonality).")
        return available

    # ---------------
    # 4) Train & Eval
    # ---------------
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        # Balance could be skewed—stratify split, class_weight handles imbalance
        print(f"Training Random Forest (test_size={test_size:.0%})...")

        self.feature_columns = self.select_features(df)
        X = df[self.feature_columns]
        y = df["target"]

        if y.nunique() != 2 or y.value_counts().min() < 2:
            raise ValueError("Invalid class distribution for stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        self.model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            oob_score=True,
            random_state=self.random_state,
        )

        print("Fitting model...")
        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auc = roc_auc_score(y_test, y_test_proba)
        cm = confusion_matrix(y_test, y_test_pred)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="roc_auc")

        results = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "auc_score": float(auc),
            "oob_score": float(self.model.oob_score_),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
        }

        self.feature_importance_ = (
            pd.DataFrame({"feature": self.feature_columns, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        self.training_stats.update(
            {
                "training_date": datetime.now().isoformat(),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "model_params": self.model.get_params(),
            }
        )

        print(
            f"Done. Test Acc: {results['test_accuracy']:.3f} | AUC: {results['auc_score']:.3f} | "
            f"OOB: {results['oob_score']:.3f} | CV AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}"
        )
        return results

    # --------------------
    # 5) Visualizations
    # --------------------
    def create_visualizations(self, results: Dict, output_dir: Path) -> None:
        print("Creating visualizations...")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Feature importance
        top = self.feature_importance_.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Top 20 Features — Pyracantha Presence (Random Forest)")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fpr = np.array(results["roc_curve"]["fpr"])
        tpr = np.array(results["roc_curve"]["tpr"])
        ax = axes[0, 0]
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
        ax.plot(fpr, tpr)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve (AUC = {results['auc_score']:.3f})"); ax.grid(alpha=0.3)

        cm = np.array(results["confusion_matrix"])
        ax = axes[0, 1]
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center")
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1, 0]
        cats = self.training_stats.get("feature_categories", {})
        labels = list(cats.keys()); counts = list(cats.values())
        if sum(counts) > 0:
            ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title("Feature Categories")
        else:
            ax.text(0.5, 0.5, "No feature categories found", ha="center", va="center"); ax.axis("off")

        ax = axes[1, 1]
        metrics = ["Train Acc", "Test Acc", "AUC", "OOB", "CV AUC"]
        values = [
            results["train_accuracy"],
            results["test_accuracy"],
            results["auc_score"],
            results["oob_score"],
            results["cv_auc_mean"],
        ]
        bars = ax.bar(metrics, values); ax.set_ylim([0, 1])
        ax.set_title("Model Metrics"); ax.set_ylabel("Score")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        fig.suptitle("Pyracantha RF — Performance Summary", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "model_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_dir / 'feature_importance.png'}")
        print(f"Saved: {output_dir / 'model_performance.png'}")

    # -------------------------------------------------------
    # Helpers: compute aggregates exactly like training needs
    # -------------------------------------------------------
    @staticmethod
    def _compute_rolling_aggregates(daily: pd.DataFrame, end_date: pd.Timestamp) -> Dict[str, float]:
        """
        Given daily weather rows with columns matching your schema and a target end_date,
        compute rolling windows (7, 30, 90, 365) and return a dict of features.
        The windows are strictly backward-looking and include the end_date.
        """
        if daily.empty:
            return {}

        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        daily = daily[daily["date"] <= end_date]

        def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
            start = end_date - pd.Timedelta(days=days - 1)
            return df[(df["date"] >= start) & (df["date"] <= end_date)]

        out: Dict[str, float] = {}

        for win in [7, 30, 90, 365]:
            w = window(daily, win)
            if w.empty:
                # Populate 0 or NaN? Use 0 to be conservative.
                out[f"rain_sum_{win}"] = 0.0
                out[f"t2m_mean_{win}"] = 0.0
                out[f"t2m_max_{win}"] = 0.0
                out[f"t2m_min_{win}"] = 0.0
                out[f"rh2m_mean_{win}"] = 0.0
                out[f"wind_mean_{win}"] = 0.0
                out[f"cloud_index_mean_{win}"] = 0.0
                out[f"gdd_base10_sum_{win}"] = 0.0
                out[f"heat_days_gt30_{win}"] = 0.0
                out[f"frost_days_lt0_{win}"] = 0.0
                continue

            # Core fields expected in your dataset
            # PRECTOTCORR (mm/day), T2M (°C), RH2M (%), WS2M (m/s),
            # radiation proxies: ALLSKY_SFC_SW_DWN vs CLRSKY_SFC_SW_DWN
            rsum = w.get("PRECTOTCORR", pd.Series([0])).sum()
            tmean = w.get("T2M", pd.Series([0])).mean()
            tmax = w.get("T2M_MAX", pd.Series([0])).max()
            tmin = w.get("T2M_MIN", pd.Series([0])).min()
            rhmean = w.get("RH2M", pd.Series([0])).mean()
            wmean = w.get("WS2M", pd.Series([0])).mean()

            # Cloud index ~ 1 - (ALLSKY/CLRSKY) if both present and CLRSKY>0
            if "ALLSKY_SFC_SW_DWN" in w.columns and "CLRSKY_SFC_SW_DWN" in w.columns:
                ci = 1.0 - (w["ALLSKY_SFC_SW_DWN"] / w["CLRSKY_SFC_SW_DWN"].replace(0, np.nan))
                cloud_idx = float(np.nanmean(ci))
                if np.isnan(cloud_idx):
                    cloud_idx = 0.0
            else:
                cloud_idx = 0.0

            # Growing degree days base 10: sum(max(T2M-10,0))
            gdd = (w.get("T2M", pd.Series([0])) - 10.0).clip(lower=0).sum()

            heat_days = (w.get("T2M_MAX", pd.Series([0])) > 30.0).sum()
            frost_days = (w.get("T2M_MIN", pd.Series([0])) < 0.0).sum()

            out.update(
                {
                    f"rain_sum_{win}": float(rsum),
                    f"t2m_mean_{win}": float(tmean),
                    f"t2m_max_{win}": float(tmax),
                    f"t2m_min_{win}": float(tmin),
                    f"rh2m_mean_{win}": float(rhmean),
                    f"wind_mean_{win}": float(wmean),
                    f"cloud_index_mean_{win}": float(cloud_idx),
                    f"gdd_base10_sum_{win}": float(gdd),
                    f"heat_days_gt30_{win}": float(heat_days),
                    f"frost_days_lt0_{win}": float(frost_days),
                }
            )

        return out

    # ------------------------------------------------------------
    # 6) Spatial Heatmap — with optional FastAPI weather endpoint
    # ------------------------------------------------------------
    async def _fetch_weather_async(
        self,
        session: aiohttp.ClientSession,
        endpoint_url: str,
        lat: float,
        lon: float,
        start: datetime,
        end: datetime,
    ) -> Dict:
        params = dict(
            latitude=lat,
            longitude=lon,
            start_year=start.year,
            start_month=start.month,
            start_day=start.day,
            end_year=end.year,
            end_month=end.month,
            end_day=end.day,
            store_in_db="false",
        )
        async with session.get(endpoint_url, params=params, timeout=300) as resp:
            resp.raise_for_status()
            return await resp.json()
        
        
    def _build_feature_row_from_weather(
        self,
        daily: pd.DataFrame,
        lat: float,
        lon: float,
        elevation_from_api: Optional[float],
        end_date: datetime,
    ) -> Dict[str, float]:
        """
        Build a single feature row using the same column names as training.
        Prefer elevation from API metadata if provided.
        """
        feats: Dict[str, float] = {}

        # Ensure datetime
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        # Snapshot on end_date (or last available <= end_date)
        snap = daily[daily["date"] == pd.to_datetime(end_date)]
        if snap.empty:
            snap = daily[daily["date"] <= pd.to_datetime(end_date)].tail(1)
        snap_row = snap.iloc[0].to_dict() if not snap.empty else {}

        # Daily weather fields (match training names)
        for k in [
            "ALLSKY_SFC_SW_DWN",
            "CLRSKY_SFC_SW_DWN",
            "PRECTOTCORR",
            "RH2M",
            "T2M",
            "T2M_MAX",
            "T2M_MIN",
            "TQV",
            "TS",
            "WS2M",
        ]:
            feats[k] = float(snap_row.get(k, 0.0))

        # Location
        feats["latitude_x"] = float(lat)
        feats["longitude_x"] = float(lon)

        # Elevation: prefer API meta, then per-day, else synthetic  (keeps spatial variation when POWER is uniform)
        if elevation_from_api is not None:
            feats["elevation"] = float(elevation_from_api)
        else:
            feats["elevation"] = float(snap_row.get("elevation", 200.0))

        # Temporal context + cyclic encodings
        month = end_date.month
        doy = end_date.timetuple().tm_yday
        feats["month"] = month
        feats["day_of_year"] = doy
        feats["sin_month"] = math.sin(2 * math.pi * (month - 1) / 12.0)
        feats["cos_month"] = math.cos(2 * math.pi * (month - 1) / 12.0)
        feats["sin_doy"] = math.sin(2 * math.pi * doy / 365.0)
        feats["cos_doy"] = math.cos(2 * math.pi * doy / 365.0)

        # Rolling aggregates
        aggs = self._compute_rolling_aggregates(daily, pd.to_datetime(end_date))
        feats.update(aggs)

        return feats


    def _fallback_feature_row_from_median(
        self,
        df_train: pd.DataFrame,
        lat: float,
        lon: float,
        end_date: datetime,
    ) -> Dict[str, float]:
        """
        If no endpoint is provided, build a 'median-conditions' feature row.
        """
        feats: Dict[str, float] = {}
        base = df_train[self.feature_columns].median(numeric_only=True)

        for feat in self.feature_columns:
            feats[feat] = float(base.get(feat, 0.0))

        # Override with the grid point coordinates and current temporal encodings
        feats["latitude_x"] = float(lat)
        feats["longitude_x"] = float(lon)

        month = end_date.month
        doy = end_date.timetuple().tm_yday
        feats["month"] = month
        feats["day_of_year"] = doy
        feats["sin_month"] = math.sin(2 * math.pi * (month - 1) / 12.0)
        feats["cos_month"] = math.cos(2 * math.pi * (month - 1) / 12.0)
        feats["sin_doy"] = math.sin(2 * math.pi * doy / 365.0)
        feats["cos_doy"] = math.cos(2 * math.pi * doy / 365.0)

        return feats

    def create_grid_heatmap(
        self,
        df_for_medians: pd.DataFrame,
        output_dir: Path,
        grid_resolution: float = 0.01,
        prediction_date: Optional[str] = None,
        weather_endpoint_url: Optional[str] = None,  # e.g. "http://127.0.0.1:8000/api/v1/weather"
        weather_years_back: int = 1,
        max_cells: int = 400,
        concurrent_requests: int = 10,
    ) -> str:
        """
        If weather_endpoint_url is provided:
        * Round each tile's lat/lon to 0.5° (POWER grid), fetch once per unique cell,
            then broadcast the prediction to all tiles inside that cell.
        * Use elevation from API 'location.elevation' if present.
        Else:
        * Fall back to median-conditions sketch.
        Also applies quantile color stretch to make subtle variation visible.
        """
        print("Generating grid heatmap...")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Dates
        if prediction_date is None:
            # Prefer timezone-aware UTC if available
            try:
                from datetime import UTC
                end_date_dt = datetime.now(UTC)
            except Exception:
                from datetime import timezone
                end_date_dt = datetime.now(timezone.utc)
        else:
            end_date_dt = pd.to_datetime(prediction_date)
        end_date = end_date_dt.date()
        start_date = end_date - timedelta(days=365 * max(1, weather_years_back))

        # Bounds from data
        lat_min, lat_max = df_for_medians[self.lat_col].min(), df_for_medians[self.lat_col].max()
        lon_min, lon_max = df_for_medians[self.lon_col].min(), df_for_medians[self.lon_col].max()

        # Padding
        lat_pad = (lat_max - lat_min) * 0.1
        lon_pad = (lon_max - lon_min) * 0.1
        lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad
        lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad

        # Fine display grid (visual resolution)
        lat_grid = np.arange(lat_min, lat_max, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max, grid_resolution)

        # Down-sample display tiles for performance
        step_lat = max(1, int(np.ceil(len(lat_grid) / np.sqrt(max_cells))))
        step_lon = max(1, int(np.ceil(len(lon_grid) / np.sqrt(max_cells))))
        lat_idx = list(range(0, len(lat_grid), step_lat))
        lon_idx = list(range(0, len(lon_grid), step_lon))
        print(f"Grid (down-sampled display): {len(lat_idx)} × {len(lon_idx)} (~{len(lat_idx)*len(lon_idx)})")

        # Helper: POWER cell rounding (0.5° grid)
        def round_power(x: float) -> float:
            return round(x * 2.0) / 2.0

        coords_display: List[Tuple[float, float]] = [(float(lat_grid[i]), float(lon_grid[j])) for i in lat_idx for j in lon_idx]

        # Build features
        rows: List[Dict[str, float]] = []
        if weather_endpoint_url:
            if aiohttp is None:
                print("aiohttp not installed; falling back to median conditions for heatmap.")
                weather_endpoint_url = None

        if weather_endpoint_url:
            print(f"Fetching real weather via: {weather_endpoint_url}")

            # Unique POWER cells to fetch
            power_cells = {(round_power(lat), round_power(lon)) for (lat, lon) in coords_display}
            power_cells = list(power_cells)

            # Map cell -> daily weather df + elevation meta
            cell_to_feats: Dict[Tuple[float, float], Dict[str, float]] = {}

            async def run_fetches():
                conn = aiohttp.TCPConnector(limit=concurrent_requests)
                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                    tasks = []
                    for (plat, plon) in power_cells:
                        tasks.append(self._fetch_weather_async(
                            session,
                            weather_endpoint_url,
                            plat,
                            plon,
                            datetime(start_date.year, start_date.month, start_date.day),
                            datetime(end_date.year, end_date.month, end_date.day),
                        ))
                    return await asyncio.gather(*tasks, return_exceptions=True)

            results = asyncio.run(run_fetches())

            # Build per-cell feature rows
            for (plat, plon), res in zip(power_cells, results):
                if isinstance(res, Exception):
                    print(f"WARNING: API error at ({plat},{plon}); using median fallback.")
                    cell_to_feats[(plat, plon)] = None
                    continue

                # The example shows metadata at top-level .location and daily list in .data
                elevation_meta = None
                if isinstance(res.get("location", {}), dict):
                    if "elevation" in res["location"]:
                        elevation_meta = float(res["location"]["elevation"])

                daily = pd.DataFrame(res.get("data", []))
                if daily.empty:
                    cell_to_feats[(plat, plon)] = None
                    continue

                if "date" not in daily.columns:
                    # Try YEAR/MO/DY
                    if all(c in daily.columns for c in ["YEAR", "MO", "DY"]):
                        daily["date"] = pd.to_datetime(daily["YEAR"].astype(str) + "-" +
                                                    daily["MO"].astype(str) + "-" +
                                                    daily["DY"].astype(str))
                    else:
                        print(f"WARNING: Missing 'date' in API response for ({plat},{plon}); using median fallback.")
                        cell_to_feats[(plat, plon)] = None
                        continue

                # Minimal field coverage check
                required = ['T2M','T2M_MAX','T2M_MIN','PRECTOTCORR','RH2M','WS2M',
                            'ALLSKY_SFC_SW_DWN','CLRSKY_SFC_SW_DWN']
                missing = [k for k in required if k not in daily.columns]
                if missing:
                    print(f"WARNING: API missing fields at ({plat},{plon}): {missing} — zeros will be used; map may flatten.")

                feat_row = self._build_feature_row_from_weather(
                    daily=daily,
                    lat=plat,
                    lon=plon,
                    elevation_from_api=elevation_meta,
                    end_date=datetime(end_date.year, end_date.month, end_date.day),
                )
                cell_to_feats[(plat, plon)] = feat_row

            # Now broadcast to each display tile by its POWER cell
            for (lat, lon) in coords_display:
                key = (round_power(lat), round_power(lon))
                row = cell_to_feats.get(key)
                if row is None:
                    # Median fallback for this cell
                    row = self._fallback_feature_row_from_median(
                        df_for_medians, lat, lon, datetime(end_date.year, end_date.month, end_date.day)
                    )
                # overwrite to local lat/lon for nicer popups
                row = dict(row)
                row["latitude_x"] = float(lat)
                row["longitude_x"] = float(lon)
                rows.append(row)

        else:
            # Median-conditions sketch
            for (lat, lon) in coords_display:
                row = self._fallback_feature_row_from_median(
                    df_for_medians, lat, lon, datetime(end_date.year, end_date.month, end_date.day)
                )
                rows.append(row)

        feat_df = pd.DataFrame(rows)
        # Ensure required features exist
        for f in self.feature_columns:
            if f not in feat_df.columns:
                feat_df[f] = 0.0

        if self.model is None:
            raise ValueError("Model not trained.")
        inv_risk = self.model.predict_proba(feat_df[self.feature_columns])[:, 1]

        # Diagnostics
        print(f"Risk stats -> min: {inv_risk.min():.4f}, max: {inv_risk.max():.4f}, mean: {inv_risk.mean():.4f}")
        print(f"Unique risks (rounded 3dp): {len(np.unique(np.round(inv_risk, 3)))}")

        # Quantile stretch for color
        qlo, qhi = np.quantile(inv_risk, [0.05, 0.95])
        def color_from_risk(r: float) -> str:
            if qhi <= qlo:
                x = 0.5
            else:
                x = 0.0 if r <= qlo else (1.0 if r >= qhi else (r - qlo) / (qhi - qlo))
            R = int(255 * x)
            G = int(255 * (1 - x))
            return f"#{R:02x}{G:02x}00"

        # Folium map
        center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
        m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")

        title_html = f"""
        <h3 align="center" style="font-size:18px"><b>Pyracantha Invasion Risk Heatmap</b></h3>
        <p align="center" style="font-size:12px">
        Prediction date: {end_date} | Source: {'API (POWER grid 0.5°)' if weather_endpoint_url else 'Median conditions'}
        </p>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Paint rectangles
        k = 0
        for ii, i in enumerate(lat_idx[:-1]):
            for jj, j in enumerate(lon_idx[:-1]):
                lat, lon = float(lat_grid[i]), float(lon_grid[j])
                risk = float(inv_risk[k]); k += 1
                color_hex = color_from_risk(risk)
                bounds = [
                    [lat_grid[i], lon_grid[j]],
                    [lat_grid[lat_idx[ii + 1]] if ii + 1 < len(lat_idx) else lat_grid[-1],
                    lon_grid[lon_idx[jj + 1]] if jj + 1 < len(lon_idx) else lon_grid[-1]],
                ]
                folium.Rectangle(
                    bounds=bounds,
                    color=color_hex,
                    fillColor=color_hex,
                    fillOpacity=0.6,
                    weight=0,
                    popup=f"Risk: {risk:.3f}\nLat: {lat:.3f}\nLon: {lon:.3f}",
                ).add_to(m)

        # Overlay observation points
        obs_locations = df_for_medians.groupby([self.lat_col, self.lon_col]).size().reset_index(name="count")
        for _, row in obs_locations.iterrows():
            folium.CircleMarker(
                location=[row[self.lat_col], row[self.lon_col]],
                radius=6,
                popup=f"Observations: {int(row['count'])}",
                color="black",
                fillColor="white",
                fillOpacity=0.9,
                weight=2,
            ).add_to(m)

        # Legend
        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; width: 260px; background: white;
                    border: 2px solid grey; z-index: 9999; font-size: 12px; padding: 10px; border-radius: 8px;">
            <b>Legend</b><br/>
            <div><span style="display:inline-block;width:18px;height:12px;background:#00ff00;border:1px solid #999;margin-right:6px;"></span>Low</div>
            <div><span style="display:inline-block;width:18px;height:12px;background:#ffff00;border:1px solid #999;margin-right:6px;"></span>Medium</div>
            <div><span style="display:inline-block;width:18px;height:12px;background:#ff0000;border:1px solid #999;margin-right:6px;"></span>High</div>
            <hr style="margin:6px 0;"/>
            Tiles: {len(lat_idx)} × {len(lon_idx)} (~{len(lat_idx)*len(lon_idx)})<br/>
            API mode: {"On" if weather_endpoint_url else "Off"} (POWER ~0.5°)<br/>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save
        try:
            from datetime import UTC
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        except Exception:
            from datetime import timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        out_path = output_dir / f"pyracantha_invasion_heatmap_{ts}.html"
        m.save(str(out_path))
        print(f"Heatmap saved: {out_path}")
        return str(out_path)


    # ---------------------------
    # 7) Save model & JSON report
    # ---------------------------
    def save_model_and_results(self, results: Dict, output_dir: Path) -> None:
        print("Saving model and results...")
        output_dir.mkdir(exist_ok=True, parents=True)

        model_blob = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "feature_importance_": self.feature_importance_,
            "training_stats": self.training_stats,
        }
        model_path = output_dir / "pyracantha_random_forest_model.pkl"
        joblib.dump(model_blob, model_path, compress=3)

        report = {
            "training_stats": self.training_stats,
            "evaluation_results": results,
            "feature_importance": self.feature_importance_.to_dict("records"),
            "model_parameters": self.model.get_params() if self.model else None,
        }
        results_path = output_dir / "evaluation_report.json"
        results_path.write_text(json.dumps(report, indent=2, default=str))

        print(f"Model saved:   {model_path}")
        print(f"Report saved:  {results_path}")


def main():
    print("=" * 80)
    print("PYRACANTHA ANGUSTIFOLIA — RANDOM FOREST PREDICTION (Seasonality + Real-Weather Heatmap)")
    print("=" * 80)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # Paths
    data_path = "../../data/dataset.csv"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Heatmap config — edit as needed
    # Example: use April 15 to reflect peak season, or July 15 for of season
    prediction_date = "2024-04-15" # 2024-07-15
    weather_endpoint_url = "http://127.0.0.1:8000/api/v1/weather"
    weather_years_back = 1
    grid_resolution = 0.01
    max_cells = 400

    predictor = PyracanthaRandomForestPredictor(
        random_state=42,
        absence_buffer_days=7,
        max_absence_multiplier=10.0,
    )

    try:
        # 1) Load
        print("STEP 1: LOAD & PREPROCESS")
        df = predictor.load_and_preprocess_data(data_path)
        print()

        # 2) Label (all non-sighting days as absences, buffer ±7d)
        print("STEP 2: BUILD TRAINING DATASET (ALL NON-SIGHTING DAYS AS ABSENCES)")
        train_df = predictor.create_training_dataset(df)
        print()

        # 3) Train
        print("STEP 3: TRAIN & EVALUATE")
        results = predictor.train_model(train_df, test_size=0.2)
        print()

        # 4) Visualize
        print("STEP 4: VISUALIZATIONS")
        predictor.create_visualizations(results, output_dir)
        print()

        # 5) Heatmap (real weather via endpoint if provided, else median fallback)
        print("STEP 5: GRID HEATMAP")
        heatmap_path = predictor.create_grid_heatmap(
            df_for_medians=train_df,
            output_dir=output_dir,
            grid_resolution=grid_resolution,
            prediction_date=prediction_date,
            weather_endpoint_url=weather_endpoint_url,
            weather_years_back=weather_years_back,
            max_cells=max_cells,
            concurrent_requests=10
        )
        print()

        # 6) Save
        print("STEP 6: SAVE ARTIFACTS")
        predictor.save_model_and_results(results, output_dir)
        print()

        print("COMPLETED\n")
        print("RESULTS SUMMARY")
        print(f"  • Rows used:        {predictor.training_stats['final_samples']:,}")
        print(f"  • Presences:        {predictor.training_stats['presence_samples']:,}")
        print(f"  • Absences:         {predictor.training_stats['absence_samples']:,}")
        print(f"  • Features:         {predictor.training_stats['total_features']}")
        print(f"  • Train samples:    {predictor.training_stats['train_samples']:,}")
        print(f"  • Test samples:     {predictor.training_stats['test_samples']:,}")
        print("  • Performance:")
        print(f"      - Test Acc:     {results['test_accuracy']:.4f}")
        print(f"      - AUC:          {results['auc_score']:.4f}")
        print(f"      - OOB:          {results['oob_score']:.4f}")
        print(f"      - CV AUC:       {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        print()
        print("TOP 5 FEATURES")
        for i, row in predictor.feature_importance_.head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        print()
        print("FILES")
        print(f"  • Model:        {output_dir / 'pyracantha_random_forest_model.pkl'}")
        print(f"  • Heatmap:      {heatmap_path}")
        print(f"  • Figures dir:  {output_dir}")
        print(f"  • Report:       {output_dir / 'evaluation_report.json'}\n")

    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()