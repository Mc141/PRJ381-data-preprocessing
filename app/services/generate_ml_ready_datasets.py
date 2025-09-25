"""
Generate ML-ready datasets (global_training_ml_ready.csv and local_validation_ml_ready.csv)
without using the database. This module fetches GBIF occurrences directly, enriches them
with WorldClim climate variables and SRTM elevation, computes temporal features, and writes
the standardized 17-feature CSVs into the repository's `data/` folder (overwriting if they exist).

Usage (from repo root):
  python -m app.services.generate_ml_ready_datasets \
    --max-global 2000 --max-local 500 --batch-size 100 --verbose

Notes:
- Requires WorldClim GeoTIFFs under data/worldclim/ (see app/services/worldclim_extractor.py)
- Uses Open-Topo-Data for elevation via the elevation extractor (internet required)
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is importable when running from anywhere.
THIS_FILE = Path(__file__).resolve()
# app/services/<file> -> parents[0]=services, [1]=app, [2]=repo root
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.gbif_fetcher import GBIFFetcher  # noqa: E402
from app.services.worldclim_extractor import get_worldclim_extractor  # noqa: E402
from app.services.elevation_extractor import get_elevation_extractor  # noqa: E402

logger = logging.getLogger("generate_ml_ready_datasets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REQUIRED_FEATURES = [
    "latitude",
    "longitude",
    "elevation",
    "bio1",
    "bio4",
    "bio5",
    "bio6",
    "bio12",
    "bio13",
    "bio14",
    "bio15",
    "month",
    "day_of_year",
    "sin_month",
    "cos_month",
]

CLIMATE_VARIABLES = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]


def _to_day_of_year(year: Optional[int], month: Optional[int], day: Optional[int], event_date: Optional[str]) -> Optional[int]:
    try:
        if year and month and day:
            return int(pd.Timestamp(year=int(year), month=int(month), day=int(day)).day_of_year)
        if event_date:
            return int(pd.to_datetime(event_date, errors="coerce").day_of_year)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _compute_temporal_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    month = rec.get("month")
    if month is None and rec.get("event_date"):
        try:
            ts = pd.to_datetime(rec["event_date"], errors="coerce")
            if pd.notna(ts):
                month = int(ts.month)
        except Exception:  # pragma: no cover
            month = None

    day_of_year = _to_day_of_year(rec.get("year"), rec.get("month"), rec.get("day"), rec.get("event_date"))

    sin_m = cos_m = None
    if month is not None and 1 <= int(month) <= 12:
        angle = 2 * math.pi * (int(month) - 1) / 12.0
        sin_m = math.sin(angle)
        cos_m = math.cos(angle)

    return {
        "month": int(month) if month is not None else None,
        "day_of_year": int(day_of_year) if day_of_year is not None else None,
        "sin_month": sin_m,
        "cos_month": cos_m,
    }


async def _fetch_occurrences(max_records: Optional[int], country: Optional[str]) -> List[Dict[str, Any]]:
    async def _noop_progress(current: int, total: int, percentage: float):  # noqa: ARG001
        return None

    async with GBIFFetcher() as fetcher:
        occurrences = await fetcher.fetch_all_occurrences(
            scientific_name="Pyracantha angustifolia (Franch.) C.K.Schneid.",
            quality_filters=True,
            coordinate_uncertainty_max=10000,
            max_records=max_records,
            progress_callback=_noop_progress,
            country=country,
        )

    processed: List[Dict[str, Any]] = []
    for r in occurrences:
        try:
            processed.append(
                {
                    "latitude": r.get("decimalLatitude"),
                    "longitude": r.get("decimalLongitude"),
                    "event_date": r.get("eventDate"),
                    "year": r.get("year"),
                    "month": r.get("month"),
                    "day": r.get("day"),
                }
            )
        except Exception:  # pragma: no cover
            continue

    processed = [r for r in processed if r.get("latitude") is not None and r.get("longitude") is not None]
    return processed


async def _enrich_environmental(records: List[Dict[str, Any]], batch_size: int, verbose: bool) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=REQUIRED_FEATURES)

    coords: List[Tuple[float, float]] = [(float(r["latitude"]), float(r["longitude"])) for r in records]

    worldclim = get_worldclim_extractor()
    elevation = get_elevation_extractor()

    climate_rows: List[Dict[str, Any]] = []
    elevation_rows: List[Dict[str, Any]] = []

    async with worldclim, elevation:
        for i in range(0, len(coords), batch_size):
            chunk = coords[i : i + batch_size]
            if verbose:
                logger.info(f"Processing environmental batch {i + 1}-{i + len(chunk)} of {len(coords)}")
            climate_task = worldclim.extract_climate_batch(chunk, CLIMATE_VARIABLES)
            elev_task = elevation.extract_elevation_batch(chunk)
            c_res, e_res = await asyncio.gather(climate_task, elev_task)
            climate_rows.extend(c_res)
            elevation_rows.extend(e_res)

    df_env = pd.DataFrame(climate_rows)
    if "elevation" not in df_env.columns:
        df_env["elevation"] = np.nan

    elev_vals = []
    for e in elevation_rows:
        if isinstance(e, dict) and e is not None:
            elev_vals.append(e.get("elevation"))
        else:
            elev_vals.append(None)

    if len(elev_vals) != len(df_env):
        needed = len(df_env) - len(elev_vals)
        if needed > 0:
            elev_vals.extend([None] * needed)
        else:
            elev_vals = elev_vals[: len(df_env)]

    df_env["elevation"] = elev_vals
    df_env["latitude"] = [c[0] for c in coords]
    df_env["longitude"] = [c[1] for c in coords]

    temporal_records = [_compute_temporal_features(r) for r in records]
    df_tmp = pd.DataFrame(temporal_records)
    df = pd.concat([df_env.reset_index(drop=True), df_tmp.reset_index(drop=True)], axis=1)

    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_FEATURES]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=CLIMATE_VARIABLES + ["month", "sin_month", "cos_month"]).copy()
    return df


def _write_csv(df: pd.DataFrame, path: Path, verbose: bool) -> None:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    if verbose:
        logger.info(f"Wrote {len(df):,} rows -> {path}")


async def run(max_global: Optional[int], max_local: Optional[int], batch_size: int, verbose: bool) -> None:
    data_dir = REPO_ROOT / "data"
    global_path = data_dir / "global_training_ml_ready.csv"
    local_path = data_dir / "local_validation_ml_ready.csv"

    if verbose:
        logger.info("Fetching global occurrences (quality-filtered)…")
    global_occ = await _fetch_occurrences(max_records=max_global, country=None)
    if verbose:
        logger.info(f"Fetched {len(global_occ):,} global occurrences with valid coordinates")

    if verbose:
        logger.info("Fetching local (South Africa) occurrences…")
    local_occ = await _fetch_occurrences(max_records=max_local, country="ZA")
    if verbose:
        logger.info(f"Fetched {len(local_occ):,} local occurrences with valid coordinates")

    if verbose:
        logger.info("Enriching global occurrences with environmental data…")
    df_global = await _enrich_environmental(global_occ, batch_size=batch_size, verbose=verbose)

    if verbose:
        logger.info("Enriching local occurrences with environmental data…")
    df_local = await _enrich_environmental(local_occ, batch_size=batch_size, verbose=verbose)

    _write_csv(df_global, global_path, verbose)
    _write_csv(df_local, local_path, verbose)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ML-ready datasets without DB usage.")
    p.add_argument("--max-global", type=int, default=None, help="Max global records to fetch (None for all)")
    p.add_argument("--max-local", type=int, default=None, help="Max local (ZA) records to fetch (None for all)")
    p.add_argument("--batch-size", type=int, default=100, help="Batch size for environmental extraction")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.INFO)
    try:
        asyncio.run(run(args.max_global, args.max_local, args.batch_size, args.verbose))
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Cancelled by user")


if __name__ == "__main__":  # pragma: no cover
    main()
