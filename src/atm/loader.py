"""ATM Data and Activity Loader for CyberTrace Phase 7.

Provides validated, cached loading of verified OpenStreetMap ATM locations and
synthetic historical ATM operational activity with strict zero-leakage enforcement.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ATM_LOCATIONS_PROCESSED_PATH,
    ATM_ACTIVITY_PROCESSED_PATH,
    ATM_LOCATIONS_SCHEMA,
    ATM_ACTIVITY_SCHEMA,
    REPORTS_DIR,
)
from src.geographic.distance import haversine_distance
from src.utils.logging import get_logger

logger = get_logger("ATMLoader")

# Module-level memory caches
_CACHED_ATM_LOCATIONS: Optional[pd.DataFrame] = None
_CACHED_ATM_ACTIVITY: Optional[pd.DataFrame] = None
_CACHED_ZONE_CENTROIDS: Optional[pd.DataFrame] = None
_CACHED_ACTIVITY_PROFILES: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None


def load_zone_centroids(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load verified withdrawal zone centroids from reports or fallback to default coordinates."""
    global _CACHED_ZONE_CENTROIDS
    if _CACHED_ZONE_CENTROIDS is not None:
        return _CACHED_ZONE_CENTROIDS.copy()

    centroids_path = filepath or (REPORTS_DIR / "zone_centroids.csv")
    if centroids_path.exists():
        df = pd.read_csv(centroids_path)
    else:
        # Default empirical centroids established in Phase 3
        df = pd.DataFrame([
            {"withdrawal_zone": "Zone_01", "centroid_latitude": 31.634329, "centroid_longitude": 74.870129},
            {"withdrawal_zone": "Zone_02", "centroid_latitude": 31.324763, "centroid_longitude": 75.577009},
            {"withdrawal_zone": "Zone_03", "centroid_latitude": 30.900601, "centroid_longitude": 75.858160},
            {"withdrawal_zone": "Zone_04", "centroid_latitude": 30.477590, "centroid_longitude": 76.659670},
            {"withdrawal_zone": "Zone_05", "centroid_latitude": 29.391692, "centroid_longitude": 76.965274},
            {"withdrawal_zone": "Zone_06", "centroid_latitude": 28.965979, "centroid_longitude": 77.690434},
            {"withdrawal_zone": "Zone_07", "centroid_latitude": 28.554406, "centroid_longitude": 77.349756},
            {"withdrawal_zone": "Zone_08", "centroid_latitude": 28.532572, "centroid_longitude": 77.135619},
            {"withdrawal_zone": "Zone_09", "centroid_latitude": 27.554209, "centroid_longitude": 76.636352},
            {"withdrawal_zone": "Zone_10", "centroid_latitude": 26.912877, "centroid_longitude": 75.788582},
        ])

    _CACHED_ZONE_CENTROIDS = df
    return df.copy()


def get_atm_zone_assignment(lat: float, lon: float, centroids_df: Optional[pd.DataFrame] = None) -> Tuple[str, float, str, float]:
    """Determine the primary and secondary nearest zone and distances (km) for given coordinates."""
    if centroids_df is None:
        centroids_df = load_zone_centroids()

    distances: List[Tuple[str, float]] = []
    for _, row in centroids_df.iterrows():
        zone = str(row["withdrawal_zone"])
        c_lat = float(row["centroid_latitude"])
        c_lon = float(row["centroid_longitude"])
        dist = haversine_distance(lat, lon, c_lat, c_lon)
        distances.append((zone, dist))

    distances.sort(key=lambda x: x[1])
    primary_zone, primary_dist = distances[0]
    secondary_zone, secondary_dist = distances[1]
    return primary_zone, round(primary_dist, 2), secondary_zone, round(secondary_dist, 2)


def load_atm_locations(
    filepath: Optional[Path] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load, validate, normalize, and spatially index verified OpenStreetMap ATM locations.

    Requirements:
    - Validate required schema columns
    - Validate valid latitude and longitude ranges
    - Validate unique atm_id values
    - Normalize missing or blank bank/operator values to 'unknown'
    - Never infer bank from incomplete or ambiguous text
    - Preserve OpenStreetMap attribution and source info
    - Spatially enrich with primary zone, secondary zone, and centroid distances
    """
    global _CACHED_ATM_LOCATIONS
    if _CACHED_ATM_LOCATIONS is not None and not force_reload:
        return _CACHED_ATM_LOCATIONS.copy()

    atm_path = filepath or ATM_LOCATIONS_PROCESSED_PATH
    if not atm_path.exists():
        raise FileNotFoundError(f"ATM locations dataset not found at: {atm_path}")

    df = pd.read_csv(atm_path)

    # 1. Validate required columns
    missing_cols = [c for c in ATM_LOCATIONS_SCHEMA if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ATM locations dataset missing required columns: {missing_cols}")

    # 2. Validate latitude & longitude bounds
    invalid_coords = (
        (df["latitude"] < -90.0) | (df["latitude"] > 90.0) |
        (df["longitude"] < -180.0) | (df["longitude"] > 180.0) |
        df["latitude"].isna() | df["longitude"].isna()
    )
    if invalid_coords.any():
        raise ValueError(f"Found {invalid_coords.sum()} invalid coordinates in ATM locations dataset.")

    # 3. Validate unique atm_id
    if df["atm_id"].duplicated().any():
        dup_count = int(df["atm_id"].duplicated().sum())
        raise ValueError(f"Duplicate atm_id found: {dup_count} duplicate records detected.")

    # 4. Normalize bank and operator
    df["bank"] = df["bank"].fillna("unknown").astype(str).str.strip()
    df.loc[df["bank"].str.lower().isin(["", "none", "nan", "null"]), "bank"] = "unknown"

    df["operator"] = df["operator"].fillna("unknown").astype(str).str.strip()
    df.loc[df["operator"].str.lower().isin(["", "none", "nan", "null"]), "operator"] = "unknown"

    # Normalize is_24x7 string/boolean
    def _parse_24x7(val: Any) -> Optional[bool]:
        if pd.isna(val):
            return None
        val_str = str(val).strip().lower()
        if val_str in ["true", "1", "yes"]:
            return True
        elif val_str in ["false", "0", "no"]:
            return False
        return None  # unknown

    df["is_24x7_parsed"] = df["is_24x7"].apply(_parse_24x7)

    # 5. Enrich with zone assignment based on distance to empirical centroids
    centroids_df = load_zone_centroids()
    zone_assignments = [
        get_atm_zone_assignment(lat, lon, centroids_df)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    df["assigned_zone"] = [z[0] for z in zone_assignments]
    df["zone_distance_km"] = [z[1] for z in zone_assignments]
    df["secondary_zone"] = [z[2] for z in zone_assignments]
    df["secondary_distance_km"] = [z[3] for z in zone_assignments]

    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "OpenStreetMap"

    _CACHED_ATM_LOCATIONS = df.copy()
    logger.info(f"Loaded and verified {len(df)} ATM locations across {df['city'].nunique()} cities.")
    return df


def load_atm_activity(
    filepath: Optional[Path] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load and validate synthetic historical ATM operational activity dataset.

    Enforces strict zero-leakage constraints:
    - No fraud-specific variables (e.g. fraud_withdrawals)
    - No withdrawal_zone
    - No complaint metadata
    - Value bounds validation (activity_score in [0, 100], hours in [0, 23], counts >= 0)
    """
    global _CACHED_ATM_ACTIVITY
    if _CACHED_ATM_ACTIVITY is not None and not force_reload:
        return _CACHED_ATM_ACTIVITY.copy()

    activity_path = filepath or ATM_ACTIVITY_PROCESSED_PATH
    if not activity_path.exists():
        raise FileNotFoundError(f"ATM activity dataset not found at: {activity_path}")

    df = pd.read_csv(activity_path)

    # 1. Validate required columns
    missing_cols = [c for c in ATM_ACTIVITY_SCHEMA if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ATM activity dataset missing required columns: {missing_cols}")

    # 2. Strict Zero Target Leakage & Forbidden Column Checks
    forbidden_terms = ["fraud", "withdrawal_zone", "complaint", "victim", "synthetic_cashout"]
    for col in df.columns:
        for term in forbidden_terms:
            if term in col.lower():
                raise ValueError(f"Forbidden column detected in ATM activity dataset: '{col}' (violates zero leakage).")

    # 3. Value bounds validations
    if (df["hour"] < 0).any() or (df["hour"] > 23).any():
        raise ValueError("Invalid hour values detected in ATM activity (must be 0-23).")

    if (df["activity_score"] < 0.0).any() or (df["activity_score"] > 100.0).any():
        raise ValueError("Invalid activity_score detected in ATM activity (must be 0-100).")

    if (df["transaction_count"] < 0).any() or (df["cash_withdrawal_count"] < 0).any():
        raise ValueError("Negative transaction counts detected in ATM activity.")

    if (df["average_amount"] < 0).any() or (df["estimated_cash_volume"] < 0).any():
        raise ValueError("Negative amount values detected in ATM activity.")

    _CACHED_ATM_ACTIVITY = df.copy()
    logger.info(f"Loaded and verified {len(df):,} synthetic ATM activity records.")
    return df


def _build_activity_profiles(activity_df: pd.DataFrame) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Pre-aggregate mean operational activity profile by ATM and hour for instantaneous lookup."""
    logger.info("Building pre-aggregated ATM hourly activity profiles...")
    grouped = activity_df.groupby(["atm_id", "hour"]).agg({
        "activity_score": "mean",
        "transaction_count": "mean",
        "cash_withdrawal_count": "mean",
        "estimated_cash_volume": "mean",
        "high_value_withdrawal_count": "mean",
        "average_amount": "mean",
    }).reset_index()

    profiles: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _, row in grouped.iterrows():
        atm_id = str(row["atm_id"])
        hour = int(row["hour"])
        if atm_id not in profiles:
            profiles[atm_id] = {}
        profiles[atm_id][hour] = {
            "activity_score": round(float(row["activity_score"]), 2),
            "transaction_count": round(float(row["transaction_count"]), 2),
            "cash_withdrawal_count": round(float(row["cash_withdrawal_count"]), 2),
            "estimated_cash_volume": round(float(row["estimated_cash_volume"]), 2),
            "high_value_withdrawal_count": round(float(row["high_value_withdrawal_count"]), 2),
            "average_amount": round(float(row["average_amount"]), 2),
        }

    logger.info(f"Cached hourly activity profiles for {len(profiles)} ATMs.")
    return profiles


def get_atm_activity_profile(
    atm_id: str,
    hour: Optional[int] = None,
    window_hours: int = 2,
) -> Dict[str, float]:
    """Retrieve fast, averaged simulated operational activity metrics for an ATM around a given hour.

    Parameters:
        atm_id: Verified ATM identifier (e.g. 'OSM-NODE-1845296821')
        hour: Incident hour (0-23). If None, returns 24-hour daily average.
        window_hours: Temporal window (+/- hours) around incident hour.

    Returns:
        Dictionary of averaged operational metrics (activity_score, average_amount, etc.)
    """
    global _CACHED_ACTIVITY_PROFILES
    if _CACHED_ACTIVITY_PROFILES is None:
        act_df = load_atm_activity()
        _CACHED_ACTIVITY_PROFILES = _build_activity_profiles(act_df)

    atm_profile = _CACHED_ACTIVITY_PROFILES.get(atm_id)
    if not atm_profile:
        # Graceful fallback for ATM with no activity logs
        return {
            "activity_score": 25.0,
            "transaction_count": 2.0,
            "cash_withdrawal_count": 1.0,
            "estimated_cash_volume": 5000.0,
            "high_value_withdrawal_count": 0.0,
            "average_amount": 3500.0,
            "has_activity_data": False,
        }

    if hour is None:
        # Average across all 24 hours
        hours_to_average = list(atm_profile.keys())
    else:
        # Wrap circular 24-hour window
        hours_to_average = [
            (hour + offset) % 24
            for offset in range(-window_hours, window_hours + 1)
        ]

    matched_records = [atm_profile[h] for h in hours_to_average if h in atm_profile]
    if not matched_records:
        matched_records = list(atm_profile.values())

    return {
        "activity_score": round(float(np.mean([r["activity_score"] for r in matched_records])), 2),
        "transaction_count": round(float(np.mean([r["transaction_count"] for r in matched_records])), 2),
        "cash_withdrawal_count": round(float(np.mean([r["cash_withdrawal_count"] for r in matched_records])), 2),
        "estimated_cash_volume": round(float(np.mean([r["estimated_cash_volume"] for r in matched_records])), 2),
        "high_value_withdrawal_count": round(float(np.mean([r["high_value_withdrawal_count"] for r in matched_records])), 2),
        "average_amount": round(float(np.mean([r["average_amount"] for r in matched_records])), 2),
        "has_activity_data": True,
    }
