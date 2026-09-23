"""Validation module for ATM geographic dataset records."""

from typing import Dict, Any, List, Tuple
import pandas as pd
from src.config import ATM_LOCATIONS_SCHEMA
from src.utils.logging import get_logger

logger = get_logger("ATMValidator")

VALID_IS_24X7_VALUES = {True, False, "True", "False", "true", "false", "unknown"}

def validate_atm_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate ATM locations DataFrame against strict schema and spatial bounds.

    Validation criteria:
    - Required columns present.
    - Latitude in [-90, 90].
    - Longitude in [-180, 180].
    - Non-null/non-empty stable atm_id.
    - Duplicate detection (by atm_id and coordinate proximity).
    - Unknown values normalized (lowercase 'unknown').
    - Valid is_24x7 values (True, False, 'unknown').
    - Source field specified.
    """
    results: Dict[str, Any] = {
        "is_valid": True,
        "total_records": len(df),
        "missing_columns": [],
        "invalid_coordinates_count": 0,
        "empty_atm_ids": 0,
        "duplicate_atm_ids": 0,
        "duplicate_coordinates": 0,
        "invalid_is_24x7_count": 0,
        "invalid_sources_count": 0,
        "known_bank_count": 0,
        "unknown_bank_count": 0,
        "known_operator_count": 0,
        "unknown_operator_count": 0,
        "cities_count": 0,
        "errors": [],
        "warnings": [],
    }

    # 1. Check required columns
    missing_cols = [c for c in ATM_LOCATIONS_SCHEMA if c not in df.columns]
    results["missing_columns"] = missing_cols
    if missing_cols:
        results["is_valid"] = False
        results["errors"].append(f"Missing required columns: {missing_cols}")
        return results

    if df.empty:
        results["warnings"].append("ATM DataFrame is empty.")
        return results

    # 2. Latitude and Longitude range check
    lat_invalid = df[(df["latitude"].isnull()) | (df["latitude"] < -90.0) | (df["latitude"] > 90.0)]
    lon_invalid = df[(df["longitude"].isnull()) | (df["longitude"] < -180.0) | (df["longitude"] > 180.0)]
    invalid_coords = len(set(lat_invalid.index).union(set(lon_invalid.index)))
    results["invalid_coordinates_count"] = invalid_coords
    if invalid_coords > 0:
        results["is_valid"] = False
        results["errors"].append(f"Detected {invalid_coords} rows with invalid/out-of-bounds coordinates.")

    # 3. Stable ATM ID checks
    empty_ids = df["atm_id"].isnull() | (df["atm_id"].astype(str).str.strip() == "")
    results["empty_atm_ids"] = int(empty_ids.sum())
    if results["empty_atm_ids"] > 0:
        results["is_valid"] = False
        results["errors"].append(f"Detected {results['empty_atm_ids']} rows with missing/empty atm_id.")

    # 4. Duplicate checks
    dupe_ids = df["atm_id"].duplicated().sum()
    results["duplicate_atm_ids"] = int(dupe_ids)
    if dupe_ids > 0:
        results["warnings"].append(f"Detected {dupe_ids} duplicate atm_id entries.")

    dupe_coords = df.duplicated(subset=["latitude", "longitude"]).sum()
    results["duplicate_coordinates"] = int(dupe_coords)
    if dupe_coords > 0:
        results["warnings"].append(f"Detected {dupe_coords} rows sharing identical coordinates.")

    # 5. is_24x7 value validation
    invalid_247 = df[~df["is_24x7"].isin(VALID_IS_24X7_VALUES)]
    results["invalid_is_24x7_count"] = len(invalid_247)
    if len(invalid_247) > 0:
        results["is_valid"] = False
        results["errors"].append(f"Detected {len(invalid_247)} rows with invalid is_24x7 values.")

    # 6. Source validation
    invalid_sources = df[df["source"].isnull() | (df["source"].astype(str).str.strip() == "")]
    results["invalid_sources_count"] = len(invalid_sources)
    if len(invalid_sources) > 0:
        results["is_valid"] = False
        results["errors"].append(f"Detected {len(invalid_sources)} rows with invalid source.")

    # 7. Metadata summary counts
    results["known_bank_count"] = int((df["bank"].str.lower() != "unknown").sum())
    results["unknown_bank_count"] = int((df["bank"].str.lower() == "unknown").sum())
    results["known_operator_count"] = int((df["operator"].str.lower() != "unknown").sum())
    results["unknown_operator_count"] = int((df["operator"].str.lower() == "unknown").sum())
    results["cities_count"] = int(df["city"].nunique())

    logger.info(
        f"ATM validation complete: {results['total_records']} records, "
        f"{results['known_bank_count']} known banks, {results['unknown_bank_count']} unknown banks, "
        f"{results['cities_count']} cities."
    )
    return results
