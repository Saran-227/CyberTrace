"""Pipeline runner for harvesting, caching, normalizing, and validating ATM locations."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from src.config import (
    ATM_LOCATIONS_PROCESSED_PATH,
    ATM_LOCATIONS_SCHEMA,
    ATM_COVERAGE_REPORT_PATH,
    ATM_COVERAGE_SCHEMA,
    OSM_CACHE_DIR,
)
from src.geographic.study_regions import STUDY_REGIONS
from src.atm.osm_loader import fetch_osm_atms_with_status
from src.atm.validator import validate_atm_dataframe
from src.utils.logging import get_logger

logger = get_logger("ATMPipeline")

def build_atm_locations_dataset(
    regions: Optional[Dict[str, Dict[str, Any]]] = None,
    save_path: Optional[Path] = None,
    coverage_report_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Execute the end-to-end ATM dataset pipeline across study regions.

    Flow:
    1. Query/load cached OSM ATM elements for each region and record status.
    2. Normalize tags into standard CyberTrace schema.
    3. Generate and persist ATM coverage report (reports/atm_coverage_report.csv).
    4. Aggregate and deduplicate across study zones.
    5. Validate spatial bounds and schema.
    6. Save to data/processed/atm_locations.csv.
    """
    study_regions = regions or STUDY_REGIONS
    output_file = save_path or ATM_LOCATIONS_PROCESSED_PATH
    output_file.parent.mkdir(parents=True, exist_ok=True)

    report_file = coverage_report_path or ATM_COVERAGE_REPORT_PATH
    report_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ATM ingestion across {len(study_regions)} study regions...")
    all_records: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    region_counts: Dict[str, int] = {}

    for city_name, reg in study_regions.items():
        bbox = reg["bbox"]
        res = fetch_osm_atms_with_status(
            bbox=bbox,
            name=city_name,
            force_refresh=force_refresh,
            default_city=reg["city"],
            default_district=reg["district"],
            default_state=reg["state"],
        )

        records = res.get("records", [])
        raw_count = res.get("raw_count", len(records))
        status = res.get("status", "QUERY_FAILED")
        notes = res.get("notes", "")

        known_count = sum(1 for r in records if r["bank"] != "unknown" or r["operator"] != "unknown")
        unknown_count = sum(1 for r in records if r["bank"] == "unknown" and r["operator"] == "unknown")

        bbox_str = f"({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"
        coverage_rows.append({
            "city": reg["city"],
            "state": reg["state"],
            "bbox": bbox_str,
            "raw_osm_objects": raw_count,
            "atm_records": len(records),
            "known_bank_operator": known_count,
            "unknown_bank_operator": unknown_count,
            "status": status,
            "notes": notes,
        })

        region_counts[city_name] = len(records)
        all_records.extend(records)

    # Persist coverage report
    coverage_df = pd.DataFrame(coverage_rows, columns=ATM_COVERAGE_SCHEMA)
    coverage_df.to_csv(report_file, index=False)
    logger.info(f"Saved ATM coverage report ({len(coverage_df)} regions) to {report_file}")

    raw_count = len(all_records)
    logger.info(f"Harvested {raw_count} total ATM raw records across all regions.")

    if not all_records:
        logger.warning("No ATM records harvested. Creating empty DataFrame matching schema.")
        df = pd.DataFrame(columns=ATM_LOCATIONS_SCHEMA)
        df.to_csv(output_file, index=False)
        return {
            "total_records": 0,
            "raw_records": 0,
            "duplicates_removed": 0,
            "validation": {"is_valid": False, "errors": ["No records harvested"]},
            "output_path": str(output_file),
            "coverage_report_path": str(report_file),
            "coverage_df": coverage_df,
        }

    df = pd.DataFrame(all_records)

    # 1. Deduplicate by unique stable atm_id
    pre_dedupe = len(df)
    df = df.drop_duplicates(subset=["atm_id"]).copy()
    id_dupes_dropped = pre_dedupe - len(df)

    # 2. Deduplicate exact coordinate collisions with identical bank
    pre_coord = len(df)
    df["lat_round"] = df["latitude"].round(5)
    df["lon_round"] = df["longitude"].round(5)
    df = df.drop_duplicates(subset=["lat_round", "lon_round", "bank"]).copy()
    coord_dupes_dropped = pre_coord - len(df)
    df = df.drop(columns=["lat_round", "lon_round"])

    total_duplicates_dropped = id_dupes_dropped + coord_dupes_dropped
    logger.info(
        f"Deduplication removed {total_duplicates_dropped} duplicates "
        f"({id_dupes_dropped} ID, {coord_dupes_dropped} coordinate collisions)."
    )

    # Reorder columns strictly according to schema
    df = df[ATM_LOCATIONS_SCHEMA]

    # Validate final dataset
    val_results = validate_atm_dataframe(df)

    # Save to data/processed/atm_locations.csv
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} validated ATM records to {output_file}")

    cache_files = list(OSM_CACHE_DIR.glob("*.json"))

    summary = {
        "output_path": str(output_file),
        "coverage_report_path": str(report_file),
        "raw_records": raw_count,
        "total_records": len(df),
        "duplicates_removed": total_duplicates_dropped,
        "known_bank_count": val_results["known_bank_count"],
        "unknown_bank_count": val_results["unknown_bank_count"],
        "known_operator_count": val_results["known_operator_count"],
        "unknown_operator_count": val_results["unknown_operator_count"],
        "cities_count": val_results["cities_count"],
        "cities_represented": sorted(df["city"].unique().tolist()),
        "cache_files_count": len(cache_files),
        "cache_files": [f.name for f in cache_files],
        "validation_passed": val_results["is_valid"],
        "warnings": val_results["warnings"],
        "errors": val_results["errors"],
        "coverage_df": coverage_df,
    }
    return summary

if __name__ == "__main__":
    summary = build_atm_locations_dataset()
    print("=" * 60)
    print("CYBERTRACE PHASE 2A.1: ATM DATASET SUMMARY")
    print("=" * 60)
    print(f"Output File:           {summary['output_path']}")
    print(f"Coverage Report:       {summary['coverage_report_path']}")
    print(f"Total ATM Records:     {summary['total_records']}")
    print(f"Duplicates Removed:    {summary['duplicates_removed']}")
    print(f"Known Bank/Operator:   {summary['known_bank_count']}")
    print(f"Unknown Bank/Operator: {summary['unknown_bank_count']}")
    print(f"Cities Represented:    {summary['cities_count']}")
    print(f"Cache Files Created:   {summary['cache_files_count']}")
    print(f"Validation Status:     {'PASSED' if summary['validation_passed'] else 'FAILED'}")
    print("=" * 60)

