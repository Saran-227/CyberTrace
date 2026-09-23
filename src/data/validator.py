"""Data validation module for complaint datasets."""

from typing import Dict, Any, List
import pandas as pd
from src.config import COMPLAINT_SCHEMA, TARGET_COLUMN
from src.utils.logging import get_logger

logger = get_logger("DataValidator")

def validate_complaints_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate complaint DataFrame for schema integrity, missingness, and duplicates."""
    results: Dict[str, Any] = {
        "is_valid": True,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_columns": [],
        "duplicate_rows_count": 0,
        "missing_values_by_column": {},
        "target_distribution": {},
        "warnings": [],
    }

    # Column completeness
    missing_cols = [c for c in COMPLAINT_SCHEMA if c not in df.columns]
    results["missing_columns"] = missing_cols
    if missing_cols:
        results["is_valid"] = False
        results["warnings"].append(f"Missing expected columns: {missing_cols}")

    # Duplicates check
    dupes = df.duplicated().sum()
    results["duplicate_rows_count"] = int(dupes)
    if dupes > 0:
        results["warnings"].append(f"Detected {dupes} duplicate rows.")

    # Missing values check
    nulls = df.isnull().sum().to_dict()
    results["missing_values_by_column"] = {k: int(v) for k, v in nulls.items() if v > 0}

    # Target variable check
    if TARGET_COLUMN in df.columns:
        results["target_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()
    else:
        results["warnings"].append(f"Target column '{TARGET_COLUMN}' is not present.")
        results["is_valid"] = False

    logger.info(
        f"Validation complete: {results['total_records']} rows, "
        f"{results['duplicate_rows_count']} duplicates, "
        f"{len(results['missing_values_by_column'])} columns with missing data."
    )
    return results
