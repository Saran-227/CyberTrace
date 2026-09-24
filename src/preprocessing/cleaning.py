"""Data cleaning utilities for CyberTrace cybercrime complaints.

Provides robust, logged routines for duplicate removal, missing-value imputation,
categorical normalization, and data validity checks without silently discarding rows.
"""

from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger("DataCleaning")


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """Identify and drop duplicate rows with structured logging."""
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=subset).copy()
    dropped = initial_count - len(df_cleaned)
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate rows (subset={subset}). Initial: {initial_count}, Remaining: {len(df_cleaned)}.")
    else:
        logger.info("Deduplication check: 0 duplicate rows found.")
    return df_cleaned, dropped


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using domain-safe rules.

    Preserves backward compatibility while guaranteeing clean categorical defaults.
    """
    df_out = df.copy()

    # Numeric imputation
    if "amount" in df_out.columns and df_out["amount"].isnull().any():
        median_amount = float(df_out["amount"].median())
        n_missing = int(df_out["amount"].isnull().sum())
        df_out["amount"] = df_out["amount"].fillna(median_amount)
        logger.info(f"Imputed {n_missing} missing 'amount' values with median ({median_amount:.2f}).")

    # Categorical imputation defaults
    cat_defaults: Dict[str, str] = {
        "bank": "Unknown Bank",
        "transaction_type": "Unknown Type",
        "fraud_type": "Other / Unspecified",
        "city": "Unknown City",
        "district": "Unknown District",
        "state": "Unknown State",
        "amount_category": "Medium (5k-25k)",
    }

    for col, default_val in cat_defaults.items():
        if col in df_out.columns and df_out[col].isnull().any():
            n_missing = int(df_out[col].isnull().sum())
            df_out[col] = df_out[col].fillna(default_val)
            logger.info(f"Imputed {n_missing} missing '{col}' values with default '{default_val}'.")

    return df_out


def normalize_categorical_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize categorical strings by stripping whitespace and replacing blanks with 'Unknown'."""
    df_out = df.copy()
    if columns is None:
        columns = [
            col for col in ["bank", "transaction_type", "fraud_type", "city", "state", "district", "amount_category"]
            if col in df_out.columns
        ]

    for col in columns:
        if col in df_out.columns and df_out[col].dtype == object:
            df_out[col] = df_out[col].astype(str).str.strip()
            df_out[col] = df_out[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    return df_out


def validate_cleaning_bounds(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify numerical and geographic validity of the complaint dataset."""
    issues: List[str] = []

    # Amount validation
    if "amount" in df.columns:
        neg_amounts = int((df["amount"] <= 0).sum())
        if neg_amounts > 0:
            issues.append(f"{neg_amounts} records have non-positive amounts.")

    # Geographic coordinates validation
    if "complaint_latitude" in df.columns and "complaint_longitude" in df.columns:
        inv_lat = int(((df["complaint_latitude"] < -90.0) | (df["complaint_latitude"] > 90.0)).sum())
        inv_lon = int(((df["complaint_longitude"] < -180.0) | (df["complaint_longitude"] > 180.0)).sum())
        if inv_lat > 0:
            issues.append(f"{inv_lat} records have latitude outside [-90, 90].")
        if inv_lon > 0:
            issues.append(f"{inv_lon} records have longitude outside [-180, 180].")

    # Hour validation
    if "hour" in df.columns:
        inv_hours = int(((df["hour"] < 0) | (df["hour"] > 23)).sum())
        if inv_hours > 0:
            issues.append(f"{inv_hours} records have hour outside [0, 23].")

    # Day of week validation
    if "day_of_week" in df.columns:
        inv_dow = int(((df["day_of_week"] < 0) | (df["day_of_week"] > 6)).sum())
        if inv_dow > 0:
            issues.append(f"{inv_dow} records have day_of_week outside [0, 6].")

    is_valid = len(issues) == 0
    if not is_valid:
        logger.warning(f"Data validation issues encountered: {issues}")
    else:
        logger.info("All numerical and geographic bounds verified successfully.")

    return {
        "is_valid": is_valid,
        "issues": issues,
        "record_count": len(df),
    }


def clean_complaints_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning pipeline: deduplication, normalization, and domain imputation."""
    df_no_dupes, _ = remove_duplicates(df)
    df_norm = normalize_categorical_values(df_no_dupes)
    df_clean = impute_missing_values(df_norm)
    validate_cleaning_bounds(df_clean)
    return df_clean
