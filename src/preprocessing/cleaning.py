"""Data cleaning utilities for cybercrime complaints."""

from typing import Tuple
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("DataCleaning")

def remove_duplicates(df: pd.DataFrame, subset: list = None) -> Tuple[pd.DataFrame, int]:
    """Identify and drop duplicate rows."""
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=subset).copy()
    dropped = initial_count - len(df_cleaned)
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate rows.")
    return df_cleaned, dropped

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using domain-safe rules."""
    df_out = df.copy()

    # Numeric imputation
    if "amount" in df_out.columns and df_out["amount"].isnull().any():
        median_amount = df_out["amount"].median()
        df_out["amount"] = df_out["amount"].fillna(median_amount)

    # Categorical imputation
    cat_defaults = {
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
            df_out[col] = df_out[col].fillna(default_val)

    return df_out

def clean_complaints_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning pipeline: deduplication and imputation."""
    df_no_dupes, _ = remove_duplicates(df)
    df_clean = impute_missing_values(df_no_dupes)
    return df_clean
