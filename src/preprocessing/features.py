"""Feature engineering utilities for CyberTrace."""

import pandas as pd
import numpy as np

def categorize_amount(amount: float) -> str:
    """Classify transaction amount into analytical risk tier."""
    if pd.isna(amount):
        return "Unknown"
    if amount < 5000:
        return "Low (<5k)"
    elif amount < 25000:
        return "Medium (5k-25k)"
    elif amount < 50000:
        return "High (25k-50k)"
    else:
        return "Critical (>50k)"

def engineer_complaint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive or normalize temporal and risk category features."""
    df_out = df.copy()

    # Re-verify amount_category
    if "amount" in df_out.columns:
        if "amount_category" not in df_out.columns or df_out["amount_category"].isnull().any():
            df_out["amount_category"] = df_out["amount"].apply(categorize_amount)

    # Derive temporal flags if complaint_date or hour is available
    if "hour" in df_out.columns:
        if "is_night" not in df_out.columns:
            df_out["is_night"] = df_out["hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

    if "day_of_week" in df_out.columns:
        if "is_weekend" not in df_out.columns:
            df_out["is_weekend"] = df_out["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

    return df_out
