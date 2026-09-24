"""Feature engineering utilities for CyberTrace cybercrime complaints.

Provides transformations for cyclical temporal coordinates (hour/day_of_week sin/cos),
log-transformed monetary amounts, and financial risk categories.
"""

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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclical temporal features and log-transformed monetary amounts.

    Engineered Features:
    - amount_log: log1p(amount) to stabilize right-skewed financial distribution.
    - hour_sin, hour_cos: 24-hour circular trigonometric transformation.
    - day_of_week_sin, day_of_week_cos: 7-day circular trigonometric transformation.
    """
    df_out = df.copy()

    # 1. Log-transformed amount
    if "amount" in df_out.columns:
        safe_amount = np.maximum(0.0, pd.to_numeric(df_out["amount"], errors="coerce").fillna(0.0))
        df_out["amount_log"] = np.round(np.log1p(safe_amount), 4)

    # 2. Cyclical hour features
    if "hour" in df_out.columns:
        safe_hour = pd.to_numeric(df_out["hour"], errors="coerce").fillna(0.0)
        df_out["hour_sin"] = np.round(np.sin(2.0 * np.pi * safe_hour / 24.0), 4)
        df_out["hour_cos"] = np.round(np.cos(2.0 * np.pi * safe_hour / 24.0), 4)

    # 3. Cyclical day-of-week features
    if "day_of_week" in df_out.columns:
        safe_dow = pd.to_numeric(df_out["day_of_week"], errors="coerce").fillna(0.0)
        df_out["day_of_week_sin"] = np.round(np.sin(2.0 * np.pi * safe_dow / 7.0), 4)
        df_out["day_of_week_cos"] = np.round(np.cos(2.0 * np.pi * safe_dow / 7.0), 4)

    return df_out


def engineer_complaint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline including cyclical time, log amounts, and risk tiers."""
    df_out = df.copy()

    # Re-verify amount_category
    if "amount" in df_out.columns:
        if "amount_category" not in df_out.columns or df_out["amount_category"].isnull().any():
            df_out["amount_category"] = df_out["amount"].apply(categorize_amount)

    # Derive temporal flags if hour/day_of_week available
    if "hour" in df_out.columns:
        if "is_night" not in df_out.columns or df_out["is_night"].isnull().any():
            df_out["is_night"] = df_out["hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

    if "day_of_week" in df_out.columns:
        if "is_weekend" not in df_out.columns or df_out["is_weekend"].isnull().any():
            df_out["is_weekend"] = df_out["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

    # Add cyclical and log features
    df_out = add_engineered_features(df_out)

    return df_out
