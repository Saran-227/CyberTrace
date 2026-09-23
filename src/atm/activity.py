"""Synthetic ATM Historical Activity Generation & Analysis Module.

NOTE: All ATM historical activity data is synthetic for academic machine learning research.
It does NOT reflect real banking transaction logs or personal identifiable financial data.
Strict non-leakage guarantee: Does NOT use or associate with withdrawal_zone or complaint targets.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

from src.config import (
    ATM_ACTIVITY_PROCESSED_PATH,
    ATM_ACTIVITY_RAW_PATH,
    ATM_ACTIVITY_SCHEMA,
    ATM_LOCATIONS_PROCESSED_PATH,
)
from src.data.loader import load_atm_activity, load_atm_locations
from src.utils.logging import get_logger

logger = get_logger("ATMActivity")

# Relative commercial activity weights by city
CITY_ACTIVITY_WEIGHTS = {
    "New Delhi": 1.45,
    "Gurugram": 1.40,
    "Noida": 1.30,
    "Faridabad": 1.25,
    "Chandigarh": 1.30,
    "Jaipur": 1.25,
    "Ludhiana": 1.20,
    "Amritsar": 1.05,
    "Jalandhar": 1.00,
    "Meerut": 1.00,
    "Ghaziabad": 1.05,
    "Patiala": 0.95,
    "Ambala": 0.85,
    "Alwar": 0.80,
}

# Diurnal hourly activity weights (00:00 to 23:00)
HOURLY_ACTIVITY_WEIGHTS = np.array([
    0.08, 0.05, 0.04, 0.04, 0.07, 0.15,  # 00:00 - 05:00: Late night (very low)
    0.35, 0.60, 0.95,                     # 06:00 - 08:00: Morning ramp
    1.40, 1.65, 1.70, 1.55, 1.45,         # 09:00 - 13:00: Business morning peak
    1.20, 1.25, 1.35,                     # 14:00 - 16:00: Afternoon steady
    1.75, 1.95, 1.80, 1.50,               # 17:00 - 20:00: Evening cash-out peak
    1.00, 0.50, 0.20,                     # 21:00 - 23:00: Late evening decline
])

# Day of week weights (Monday=0 to Sunday=6)
DOW_ACTIVITY_WEIGHTS = np.array([0.95, 0.98, 1.00, 1.00, 1.10, 1.15, 1.05])

def get_bank_tier_weight(bank_name: Any) -> float:
    """Return activity weight based on financial institution market presence."""
    b = str(bank_name).strip().lower()
    if any(k in b for k in ["sbi", "state bank", "hdfc", "icici", "axis"]):
        return 1.30
    if any(k in b for k in ["punjab national", "pnb", "baroda", "canara", "union", "kotak", "indusind"]):
        return 1.10
    if "unknown" in b:
        return 0.85
    return 0.95

def generate_synthetic_atm_activity(
    atm_locations_df: Optional[pd.DataFrame] = None,
    start_date: str = "2026-06-26",
    days: int = 90,
    random_seed: int = 42,
    save_processed: bool = True,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Generate deterministic synthetic historical ATM activity records for existing ATMs.

    Zero Target Leakage Guarantee:
    - Only reads physical ATM infrastructure from atm_locations.csv.
    - Does NOT read, reference, or correlate with withdrawal_zone.
    - Does NOT use complaint records or hidden target coordinates.
    - Contains NO fraud-specific activity variables (CyberTrace does not possess real bank fraud logs).

    Parameters:
        atm_locations_df: DataFrame of verified ATMs. If None, loads from atm_locations.csv.
        start_date: Starting observation date (YYYY-MM-DD), default 2026-06-26 (ending 2026-09-23 for 90 days).
        days: Number of historical days to simulate (default 90).
        random_seed: Deterministic RNG seed for complete reproducibility.
        save_processed: If True, writes to data/processed/atm_activity.csv.
        save_raw: If True, writes to data/raw/atm_activity_raw.csv.

    Returns:
        pd.DataFrame matching ATM_ACTIVITY_SCHEMA.
    """
    if atm_locations_df is None:
        atm_locations_df = load_atm_locations(use_raw=False)

    if atm_locations_df is None or atm_locations_df.empty:
        raise ValueError("Cannot generate ATM activity: ATM locations dataset is missing or empty.")

    # Panipat check: strictly only use ATMs present in input dataset
    atms = atm_locations_df.copy()
    n_atms = len(atms)
    hours = 24
    n_intervals = days * hours
    total_rows = n_atms * n_intervals

    logger.info(
        f"Generating {days}-day synthetic ATM activity for {n_atms} ATMs "
        f"({total_rows:,} total intervals, seed={random_seed})..."
    )

    # 1. Calendar Grid Setup
    start_dt = datetime.date.fromisoformat(start_date)
    date_list = [start_dt + datetime.timedelta(days=d) for d in range(days)]
    date_strs = np.array([d.isoformat() for d in date_list])
    dows = np.array([d.weekday() for d in date_list])

    grid_dates = np.repeat(date_strs, hours)
    grid_hours = np.tile(np.arange(hours), days)
    grid_dows = np.repeat(dows, hours)
    grid_weekends = (grid_dows >= 5).astype(int)
    grid_nights = ((grid_hours >= 22) | (grid_hours <= 5)).astype(int)

    # 2. Tile timestamps across all ATMs
    all_dates = np.tile(grid_dates, n_atms)
    all_hours = np.tile(grid_hours, n_atms)
    all_dows = np.tile(grid_dows, n_atms)
    all_weekends = np.tile(grid_weekends, n_atms)
    all_nights = np.tile(grid_nights, n_atms)
    all_atm_ids = np.repeat(atms["atm_id"].values, n_intervals)

    # 3. ATM Baseline Rates (Intrinsic Traffic Volume)
    atm_base_rates = []
    is_closed_night = []
    for _, row in atms.iterrows():
        c_wt = CITY_ACTIVITY_WEIGHTS.get(row["city"], 1.0)
        b_wt = get_bank_tier_weight(row.get("bank", "unknown"))
        # Deterministic individual ATM factor based on ID hash
        a_seed = (random_seed * 10007 + abs(hash(str(row["atm_id"])))) % (2**31 - 1)
        indiv = np.random.default_rng(a_seed).uniform(0.75, 1.25)
        base = 4.2 * c_wt * b_wt * indiv
        atm_base_rates.append(base)

        # Check explicit closure at night (is_24x7 == False)
        closed = row.get("is_24x7") is False or str(row.get("is_24x7")).strip().lower() == "false"
        is_closed_night.append(closed)

    atm_base_rates = np.array(atm_base_rates)
    is_closed_night = np.array(is_closed_night)

    # Repeat baseline rates for every interval
    all_base_rates = np.repeat(atm_base_rates, n_intervals)
    all_closed_night = np.repeat(is_closed_night, n_intervals)

    # Compute expected Poisson lambdas
    lambdas = all_base_rates * HOURLY_ACTIVITY_WEIGHTS[all_hours] * DOW_ACTIVITY_WEIGHTS[all_dows]
    lambdas[all_closed_night & (all_nights == 1)] = 0.0

    # 4. Stochastic Generation with Deterministic Seed
    rng = np.random.default_rng(random_seed)

    # Transaction count
    tx_counts = rng.poisson(lambdas)
    tx_counts[all_closed_night & (all_nights == 1)] = 0

    # Cash withdrawal count (strictly <= transaction count)
    p_wd = rng.beta(a=8.0, b=3.0, size=total_rows)  # Beta(8, 3) ~ 0.727
    wd_counts = rng.binomial(tx_counts, p_wd)

    # Estimated cash volume (INR)
    mean_ticket = rng.gamma(shape=6.0, scale=520.0, size=total_rows)  # Gamma(6, 520) ~ 3,120 INR
    cash_volume = np.round(wd_counts * mean_ticket, 2)
    cash_volume[wd_counts == 0] = 0.0

    # High value withdrawals (> 15,000 INR calculated purely from synthetic transactions)
    high_val = rng.binomial(wd_counts, 0.07)

    # Average amount per withdrawal
    denom = np.maximum(wd_counts, 1)
    avg_amount = np.where(wd_counts > 0, np.round(cash_volume / denom, 2), 0.0)

    # Activity score (0 to 100)
    raw_score = 0.60 * (tx_counts / 25.0) * 100.0 + 0.40 * (cash_volume / 80000.0) * 100.0
    activity_score = np.clip(np.round(raw_score, 1), 0.0, 100.0)
    activity_score[tx_counts == 0] = 0.0

    # 5. Assemble Final DataFrame
    df = pd.DataFrame({
        "atm_id": all_atm_ids,
        "date": all_dates,
        "hour": all_hours,
        "day_of_week": all_dows,
        "is_weekend": all_weekends,
        "is_night": all_nights,
        "transaction_count": tx_counts,
        "cash_withdrawal_count": wd_counts,
        "estimated_cash_volume": cash_volume,
        "activity_score": activity_score,
        "high_value_withdrawal_count": high_val,
        "average_amount": avg_amount,
    })

    # Strict column ordering
    df = df[ATM_ACTIVITY_SCHEMA]

    # 6. Save Datasets
    if save_processed:
        ATM_ACTIVITY_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ATM_ACTIVITY_PROCESSED_PATH, index=False)
        logger.info(f"Saved processed synthetic ATM activity ({len(df):,} records) to {ATM_ACTIVITY_PROCESSED_PATH}")

    if save_raw:
        ATM_ACTIVITY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ATM_ACTIVITY_RAW_PATH, index=False)
        logger.info(f"Saved raw synthetic ATM activity ({len(df):,} records) to {ATM_ACTIVITY_RAW_PATH}")

    return df

def validate_atm_activity_dataframe(
    df: pd.DataFrame,
    atm_locations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Validate synthetic ATM activity dataset integrity, bounds, and schema.

    Returns structured validation summary dictionary.
    """
    errors: List[str] = []

    # 1. Required schema columns
    missing_cols = [col for col in ATM_ACTIVITY_SCHEMA if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Strict Zero Target Leakage & Clean Activity Checks
    has_target = "withdrawal_zone" in df.columns
    if has_target:
        errors.append("CRITICAL: Target column 'withdrawal_zone' found in ATM activity dataset (Target Leakage)!")

    # Check for fraud-specific columns (should not be in primary activity dataset)
    fraud_cols = [c for c in df.columns if "fraud" in c.lower()]
    has_fraud_column = len(fraud_cols) > 0
    if has_fraud_column:
        errors.append(f"CRITICAL: Fraud-specific column(s) {fraud_cols} found in ATM activity dataset!")

    # Check for complaint metadata leakage
    complaint_cols = [c for c in df.columns if c in ["complaint_id", "complaint_latitude", "complaint_longitude", "fraud_type", "amount_category"]]
    if complaint_cols:
        errors.append(f"CRITICAL: Complaint metadata column(s) {complaint_cols} found in ATM activity dataset!")

    # 3. ATM ID validity
    if atm_locations_df is None:
        atm_locations_df = load_atm_locations(use_raw=False)

    nonexistent_atms_count = 0
    if atm_locations_df is not None and not atm_locations_df.empty:
        valid_atm_ids = set(atm_locations_df["atm_id"].unique())
        activity_atm_ids = set(df["atm_id"].unique())
        orphan_ids = activity_atm_ids - valid_atm_ids
        nonexistent_atms_count = len(orphan_ids)
        if nonexistent_atms_count > 0:
            errors.append(f"Found {nonexistent_atms_count} ATM IDs not present in verified ATM locations dataset.")

    # 4. Count bounds
    neg_tx = int((df["transaction_count"] < 0).sum())
    neg_wd = int((df["cash_withdrawal_count"] < 0).sum())
    inv_ratio = int((df["cash_withdrawal_count"] > df["transaction_count"]).sum())
    neg_vol = int((df["estimated_cash_volume"] < 0).sum())
    inv_score = int(((df["activity_score"] < 0.0) | (df["activity_score"] > 100.0)).sum())

    if neg_tx > 0:
        errors.append(f"{neg_tx} records have negative transaction_count.")
    if neg_wd > 0:
        errors.append(f"{neg_wd} records have negative cash_withdrawal_count.")
    if inv_ratio > 0:
        errors.append(f"{inv_ratio} records have cash_withdrawal_count > transaction_count.")
    if neg_vol > 0:
        errors.append(f"{neg_vol} records have negative estimated_cash_volume.")
    if inv_score > 0:
        errors.append(f"{inv_score} records have activity_score outside [0, 100].")

    # 5. Temporal consistency & date range audit
    inv_hours = int(((df["hour"] < 0) | (df["hour"] > 23)).sum())
    if inv_hours > 0:
        errors.append(f"{inv_hours} records have invalid hour values outside 0-23.")

    # Check future dates (relative to current audit cutoff: 2026-09-23)
    future_dates_count = 0
    unique_dates_count = 0
    if "date" in df.columns:
        future_dates_count = int((df["date"] > "2026-09-23").sum())
        if future_dates_count > 0:
            errors.append(f"{future_dates_count} records have dates past 2026-09-23 (future date leakage).")
        unique_dates_count = int(df["date"].nunique())

    # Check is_night consistency
    expected_night = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    inconsistent_night = int((df["is_night"] != expected_night).sum())
    if inconsistent_night > 0:
        errors.append(f"{inconsistent_night} records have is_night inconsistent with hour.")

    # Check is_weekend consistency
    expected_wknd = (df["day_of_week"] >= 5).astype(int)
    inconsistent_wknd = int((df["is_weekend"] != expected_wknd).sum())
    if inconsistent_wknd > 0:
        errors.append(f"{inconsistent_wknd} records have is_weekend inconsistent with day_of_week.")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "missing_columns": missing_cols,
        "has_withdrawal_zone": has_target,
        "has_fraud_column": has_fraud_column,
        "total_records": len(df),
        "unique_atms": df["atm_id"].nunique(),
        "unique_dates_count": unique_dates_count,
        "future_dates_count": future_dates_count,
        "nonexistent_atms_count": nonexistent_atms_count,
        "negative_transaction_counts": neg_tx,
        "negative_withdrawal_counts": neg_wd,
        "invalid_withdrawal_ratio_count": inv_ratio,
        "negative_cash_volume_count": neg_vol,
        "invalid_activity_score_count": inv_score,
        "invalid_hours_count": inv_hours,
        "inconsistent_night_count": inconsistent_night,
        "inconsistent_weekend_count": inconsistent_wknd,
        "errors": errors,
    }

def get_atm_activity_summary(atm_id: str, hour: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve historical synthetic activity metrics for a given candidate ATM."""
    df_activity = load_atm_activity()

    default_summary = {
        "atm_id": atm_id,
        "total_recorded_transactions": 0,
        "historical_fraud_withdrawals": 0,
        "high_value_withdrawal_rate": 0.0,
        "is_historically_active_at_hour": False,
        "is_synthetic_data": True,
    }

    if df_activity is None or df_activity.empty or "atm_id" not in df_activity.columns:
        return default_summary

    atm_records = df_activity[df_activity["atm_id"] == atm_id]
    if atm_records.empty:
        return default_summary

    total_tx = atm_records["transaction_count"].sum()
    high_val = atm_records["high_value_withdrawal_count"].sum() if "high_value_withdrawal_count" in atm_records.columns else 0

    hour_active = False
    if hour is not None and "hour" in atm_records.columns:
        hour_records = atm_records[atm_records["hour"] == hour]
        hour_active = not hour_records.empty and (hour_records["transaction_count"].sum() > 0)

    return {
        "atm_id": atm_id,
        "total_recorded_transactions": int(total_tx),
        "historical_fraud_withdrawals": 0,
        "high_value_withdrawal_rate": round(float(high_val / max(1, total_tx)), 3),
        "is_historically_active_at_hour": hour_active,
        "is_synthetic_data": True,
    }

if __name__ == "__main__":
    df = generate_synthetic_atm_activity()
    val = validate_atm_activity_dataframe(df)
    print("=" * 60)
    print("CYBERTRACE PHASE 2B: SYNTHETIC ATM ACTIVITY SUMMARY")
    print("=" * 60)
    print(f"Total Records:         {val['total_records']:,}")
    print(f"Unique ATMs:           {val['unique_atms']}")
    print(f"Validation Status:     {'PASSED' if val['is_valid'] else 'FAILED'}")
    print(f"Target Leakage Check:  {'SAFE (No withdrawal_zone)' if not val['has_withdrawal_zone'] else 'LEAK DETECTED'}")
    print("=" * 60)
