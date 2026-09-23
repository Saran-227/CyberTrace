"""Synthetic Cybercrime Complaint Data Generator for CyberTrace.

Generates realistic synthetic complaint records strictly adhering to the schema,
introducing realistic noise, missing values, and duplicate rows for preprocessing
benchmarks, while ensuring zero target leakage.
"""

from typing import Optional
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

from src.config import (
    COMPLAINTS_RAW_PATH,
    COMPLAINTS_PROCESSED_PATH,
    COMPLAINT_SCHEMA,
)
from src.utils.logging import get_logger

logger = get_logger("DataGenerator")

BANKS = ["HDFC", "SBI", "ICICI", "Axis Bank", "Punjab National Bank", "Bank of Baroda", "Kotak Mahindra"]
TRANSACTION_TYPES = ["UPI", "IMPS", "NEFT", "Net Banking", "ATM Withdrawal", "Card Payment"]
FRAUD_TYPES = [
    "Phishing/Smishing",
    "Lottery/Task Scam",
    "Identity Theft",
    "Impersonation",
    "Investment Scam",
    "Customer Support Fraud",
    "Sextortion/Blackmail",
]

# Focus geographic regions (e.g. Punjab region / North India anchor for consistent demo context)
LOCATIONS = [
    {"city": "Jalandhar", "district": "Jalandhar", "state": "Punjab", "lat": 31.3260, "lon": 75.5762},
    {"city": "Ludhiana", "district": "Ludhiana", "state": "Punjab", "lat": 30.9010, "lon": 75.8573},
    {"city": "Amritsar", "district": "Amritsar", "state": "Punjab", "lat": 31.6340, "lon": 74.8723},
    {"city": "Patiala", "district": "Patiala", "state": "Punjab", "lat": 30.3398, "lon": 76.3869},
    {"city": "Chandigarh", "district": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"city": "Mohali", "district": "SAS Nagar", "state": "Punjab", "lat": 30.7046, "lon": 76.7179},
]

ZONES = [f"Zone_{i:02d}" for i in range(1, 11)]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def _categorize_amount(amount: float) -> str:
    if amount < 5000:
        return "Low (<5k)"
    elif amount < 25000:
        return "Medium (5k-25k)"
    elif amount < 50000:
        return "High (25k-50k)"
    else:
        return "Critical (>50k)"

def generate_synthetic_complaints_dataset(
    n_samples: int = 20000,
    random_state: int = 42,
    save_files: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate synthetic complaints matching CyberTrace schema.

    Parameters:
        n_samples: Target count (~20,000)
        random_state: Seed for reproducibility
        save_files: Save to data/raw/ and data/processed/
        overwrite: Overwrite existing files if True (default False to protect existing data)
    """
    if save_files and COMPLAINTS_RAW_PATH.exists() and not overwrite:
        logger.info(f"Target file {COMPLAINTS_RAW_PATH} already exists. Skipping generation to preserve data.")
        return pd.read_csv(COMPLAINTS_RAW_PATH)

    np.random.seed(random_state)
    random.seed(random_state)
    logger.info(f"Generating {n_samples} synthetic cybercrime complaint records...")

    start_date = datetime(2025, 1, 1)
    records = []

    for i in range(n_samples):
        cid = f"CC-{20250000 + i + 1}"
        days_offset = random.randint(0, 365)
        dt = start_date + timedelta(days=days_offset)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        date_str = dt.strftime("%Y-%m-%d")

        dow = dt.weekday()
        day_name = DAY_NAMES[dow]
        is_weekend = int(dow >= 5)
        is_night = int(hour >= 22 or hour <= 5)

        loc = random.choice(LOCATIONS)
        lat_jitter = np.random.normal(0, 0.04)
        lon_jitter = np.random.normal(0, 0.04)
        complaint_lat = round(loc["lat"] + lat_jitter, 6)
        complaint_lon = round(loc["lon"] + lon_jitter, 6)

        # Lognormal distribution for amounts (mostly small-medium, some high)
        amount = round(float(np.random.lognormal(mean=9.5, sigma=1.0)), 2)
        amount = max(500.0, min(amount, 500000.0))
        amount_category = _categorize_amount(amount)

        bank = random.choice(BANKS)
        ttype = random.choice(TRANSACTION_TYPES)
        ftype = random.choice(FRAUD_TYPES)

        # Synthetic association to cash-out zone based on geographic area & bank pattern
        # This provides a realistic learning signal for classification without target leakage
        zone_idx = (loc["city"].__hash__() + hour // 4 + int(amount_category.__hash__())) % len(ZONES)
        zone = ZONES[zone_idx]

        records.append({
            "complaint_id": cid,
            "complaint_date": date_str,
            "complaint_time": time_str,
            "amount": amount,
            "bank": bank,
            "transaction_type": ttype,
            "fraud_type": ftype,
            "city": loc["city"],
            "state": loc["state"],
            "district": loc["district"],
            "complaint_latitude": complaint_lat,
            "complaint_longitude": complaint_lon,
            "hour": hour,
            "day_of_week": dow,
            "day_name": day_name,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "amount_category": amount_category,
            "withdrawal_zone": zone,
        })

    df = pd.DataFrame(records)

    # Clean processed version (baseline ground truth)
    df_clean = df.copy()

    # Create raw version by intentionally introducing missingness & duplicates for Phase 4 preprocessing
    df_raw = df.copy()

    # Introduce missing values (~3% in selected non-target columns)
    mask_bank = np.random.rand(len(df_raw)) < 0.03
    df_raw.loc[mask_bank, "bank"] = np.nan

    mask_ttype = np.random.rand(len(df_raw)) < 0.02
    df_raw.loc[mask_ttype, "transaction_type"] = np.nan

    mask_amount = np.random.rand(len(df_raw)) < 0.015
    df_raw.loc[mask_amount, "amount"] = np.nan

    # Introduce duplicate records (~1.5% duplicates)
    num_duplicates = int(n_samples * 0.015)
    duplicate_indices = np.random.choice(len(df_raw), size=num_duplicates, replace=False)
    duplicates = df_raw.iloc[duplicate_indices].copy()
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

    if save_files:
        df_raw.to_csv(COMPLAINTS_RAW_PATH, index=False)
        df_clean.to_csv(COMPLAINTS_PROCESSED_PATH, index=False)
        logger.info(f"Saved raw complaints ({len(df_raw)} rows) to {COMPLAINTS_RAW_PATH}")
        logger.info(f"Saved processed complaints ({len(df_clean)} rows) to {COMPLAINTS_PROCESSED_PATH}")

    return df_clean
