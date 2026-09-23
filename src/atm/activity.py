"""Historical ATM Activity Analysis Module.

NOTE: All historical activity data is synthetic for modeling and research purposes.
Does not reflect actual banking transaction logs.
"""

from typing import Dict, Any, Optional
import pandas as pd
from src.data.loader import load_atm_activity
from src.utils.logging import get_logger

logger = get_logger("ATMActivity")

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
    fraud_tx = atm_records["fraud_withdrawal_count"].sum()
    high_val = atm_records["high_value_withdrawal_count"].sum()

    hour_active = False
    if hour is not None and "hour" in atm_records.columns:
        hour_records = atm_records[atm_records["hour"] == hour]
        hour_active = not hour_records.empty and (hour_records["transaction_count"].sum() > 0)

    return {
        "atm_id": atm_id,
        "total_recorded_transactions": int(total_tx),
        "historical_fraud_withdrawals": int(fraud_tx),
        "high_value_withdrawal_rate": round(float(high_val / max(1, total_tx)), 3),
        "is_historically_active_at_hour": hour_active,
        "is_synthetic_data": True,
    }
