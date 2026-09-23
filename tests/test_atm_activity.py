"""Unit tests for Phase 2B Synthetic Historical ATM Activity Dataset."""

import pytest
import pandas as pd
import numpy as np

from src.config import (
    ATM_ACTIVITY_PROCESSED_PATH,
    ATM_ACTIVITY_RAW_PATH,
    ATM_ACTIVITY_SCHEMA,
    ATM_LOCATIONS_PROCESSED_PATH,
)
from src.atm.activity import (
    generate_synthetic_atm_activity,
    validate_atm_activity_dataframe,
    get_atm_activity_summary,
)

@pytest.fixture(scope="module")
def sample_activity_df():
    """Module-level fixture to load the generated activity dataset."""
    if ATM_ACTIVITY_PROCESSED_PATH.exists():
        df = pd.read_csv(ATM_ACTIVITY_PROCESSED_PATH)
        return df
    return generate_synthetic_atm_activity(start_date="2026-06-26", days=90, random_seed=42, save_processed=False, save_raw=False)

def test_activity_required_columns_exist(sample_activity_df):
    """1. Test that all required schema columns exist in the activity dataset."""
    expected_cols = [
        "atm_id",
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "transaction_count",
        "cash_withdrawal_count",
        "estimated_cash_volume",
        "activity_score",
        "high_value_withdrawal_count",
        "average_amount",
    ]
    for col in expected_cols:
        assert col in sample_activity_df.columns, f"Required column '{col}' is missing"

def test_fraud_withdrawal_count_is_absent(sample_activity_df):
    """2. Test that fraud_withdrawal_count and any fraud-specific columns are strictly absent."""
    assert "fraud_withdrawal_count" not in sample_activity_df.columns
    fraud_cols = [col for col in sample_activity_df.columns if "fraud" in col.lower()]
    assert len(fraud_cols) == 0, f"Found unexpected fraud column(s): {fraud_cols}"

def test_completed_historical_date_range(sample_activity_df):
    """3. Test that date range is strictly 2026-06-26 through 2026-09-23 (90 completed historical days)."""
    dates = sample_activity_df["date"]
    min_date = dates.min()
    max_date = dates.max()
    unique_dates = dates.nunique()

    assert min_date == "2026-06-26", f"Expected min date 2026-06-26, got {min_date}"
    assert max_date == "2026-09-23", f"Expected max date 2026-09-23, got {max_date}"
    assert unique_dates == 90, f"Expected exactly 90 unique dates, got {unique_dates}"

def test_no_future_dates_present(sample_activity_df):
    """4. Test that no future dates exist (all dates <= 2026-09-23 relative to audit date 2026-09-24)."""
    future_records = sample_activity_df[sample_activity_df["date"] > "2026-09-23"]
    assert len(future_records) == 0, f"Found {len(future_records)} records with future dates past 2026-09-23"

def test_every_atm_id_exists_in_atm_locations(sample_activity_df):
    """5. Test that every atm_id in activity exists in verified atm_locations, with no orphan/nonexistent ATMs."""
    assert ATM_LOCATIONS_PROCESSED_PATH.exists()
    loc_df = pd.read_csv(ATM_LOCATIONS_PROCESSED_PATH)
    valid_ids = set(loc_df["atm_id"].unique())
    activity_ids = set(sample_activity_df["atm_id"].unique())

    # Every activity ID must exist in locations
    orphan_ids = activity_ids - valid_ids
    assert len(orphan_ids) == 0, f"Orphan ATM IDs found: {orphan_ids}"

    # Panipat check: No Panipat ATMs exist in locations, so none should exist in activity
    panipat_atms = loc_df[loc_df["city"].str.lower() == "panipat"]["atm_id"].unique()
    assert len(panipat_atms) == 0
    # Also verify no synthetic Panipat records exist
    loc_cities = sample_activity_df.merge(loc_df[["atm_id", "city"]], on="atm_id", how="left")
    assert "Panipat" not in loc_cities["city"].values

def test_transaction_count_non_negative(sample_activity_df):
    """6. Test transaction_count >= 0 across all records."""
    assert (sample_activity_df["transaction_count"] >= 0).all()

def test_cash_withdrawal_count_non_negative(sample_activity_df):
    """7. Test cash_withdrawal_count >= 0 across all records."""
    assert (sample_activity_df["cash_withdrawal_count"] >= 0).all()

def test_cash_withdrawal_less_than_or_equal_to_transactions(sample_activity_df):
    """8. Test that cash_withdrawal_count <= transaction_count always."""
    violations = sample_activity_df[
        sample_activity_df["cash_withdrawal_count"] > sample_activity_df["transaction_count"]
    ]
    assert len(violations) == 0, f"Found {len(violations)} records where cash_withdrawal > transaction_count"

def test_estimated_cash_volume_non_negative(sample_activity_df):
    """9. Test estimated_cash_volume >= 0 across all records."""
    assert (sample_activity_df["estimated_cash_volume"] >= 0.0).all()
    # If cash_withdrawal_count is 0, volume must be 0.0
    zero_wd = sample_activity_df[sample_activity_df["cash_withdrawal_count"] == 0]
    assert (zero_wd["estimated_cash_volume"] == 0.0).all()

def test_activity_score_bounds(sample_activity_df):
    """10. Test that activity_score is bounded within [0, 100]."""
    scores = sample_activity_df["activity_score"]
    assert (scores >= 0.0).all(), "Found negative activity scores"
    assert (scores <= 100.0).all(), "Found activity scores exceeding 100.0"

def test_date_and_hour_fields_valid(sample_activity_df):
    """11. Test that date strings and hour integers are strictly valid."""
    hours = sample_activity_df["hour"]
    assert hours.min() >= 0
    assert hours.max() <= 23

    parsed_dates = pd.to_datetime(sample_activity_df["date"], format="%Y-%m-%d", errors="coerce")
    assert not parsed_dates.isna().any(), "Found invalid or unparseable date strings"

def test_is_weekend_consistent_with_date(sample_activity_df):
    """12. Test that is_weekend matches day_of_week (1 for Saturday/Sunday, 0 for Mon-Fri)."""
    dow = sample_activity_df["day_of_week"]
    expected_wknd = (dow >= 5).astype(int)
    assert (sample_activity_df["is_weekend"] == expected_wknd).all()

def test_is_night_consistent_with_hour(sample_activity_df):
    """13. Test that is_night is 1 between 22:00 and 05:00, and 0 otherwise."""
    hours = sample_activity_df["hour"]
    expected_night = ((hours >= 22) | (hours <= 5)).astype(int)
    assert (sample_activity_df["is_night"] == expected_night).all()

def test_reproducibility_with_same_seed():
    """14. Test that identical random seed produces bitwise identical DataFrames."""
    df1 = generate_synthetic_atm_activity(start_date="2026-06-26", days=2, random_seed=42, save_processed=False, save_raw=False)
    df2 = generate_synthetic_atm_activity(start_date="2026-06-26", days=2, random_seed=42, save_processed=False, save_raw=False)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seed produces different results
    df3 = generate_synthetic_atm_activity(start_date="2026-06-26", days=2, random_seed=999, save_processed=False, save_raw=False)
    assert not df1["transaction_count"].equals(df3["transaction_count"])

def test_zero_target_leakage_no_withdrawal_zone(sample_activity_df):
    """15. Test that withdrawal_zone and hidden target columns are strictly absent."""
    forbidden = ["withdrawal_zone", "zone", "target", "complaint_latitude", "complaint_longitude"]
    for col in forbidden:
        assert col not in sample_activity_df.columns, f"Target leakage detected! Found '{col}' in activity dataset"

def test_complaint_metadata_is_absent(sample_activity_df):
    """16. Test that complaint metadata columns are strictly absent from ATM activity dataset."""
    complaint_cols = ["complaint_id", "fraud_type", "amount_category", "complainant_bank"]
    for col in complaint_cols:
        assert col not in sample_activity_df.columns, f"Complaint metadata '{col}' leaked into activity dataset!"

def test_validator_detects_invalid_activity_records():
    """17. Test that validate_atm_activity_dataframe catches out-of-bound, future, and inconsistent values."""
    bad_df = pd.DataFrame([{
        "atm_id": "NONEXISTENT_ATM_999",
        "date": "2026-09-30",  # Future date > 2026-09-23
        "hour": 25,  # Invalid hour
        "day_of_week": 0,
        "is_weekend": 1,  # Inconsistent with Monday
        "is_night": 0,    # Inconsistent with 25
        "transaction_count": 5,
        "cash_withdrawal_count": 10,  # Invalid: wd > tx
        "estimated_cash_volume": -500.0,  # Negative
        "activity_score": 150.0,  # > 100
        "high_value_withdrawal_count": 0,
        "average_amount": 0.0,
        "fraud_withdrawal_count": 1,  # Forbidden fraud column
        "withdrawal_zone": "Zone_01",  # Leakage!
    }])

    val = validate_atm_activity_dataframe(bad_df)
    assert val["is_valid"] is False
    assert val["has_withdrawal_zone"] is True
    assert val["has_fraud_column"] is True
    assert val["future_dates_count"] == 1
    assert val["invalid_withdrawal_ratio_count"] == 1
    assert val["negative_cash_volume_count"] == 1
    assert val["invalid_activity_score_count"] == 1
    assert val["invalid_hours_count"] == 1
    assert val["inconsistent_weekend_count"] == 1
    assert val["nonexistent_atms_count"] == 1

def test_atm_activity_summary_helper(sample_activity_df):
    """18. Test get_atm_activity_summary returns expected dictionary structure."""
    atm_id = sample_activity_df["atm_id"].iloc[0]
    summary = get_atm_activity_summary(atm_id, hour=14)

    assert summary["atm_id"] == atm_id
    assert summary["total_recorded_transactions"] >= 0
    assert summary["is_synthetic_data"] is True
    assert "high_value_withdrawal_rate" in summary
    assert "is_historically_active_at_hour" in summary

