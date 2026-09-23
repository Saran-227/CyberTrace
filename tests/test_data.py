"""Unit tests for dataset loading, validation, and generation interfaces."""

import pytest
import pandas as pd
from src.data.generator import generate_synthetic_complaints_dataset
from src.data.validator import validate_complaints_data
from src.data.loader import get_dataset_status
from src.config import COMPLAINT_SCHEMA, TARGET_COLUMN

def test_generate_synthetic_sample():
    """Verify synthetic dataset generator produces correct schema and record count."""
    df = generate_synthetic_complaints_dataset(n_samples=50, random_state=123, save_files=False)
    assert len(df) == 50
    for col in COMPLAINT_SCHEMA:
        assert col in df.columns
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].nunique() > 1

def test_validator_detects_duplicates_and_missing():
    """Test validation reports duplicate rows and missing values."""
    df = pd.DataFrame([
        {"complaint_id": "C1", "amount": 1000.0, "bank": "HDFC", "withdrawal_zone": "Zone_01"},
        {"complaint_id": "C1", "amount": 1000.0, "bank": "HDFC", "withdrawal_zone": "Zone_01"},
        {"complaint_id": "C2", "amount": None, "bank": "SBI", "withdrawal_zone": "Zone_02"},
    ])
    res = validate_complaints_data(df)
    assert res["duplicate_rows_count"] == 1
    assert "amount" in res["missing_values_by_column"]
    assert res["missing_values_by_column"]["amount"] == 1

def test_dataset_status_interface():
    """Verify status dictionary contains all expected dataset keys."""
    status = get_dataset_status()
    expected_keys = [
        "raw_complaints",
        "processed_complaints",
        "raw_atm_locations",
        "processed_atm_locations",
        "raw_atm_activity",
        "processed_atm_activity",
    ]
    for key in expected_keys:
        assert key in status
        assert "exists" in status[key]
        assert "records" in status[key]
