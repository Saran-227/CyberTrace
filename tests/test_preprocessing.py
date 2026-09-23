"""Unit tests for preprocessing, deduplication, imputation, and leakage checks."""

import pytest
import pandas as pd
from src.preprocessing.cleaning import remove_duplicates, impute_missing_values
from src.preprocessing.features import categorize_amount, engineer_complaint_features
from src.preprocessing.pipeline import build_preprocessing_pipeline, prepare_xy
from src.config import TARGET_COLUMN

def test_amount_categorization():
    """Verify threshold categorization for financial amounts."""
    assert categorize_amount(2500) == "Low (<5k)"
    assert categorize_amount(15000) == "Medium (5k-25k)"
    assert categorize_amount(35000) == "High (25k-50k)"
    assert categorize_amount(80000) == "Critical (>50k)"

def test_deduplication():
    """Verify duplicate removal drops identical rows."""
    df = pd.DataFrame([
        {"id": 1, "val": "a"},
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
    ])
    cleaned, dropped = remove_duplicates(df)
    assert len(cleaned) == 2
    assert dropped == 1

def test_missing_value_imputation():
    """Verify imputation replaces NaNs with standard defaults."""
    df = pd.DataFrame([
        {"amount": None, "bank": None},
        {"amount": 10000.0, "bank": "HDFC"},
    ])
    imputed = impute_missing_values(df)
    assert not imputed["amount"].isnull().any()
    assert imputed["bank"].iloc[0] == "Unknown Bank"

def test_zero_target_leakage_enforcement():
    """Verify target leakage check raises ValueError if withdrawal_zone is in features."""
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        build_preprocessing_pipeline(
            numerical_cols=["amount"],
            categorical_cols=[TARGET_COLUMN],
        )

def test_prepare_xy_strict_separation():
    """Verify prepare_xy extracts X without withdrawal_zone and extracts y separately."""
    df = pd.DataFrame([
        {"amount": 5000, "bank": "HDFC", "withdrawal_zone": "Zone_01"},
        {"amount": 15000, "bank": "SBI", "withdrawal_zone": "Zone_02"},
    ])
    X, y = prepare_xy(df)
    assert TARGET_COLUMN not in X.columns
    assert len(y) == 2
    assert y.iloc[0] == "Zone_01"
