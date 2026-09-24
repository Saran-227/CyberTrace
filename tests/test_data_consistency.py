"""Phase 4.1 Data Consistency & Source Dataset Immutability Tests.

Verifies:
1. Source dataset has exactly 20,000 records.
2. complaint_id values are strictly unique.
3. withdrawal_zone distribution matches canonical counts.
4. Preprocessing operations do not modify source data files on disk.
5. Feature extraction does not modify target vector y.
6. Amount statistics (min, max, median, mean) are reproducible.
7. Missingness statistics (bank, transaction_type, district) are reproducible.
"""

import hashlib
import pytest
import pandas as pd
import numpy as np

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    TARGET_COLUMN,
)
from src.preprocessing.cleaning import clean_complaints_data
from src.preprocessing.pipeline import prepare_dataset

CANONICAL_SHA256 = "2be4188500ff2be70522220753ba8bddd0af5466c777bd8e6ef6528d916cb4ff"

EXPECTED_ZONE_DISTRIBUTION = {
    "Zone_01": 1241,
    "Zone_02": 1577,
    "Zone_03": 1818,
    "Zone_04": 2592,
    "Zone_05": 764,
    "Zone_06": 841,
    "Zone_07": 5436,
    "Zone_08": 4339,
    "Zone_09": 232,
    "Zone_10": 1160,
}


@pytest.fixture(scope="module")
def source_df():
    """Load canonical source complaint dataframe."""
    assert COMPLAINTS_PROCESSED_PATH.exists()
    return pd.read_csv(COMPLAINTS_PROCESSED_PATH)


def test_01_source_dataset_has_20000_records(source_df):
    """1. Test that the source dataset contains exactly 20,000 records."""
    assert len(source_df) == 20000
    assert source_df.shape[0] == 20000
    assert source_df.shape[1] == 19


def test_02_complaint_id_values_are_unique(source_df):
    """2. Test that complaint_id values are completely unique with zero duplicates."""
    assert "complaint_id" in source_df.columns
    assert source_df["complaint_id"].nunique() == 20000
    assert source_df["complaint_id"].duplicated().sum() == 0


def test_03_withdrawal_zone_distribution_is_reproducible(source_df):
    """3. Test that the withdrawal_zone distribution exactly matches canonical counts."""
    assert TARGET_COLUMN in source_df.columns
    counts = source_df[TARGET_COLUMN].value_counts().to_dict()
    assert counts == EXPECTED_ZONE_DISTRIBUTION
    assert sum(counts.values()) == 20000


def test_04_phase4_does_not_modify_source_data():
    """4. Test that running Phase 4 preprocessing pipelines preserves source file immutability."""
    # Hash before preprocessing
    with open(COMPLAINTS_PROCESSED_PATH, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    assert hash_before == CANONICAL_SHA256

    # Execute full preprocessing pipelines
    X_full, y_full = prepare_dataset(feature_set="full")
    X_geo, y_geo = prepare_dataset(feature_set="geographic_blind")

    assert len(X_full) == 20000
    assert len(X_geo) == 20000

    # Hash after preprocessing
    with open(COMPLAINTS_PROCESSED_PATH, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert hash_after == CANONICAL_SHA256, "Source file was modified during preprocessing!"


def test_05_feature_extraction_does_not_modify_y(source_df):
    """5. Test that feature extraction produces y identical to source withdrawal_zone."""
    _, y = prepare_dataset(feature_set="full")

    assert isinstance(y, pd.Series)
    assert len(y) == len(source_df)
    assert (y.values == source_df[TARGET_COLUMN].values).all()
    assert y.name == TARGET_COLUMN


def test_06_amount_statistics_are_reproducible(source_df):
    """6. Test that financial amount column statistics are exact and reproducible."""
    amt = source_df["amount"]

    assert amt.min() == 500.0
    assert amt.max() == 2339237.0
    assert amt.median() == 18000.0
    assert np.isclose(amt.mean(), 28607.84, atol=0.01)
    assert amt.quantile(0.25) == 10100.0
    assert amt.quantile(0.75) == 32125.0
    assert np.isclose(amt.quantile(0.95), 74205.0, atol=1.0)
    assert np.isclose(amt.quantile(0.99), 143701.0, atol=1.0)


def test_07_missingness_statistics_are_reproducible(source_df):
    """7. Test that missingness in bank, transaction_type, and district matches expectations."""
    missing = source_df.isnull().sum()

    assert missing["bank"] == 160
    assert missing["transaction_type"] == 120
    assert missing["district"] == 80

    # All other columns must have 0 missing values
    non_missing_cols = [c for c in source_df.columns if c not in ["bank", "transaction_type", "district"]]
    for col in non_missing_cols:
        assert missing[col] == 0, f"Unexpected missing values in column '{col}': {missing[col]}"
