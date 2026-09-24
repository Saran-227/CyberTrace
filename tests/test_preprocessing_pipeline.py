"""Comprehensive Unit and Integration Tests for CyberTrace Phase 4: ML Preprocessing & Feature Engineering."""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    TARGET_COLUMN,
    FEATURE_SET_FULL,
    FEATURE_SET_GEOGRAPHIC_BLIND,
)
from src.preprocessing.cleaning import (
    clean_complaints_data,
    remove_duplicates,
    impute_missing_values,
    normalize_categorical_values,
    validate_cleaning_bounds,
)
from src.preprocessing.features import (
    categorize_amount,
    add_engineered_features,
    engineer_complaint_features,
)
from src.preprocessing.pipeline import (
    prepare_dataset,
    build_preprocessor,
    build_feature_sets,
    split_data,
    get_feature_names,
    get_stratified_cv,
    compute_class_weights,
)


@pytest.fixture(scope="module")
def raw_complaints_df():
    """Load processed complaints dataset."""
    assert COMPLAINTS_PROCESSED_PATH.exists()
    return pd.read_csv(COMPLAINTS_PROCESSED_PATH)


@pytest.fixture(scope="module")
def full_dataset(raw_complaints_df):
    """Prepare full feature set dataset."""
    return prepare_dataset(raw_complaints_df, feature_set="full")


@pytest.fixture(scope="module")
def geo_blind_dataset(raw_complaints_df):
    """Prepare geographic-blind feature set dataset."""
    return prepare_dataset(raw_complaints_df, feature_set="geographic_blind")


# -------------------------------------------------------------
# LEAKAGE TESTS
# -------------------------------------------------------------

def test_01_withdrawal_zone_never_in_x(full_dataset, geo_blind_dataset):
    """1. Test that withdrawal_zone is never in X for both feature configurations."""
    X_full, _ = full_dataset
    X_geo, _ = geo_blind_dataset

    assert TARGET_COLUMN not in X_full.columns
    assert TARGET_COLUMN not in X_geo.columns


def test_02_synthetic_cashout_lat_cannot_enter_x(full_dataset, geo_blind_dataset):
    """2. Test that synthetic_cashout_latitude is never in X."""
    X_full, _ = full_dataset
    X_geo, _ = geo_blind_dataset

    forbidden = ["synthetic_cashout_latitude", "cashout_latitude"]
    for col in forbidden:
        assert col not in X_full.columns
        assert col not in X_geo.columns


def test_03_synthetic_cashout_lon_cannot_enter_x(full_dataset, geo_blind_dataset):
    """3. Test that synthetic_cashout_longitude is never in X."""
    X_full, _ = full_dataset
    X_geo, _ = geo_blind_dataset

    forbidden = ["synthetic_cashout_longitude", "cashout_longitude"]
    for col in forbidden:
        assert col not in X_full.columns
        assert col not in X_geo.columns


def test_04_target_distribution_unchanged_by_preprocessing(raw_complaints_df, full_dataset):
    """4. Test that the target distribution y is bitwise identical to raw target."""
    _, y_prep = full_dataset
    y_raw = raw_complaints_df[TARGET_COLUMN]

    pd.testing.assert_series_equal(y_prep.reset_index(drop=True), y_raw.reset_index(drop=True))
    assert y_prep.value_counts().equals(y_raw.value_counts())


def test_05_train_test_split_is_stratified(full_dataset):
    """5. Test that train/test split maintains exact proportional class distributions."""
    X, y = full_dataset
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    assert len(X_train) == 16000
    assert len(X_test) == 4000
    assert len(y_train) == 16000
    assert len(y_test) == 4000

    train_props = y_train.value_counts(normalize=True).sort_index()
    test_props = y_test.value_counts(normalize=True).sort_index()

    # Divergence across all 10 classes should be negligible (< 0.001)
    max_divergence = (train_props - test_props).abs().max()
    assert max_divergence < 0.001


def test_06_preprocessing_handles_unseen_categorical_values(full_dataset):
    """6. Test that preprocessor encodes unseen categorical values without raising errors."""
    X, y = full_dataset
    X_train, X_test, _, _ = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    preprocessor = build_preprocessor("full")
    preprocessor.fit(X_train)

    # Create unseen categories in test instance
    novel_instance = X_test.iloc[0:1].copy()
    novel_instance["bank"] = "NEVER_SEEN_MOON_BANK"
    novel_instance["city"] = "NEVER_SEEN_CITY"
    novel_instance["fraud_type"] = "NEVER_SEEN_SCAM_2099"

    # Must transform cleanly via handle_unknown='ignore'
    transformed = preprocessor.transform(novel_instance)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_07_missing_values_handled_by_imputer(full_dataset):
    """7. Test that missing numerical and categorical values are properly imputed."""
    X, y = full_dataset
    X_train, X_test, _, _ = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    preprocessor = build_preprocessor("full")
    preprocessor.fit(X_train)

    # Inject NaNs in test instance
    nan_instance = X_test.iloc[0:2].copy()
    nan_instance.loc[nan_instance.index[0], "amount"] = np.nan
    nan_instance.loc[nan_instance.index[0], "bank"] = np.nan
    nan_instance.loc[nan_instance.index[1], "complaint_latitude"] = np.nan

    transformed = preprocessor.transform(nan_instance)
    assert not np.isnan(transformed).any()


def test_08_no_leakage_from_test_set_during_fitting(full_dataset):
    """8. Test that preprocessor statistics are fitted solely on training data."""
    X, y = full_dataset
    X_train, X_test, _, _ = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    preprocessor = build_preprocessor("full")
    preprocessor.fit(X_train)

    # Numeric scaler mean should match X_train mean, NOT full X mean
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train_amount_mean = X_train["amount"].mean()
    scaler_amount_mean = scaler.mean_[0]  # First numerical feature is 'amount'

    assert np.isclose(scaler_amount_mean, train_amount_mean, atol=1e-3)


def test_09_feature_configurations_are_deterministic():
    """9. Test that build_feature_sets returns deterministic, immutable schemas."""
    sets1 = build_feature_sets()
    sets2 = build_feature_sets()

    assert sets1["full"]["numerical"] == sets2["full"]["numerical"]
    assert sets1["full"]["categorical"] == sets2["full"]["categorical"]
    assert sets1["geographic_blind"]["numerical"] == sets2["geographic_blind"]["numerical"]
    assert sets1["geographic_blind"]["categorical"] == sets2["geographic_blind"]["categorical"]


def test_10_geographic_blind_excludes_all_direct_geography(geo_blind_dataset):
    """10. Test that the geographic-blind feature set contains zero direct spatial identifiers."""
    X_geo, _ = geo_blind_dataset
    forbidden_geo = ["city", "state", "district", "complaint_latitude", "complaint_longitude"]

    for col in forbidden_geo:
        assert col not in X_geo.columns, f"Direct geographic column '{col}' leaked into geographic-blind configuration!"


# -------------------------------------------------------------
# DATA QUALITY TESTS
# -------------------------------------------------------------

def test_11_valid_amounts(full_dataset):
    """11. Test that all amounts in preprocessed features are strictly positive."""
    X, _ = full_dataset
    assert (X["amount"] > 0).all()
    assert (X["amount_log"] > 0).all()


def test_12_valid_coordinates(full_dataset):
    """12. Test that complaint coordinates fall strictly within physical and study boundaries."""
    X, _ = full_dataset
    assert (X["complaint_latitude"] >= -90.0).all() and (X["complaint_latitude"] <= 90.0).all()
    assert (X["complaint_longitude"] >= -180.0).all() and (X["complaint_longitude"] <= 180.0).all()


def test_13_valid_categorical_values(full_dataset):
    """13. Test that categorical features do not contain empty or whitespace strings."""
    X, _ = full_dataset
    for col in ["bank", "transaction_type", "fraud_type", "city", "state", "district", "amount_category"]:
        assert not (X[col] == "").any()
        assert not (X[col] == " ").any()


def test_14_valid_hour_range(full_dataset):
    """14. Test that hour and circular hour embeddings fall within mathematical bounds."""
    X, _ = full_dataset
    assert (X["hour"] >= 0).all() and (X["hour"] <= 23).all()
    assert (X["hour_sin"] >= -1.0).all() and (X["hour_sin"] <= 1.0).all()
    assert (X["hour_cos"] >= -1.0).all() and (X["hour_cos"] <= 1.0).all()


def test_15_valid_day_of_week_range(full_dataset):
    """15. Test that day_of_week and circular embeddings fall within mathematical bounds."""
    X, _ = full_dataset
    assert (X["day_of_week"] >= 0).all() and (X["day_of_week"] <= 6).all()
    assert (X["day_of_week_sin"] >= -1.0).all() and (X["day_of_week_sin"] <= 1.0).all()
    assert (X["day_of_week_cos"] >= -1.0).all() and (X["day_of_week_cos"] <= 1.0).all()


def test_16_expected_dataset_size_and_target_completeness(full_dataset):
    """16. Test that total dataset contains exactly 20,000 instances and all 10 target classes."""
    X, y = full_dataset
    assert len(X) == 20000
    assert len(y) == 20000
    assert y.nunique() == 10
    expected_zones = {f"Zone_{i:02d}" for i in range(1, 11)}
    assert set(y.unique()) == expected_zones


def test_17_class_weights_computation(full_dataset):
    """17. Test that compute_class_weights properly weights minority class higher than majority."""
    _, y = full_dataset
    weights = compute_class_weights(y)
    assert len(weights) == 10
    # Zone_09 (minority) must have highest weight, Zone_07 (majority) lowest
    assert weights["Zone_09"] > weights["Zone_07"]
    assert weights["Zone_09"] > 5.0
    assert weights["Zone_07"] < 1.0
