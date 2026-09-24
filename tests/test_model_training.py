"""Comprehensive Unit and Integration Tests for CyberTrace Phase 5: Location Classification Model Training & Experimentation."""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    LOCATION_CLASSIFIER_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    CANONICAL_COMPLAINTS_SHA256,
    EXPERIMENT_CONFIGS,
    FEATURE_SET_FULL,
    FEATURE_SET_GEOGRAPHIC_BLIND,
    TARGET_COLUMN,
)
from src.preprocessing.pipeline import (
    prepare_dataset,
    split_data,
    build_preprocessor,
)


@pytest.fixture(scope="module")
def registry_data():
    """Load model registry JSON."""
    reg_path = LOCATION_CLASSIFIER_DIR / "model_registry.json"
    assert reg_path.exists(), f"Model registry missing at {reg_path}"
    with open(reg_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def source_data():
    """Load canonical source complaint dataframe."""
    assert COMPLAINTS_PROCESSED_PATH.exists()
    return pd.read_csv(COMPLAINTS_PROCESSED_PATH)


def test_01_configured_experiments_have_unique_ids():
    """1. Test that every configured experiment has a unique identifier."""
    exp_ids = [c["experiment_id"] for c in EXPERIMENT_CONFIGS]
    assert len(exp_ids) == 14
    assert len(set(exp_ids)) == 14


def test_02_training_uses_canonical_dataset_hash(registry_data):
    """2. Test that training pipeline verified the canonical dataset SHA-256 fingerprint."""
    with open(COMPLAINTS_PROCESSED_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert actual_hash == CANONICAL_COMPLAINTS_SHA256
    assert registry_data["dataset_sha256"] == CANONICAL_COMPLAINTS_SHA256


def test_03_x_never_contains_withdrawal_zone():
    """3. Test that input feature matrix X strictly excludes the target column."""
    X_full, _ = prepare_dataset(feature_set="full")
    X_geo, _ = prepare_dataset(feature_set="geographic_blind")

    assert TARGET_COLUMN not in X_full.columns
    assert TARGET_COLUMN not in X_geo.columns


def test_04_x_never_contains_hidden_cashout_coordinates():
    """4. Test that forbidden hidden cash-out coordinates cannot enter X."""
    X_full, _ = prepare_dataset(feature_set="full")
    X_geo, _ = prepare_dataset(feature_set="geographic_blind")

    forbidden = [
        "synthetic_cashout_latitude",
        "synthetic_cashout_longitude",
        "cashout_latitude",
        "cashout_longitude",
    ]
    for col in forbidden:
        assert col not in X_full.columns
        assert col not in X_geo.columns


def test_05_saved_pipelines_contain_preprocessor_and_estimator():
    """5. Test that every saved artifact is a valid Pipeline with preprocessor and classifier."""
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        art_path = LOCATION_CLASSIFIER_DIR / f"{exp_id}.joblib"
        assert art_path.exists(), f"Artifact missing: {art_path}"

        pipeline = joblib.load(art_path)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps
        assert hasattr(pipeline.named_steps["classifier"], "predict")


def test_06_stratified_split_preserves_all_classes(source_data):
    """6. Test that train/test split preserves all 10 target classes in both partitions."""
    X, y = prepare_dataset(feature_set="full")
    _, _, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    assert y_train.nunique() == 10
    assert y_test.nunique() == 10
    assert set(y_train.unique()) == set(source_data[TARGET_COLUMN].unique())
    assert set(y_test.unique()) == set(source_data[TARGET_COLUMN].unique())


def test_07_probability_rows_sum_to_one():
    """7. Test that prediction probability distributions sum to 1.0 within floating tolerance."""
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        assert pred_path.exists()

        pred_df = pd.read_csv(pred_path)
        prob_cols = [c for c in pred_df.columns if c.startswith("prob_zone_")]
        assert len(prob_cols) == 10

        # Row sums
        row_sums = pred_df[prob_cols].sum(axis=1)
        assert np.isclose(row_sums, 1.0, atol=1e-2).all()


def test_08_predicted_class_equals_maximum_probability_class():
    """8. Test that predicted_zone has probability equal to the maximum probability in its row."""
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        pred_df = pd.read_csv(pred_path)

        prob_cols = [f"prob_zone_{i:02d}" for i in range(1, 11)]

        # Extract probability assigned to the predicted class
        predicted_probs = np.array([
            pred_df.loc[i, f"prob_{pz.lower()}"]
            for i, pz in enumerate(pred_df["predicted_zone"])
        ])
        max_probs = pred_df[prob_cols].max(axis=1).values

        # The probability of the predicted zone must equal the maximum probability
        assert np.isclose(predicted_probs, max_probs, atol=1e-4).all()


def test_09_all_14_saved_model_artifacts_exist():
    """9. Test that all 14 configured model artifacts exist on disk."""
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        art_path = LOCATION_CLASSIFIER_DIR / f"{exp_id}.joblib"
        assert art_path.exists()
        assert art_path.stat().st_size > 1000  # Non-empty file


def test_10_model_registry_contains_every_successful_model(registry_data):
    """10. Test that model_registry.json contains records for all 14 experiments."""
    assert "models" in registry_data
    assert len(registry_data["models"]) == 14
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        assert exp_id in registry_data["models"]
        entry = registry_data["models"][exp_id]
        assert "cv_metrics" in entry
        assert "test_metrics" in entry
        assert "hyperparameters" in entry
        assert "artifact_path" in entry


def test_11_feature_set_definitions_are_correct():
    """11. Test that FEATURE_SET_FULL and FEATURE_SET_GEOGRAPHIC_BLIND match requirements."""
    assert len(FEATURE_SET_FULL["numerical"]) == 12
    assert len(FEATURE_SET_FULL["categorical"]) == 7
    assert len(FEATURE_SET_GEOGRAPHIC_BLIND["numerical"]) == 10
    assert len(FEATURE_SET_GEOGRAPHIC_BLIND["categorical"]) == 4


def test_12_geographic_blind_excludes_all_direct_geography():
    """12. Test that geographic-blind configuration contains zero spatial columns."""
    blind_cols = FEATURE_SET_GEOGRAPHIC_BLIND["numerical"] + FEATURE_SET_GEOGRAPHIC_BLIND["categorical"]
    forbidden_geo = ["city", "state", "district", "complaint_latitude", "complaint_longitude"]
    for col in forbidden_geo:
        assert col not in blind_cols


def test_13_test_data_is_not_used_during_fitting():
    """13. Test that pipeline preprocessor is fitted strictly on training data."""
    X, y = prepare_dataset(feature_set="full")
    X_train, X_test, _, _ = split_data(X, y, test_size=0.20, random_state=42, stratify=True)

    art_path = LOCATION_CLASSIFIER_DIR / "random_forest_full_none.joblib"
    pipeline = joblib.load(art_path)
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]

    # Scaler mean must match X_train mean for amount, not overall dataset mean
    train_mean = X_train["amount"].mean()
    scaler_mean = scaler.mean_[0]
    assert np.isclose(train_mean, scaler_mean, atol=1e-3)


def test_14_existing_phase_reports_and_diagnostics_exist():
    """14. Test that all Phase 5 synthesis reports and diagnostics exist."""
    assert (REPORTS_DIR / "training_log.csv").exists()
    assert (REPORTS_DIR / "model_comparison.csv").exists()
    assert (REPORTS_DIR / "geographic_signal_comparison.csv").exists()
    assert (REPORTS_DIR / "class_weight_comparison.csv").exists()
    assert (REPORTS_DIR / "overfitting_diagnostics.csv").exists()
    assert (REPORTS_DIR / "figures" / "confusion_matrices").is_dir()
    assert len(list((REPORTS_DIR / "figures" / "confusion_matrices").glob("*.png"))) == 14
    assert len(list((REPORTS_DIR / "per_zone_metrics").glob("*.csv"))) == 14
    assert len(list((REPORTS_DIR / "predictions").glob("*.csv"))) == 14
