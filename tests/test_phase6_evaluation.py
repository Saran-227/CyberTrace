"""Comprehensive Unit and Verification Tests for CyberTrace Phase 6: Model Evaluation, Error Analysis & Pipeline Selection."""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import (
    BASE_DIR,
    COMPLAINTS_PROCESSED_PATH,
    LOCATION_CLASSIFIER_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    CANONICAL_COMPLAINTS_SHA256,
    EXPERIMENT_CONFIGS,
    TARGET_COLUMN,
)


@pytest.fixture(scope="module")
def model_registry():
    """Load and parse model registry JSON."""
    reg_path = LOCATION_CLASSIFIER_DIR / "model_registry.json"
    assert reg_path.exists()
    with open(reg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_01_model_registry_is_readable(model_registry):
    """1. Test that model_registry.json is well-formed, readable, and contains 14 models."""
    assert "models" in model_registry
    assert len(model_registry["models"]) == 14
    assert model_registry["training_rows"] == 16000
    assert model_registry["test_rows"] == 4000


def test_02_all_selected_artifacts_exist():
    """2. Test that primary and fallback model artifacts exist and are loadable."""
    primary_txt = LOCATION_CLASSIFIER_DIR / "PRIMARY_MODEL.txt"
    fallback_txt = LOCATION_CLASSIFIER_DIR / "FALLBACK_MODEL.txt"
    assert primary_txt.exists()
    assert fallback_txt.exists()

    with open(primary_txt, "r") as f:
        p_lines = f.readlines()
        primary_id = p_lines[0].split(":")[1].strip()

    with open(fallback_txt, "r") as f:
        f_lines = f.readlines()
        fallback_id = f_lines[0].split(":")[1].strip()

    primary_path = LOCATION_CLASSIFIER_DIR / f"{primary_id}.joblib"
    fallback_path = LOCATION_CLASSIFIER_DIR / f"{fallback_id}.joblib"

    assert primary_path.exists()
    assert fallback_path.exists()

    p_pipe = joblib.load(primary_path)
    f_pipe = joblib.load(fallback_path)
    assert hasattr(p_pipe, "predict")
    assert hasattr(f_pipe, "predict")


def test_03_dataset_hash_matches_canonical(model_registry):
    """3. Test that canonical dataset SHA-256 fingerprint matches registered hash."""
    with open(COMPLAINTS_PROCESSED_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert actual_hash == CANONICAL_COMPLAINTS_SHA256
    assert model_registry["dataset_sha256"] == CANONICAL_COMPLAINTS_SHA256


def test_04_prediction_files_have_required_columns():
    """4. Test that prediction CSV files contain complaint_id, actual_zone, predicted_zone, confidence."""
    required_cols = {"complaint_id", "actual_zone", "predicted_zone", "prediction_confidence"}
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        assert pred_path.exists()
        df = pd.read_csv(pred_path, nrows=5)
        assert required_cols.issubset(set(df.columns))


def test_05_all_10_probability_columns_exist():
    """5. Test that prediction CSV files contain exactly prob_zone_01 through prob_zone_10."""
    expected_prob_cols = [f"prob_zone_{i:02d}" for i in range(1, 11)]
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        df = pd.read_csv(pred_path, nrows=5)
        for col in expected_prob_cols:
            assert col in df.columns


def test_06_probabilities_are_bounded_zero_to_one():
    """6. Test that all predicted probabilities fall strictly in [0.0, 1.0]."""
    prob_cols = [f"prob_zone_{i:02d}" for i in range(1, 11)]
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        df = pd.read_csv(pred_path)
        probs = df[prob_cols].values
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()


def test_07_probability_rows_sum_to_approximately_one():
    """7. Test that every row's probability distribution sums to 1.0."""
    prob_cols = [f"prob_zone_{i:02d}" for i in range(1, 11)]
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        df = pd.read_csv(pred_path)
        row_sums = df[prob_cols].sum(axis=1).values
        assert np.isclose(row_sums, 1.0, atol=1e-2).all()


def test_08_predicted_zone_matches_maximum_probability():
    """8. Test that predicted class corresponds to maximum row probability."""
    prob_cols = [f"prob_zone_{i:02d}" for i in range(1, 11)]
    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
        df = pd.read_csv(pred_path)

        predicted_probs = np.array([
            df.loc[i, f"prob_{pz.lower()}"]
            for i, pz in enumerate(df["predicted_zone"])
        ])
        max_probs = df[prob_cols].max(axis=1).values
        assert np.isclose(predicted_probs, max_probs, atol=1e-4).all()


def test_09_per_zone_metrics_cover_all_10_zones():
    """9. Test that per-zone error summary covers all 10 canonical zones for every model."""
    expected_zones = {f"Zone_{i:02d}" for i in range(1, 11)}
    summary_path = REPORTS_DIR / "phase6_per_zone_summary.csv"
    assert summary_path.exists()
    pz_df = pd.read_csv(summary_path)

    for exp in EXPERIMENT_CONFIGS:
        exp_id = exp["experiment_id"]
        m_zones = set(pz_df[pz_df["model_id"] == exp_id]["withdrawal_zone"].unique())
        assert m_zones == expected_zones


def test_10_ncr_analysis_contains_zone_07_and_zone_08():
    """10. Test that NCR boundary analysis isolates Zone_07 and Zone_08 samples."""
    ncr_path = REPORTS_DIR / "ncr_boundary_errors.csv"
    assert ncr_path.exists()
    ncr_df = pd.read_csv(ncr_path)

    assert set(ncr_df["actual_zone"].unique()) == {"Zone_07", "Zone_08"}
    assert len(ncr_df) > 1000
    assert "prob_zone_07" in ncr_df.columns
    assert "prob_zone_08" in ncr_df.columns


def test_11_no_hidden_cashout_coordinates_introduced():
    """11. Test that hidden cashout coordinates do not appear in any Phase 6 output tables."""
    forbidden = ["synthetic_cashout_latitude", "synthetic_cashout_longitude", "cashout_latitude", "cashout_longitude"]

    check_files = [
        REPORTS_DIR / "ncr_boundary_errors.csv",
        REPORTS_DIR / "uncertain_predictions.csv",
        REPORTS_DIR / "phase6_model_evaluation.csv",
        REPORTS_DIR / "phase6_subgroup_performance.csv",
    ]
    for fpath in check_files:
        if fpath.exists():
            df = pd.read_csv(fpath, nrows=5)
            for col in forbidden:
                assert col not in df.columns, f"Forbidden column '{col}' leaked into {fpath}!"


def test_12_model_selection_references_existing_artifacts():
    """12. Test that model_selection.md references existing model artifacts."""
    sel_path = REPORTS_DIR / "model_selection.md"
    assert sel_path.exists()
    with open(sel_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "random_forest_full_none" in text
    assert "logistic_full_none" in text
    assert (LOCATION_CLASSIFIER_DIR / "random_forest_full_none.joblib").exists()
    assert (LOCATION_CLASSIFIER_DIR / "logistic_full_none.joblib").exists()


def test_13_prediction_contract_contains_all_required_fields():
    """13. Test that location_prediction_contract.md documents all required schema fields."""
    contract_path = Path(BASE_DIR) / "docs" / "location_prediction_contract.md"
    assert contract_path.exists()
    with open(contract_path, "r", encoding="utf-8") as f:
        text = f.read()

    required_fields = [
        "complaint_id",
        "predicted_zone",
        "prediction_confidence",
        "second_best_zone",
        "probability_margin",
        "zone_probabilities",
        "model_id",
    ]
    for field in required_fields:
        assert field in text, f"Missing required field '{field}' in prediction contract!"


def test_14_all_10_phase6_figures_exist():
    """14. Test that all 10 required Phase 6 visualization figures were generated."""
    fig_dir = REPORTS_DIR / "figures" / "phase6"
    assert fig_dir.is_dir()

    expected_figs = [
        "01_model_macro_f1_comparison.png",
        "02_model_balanced_accuracy.png",
        "03_zone_f1_comparison.png",
        "04_ncr_confusion_heatmap.png",
        "05_confidence_distribution.png",
        "06_confidence_vs_accuracy.png",
        "07_geographic_signal_gap.png",
        "08_class_weight_effect.png",
        "09_cv_vs_test_gap.png",
        "10_feature_importance.png",
    ]
    for fig_name in expected_figs:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000
