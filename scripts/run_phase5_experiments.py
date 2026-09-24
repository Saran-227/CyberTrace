"""CLI execution script for CyberTrace Phase 5: Location Classification Model Training & Experimentation.

Executes all 14 primary experiments:
- Logistic Regression (Full / Blind, None / Balanced)
- K-Nearest Neighbors (Full / Blind)
- Decision Tree (Full / Blind, None / Balanced)
- Random Forest (Full / Blind, None / Balanced)

Generates:
- models/location_classifier/*.joblib (14 serialized complete pipelines)
- models/location_classifier/model_registry.json
- reports/training_log.csv
- reports/model_comparison.csv
- reports/geographic_signal_comparison.csv
- reports/class_weight_comparison.csv
- reports/overfitting_diagnostics.csv
- reports/figures/confusion_matrices/*.png (14 confusion matrix plots)
- reports/per_zone_metrics/*.csv (14 per-zone performance breakdowns)
- reports/predictions/*_predictions.csv (14 test prediction files with probability columns)
- reports/probability_diagnostics/probability_diagnostics_summary.csv
- reports/feature_importance/*_importance.csv (for applicable models)
"""

import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    LOCATION_CLASSIFIER_DIR,
    REPORTS_DIR,
    CONFUSION_MATRICES_DIR,
    PER_ZONE_METRICS_DIR,
    PREDICTIONS_DIR,
    PROBABILITY_DIAGNOSTICS_DIR,
    FEATURE_IMPORTANCE_DIR,
    CANONICAL_COMPLAINTS_SHA256,
    EXPERIMENT_CONFIGS,
    TARGET_COLUMN,
)
from src.preprocessing.pipeline import (
    prepare_dataset,
    split_data,
)
from src.models.train import train_and_evaluate_experiment
from src.models.evaluate import save_confusion_matrix_plot
from src.utils.logging import get_logger

logger = get_logger("Phase5Runner")


def verify_canonical_dataset_hash() -> str:
    """Verify that source complaints file matches canonical SHA-256 fingerprint."""
    if not COMPLAINTS_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Source file not found at {COMPLAINTS_PROCESSED_PATH}")

    with open(COMPLAINTS_PROCESSED_PATH, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()

    if computed_hash != CANONICAL_COMPLAINTS_SHA256:
        raise ValueError(
            f"CRITICAL DATASET INTEGRITY ERROR:\n"
            f"Expected canonical hash: {CANONICAL_COMPLAINTS_SHA256}\n"
            f"Computed file hash:     {computed_hash}\n"
            f"Source dataset appears modified. Halting Phase 5 execution."
        )

    logger.info(f"Canonical dataset hash verified: {computed_hash}")
    return computed_hash


def run_phase5():
    print("=" * 70)
    print("CYBERTRACE PHASE 5: LOCATION CLASSIFICATION TRAINING & EXPERIMENTATION")
    print("=" * 70)

    # 1. Verify Dataset Immutability & Hash
    dataset_hash = verify_canonical_dataset_hash()

    # 2. Load Raw Complaints Data for Tracking Complaint IDs
    raw_df = pd.read_csv(COMPLAINTS_PROCESSED_PATH)
    complaint_ids = raw_df["complaint_id"]
    zones_canonical = sorted(raw_df[TARGET_COLUMN].unique().tolist())
    print(f"Loaded {len(raw_df):,} complaint records across {len(zones_canonical)} target zones.")

    # 3. Prepare Datasets for both Feature Sets
    print("\nPreparing feature configurations...")
    X_full, y_full = prepare_dataset(feature_set="full")
    X_geo, y_geo = prepare_dataset(feature_set="geographic_blind")

    # Perform identical stratified split (80/20, seed=42)
    X_train_f, X_test_f, y_train_f, y_test_f = split_data(
        X_full, y_full, test_size=0.20, random_state=42, stratify=True
    )
    X_train_g, X_test_g, y_train_g, y_test_g = split_data(
        X_geo, y_geo, test_size=0.20, random_state=42, stratify=True
    )

    # Split complaint_id to track test set rows
    _, ids_test, _, _ = split_data(
        complaint_ids.to_frame(), y_full, test_size=0.20, random_state=42, stratify=True
    )
    test_ids = ids_test["complaint_id"]

    print(f"Train samples: {len(X_train_f):,} | Test samples: {len(X_test_f):,}")
    print(f"Total experiments to execute: {len(EXPERIMENT_CONFIGS)}")

    # Containers for results
    training_log_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    model_registry: Dict[str, Any] = {
        "dataset_sha256": dataset_hash,
        "execution_timestamp": datetime.utcnow().isoformat() + "Z",
        "training_rows": len(X_train_f),
        "test_rows": len(X_test_f),
        "target_column": TARGET_COLUMN,
        "classes": zones_canonical,
        "models": {},
    }
    diagnostics_records: List[Dict[str, Any]] = []

    # Execute all 14 experiments
    for exp_idx, exp_config in enumerate(EXPERIMENT_CONFIGS, 1):
        exp_id = exp_config["experiment_id"]
        model_name = exp_config["model_name"]
        algorithm = exp_config["algorithm"]
        feat_set = exp_config["feature_set"]
        class_wt = exp_config.get("class_weight", None)

        print(f"\n[{exp_idx}/{len(EXPERIMENT_CONFIGS)}] Running {exp_id} ({model_name} | {feat_set} | cw={class_wt})...")
        start_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Select data partition
        if feat_set == "full":
            X_tr, X_te, y_tr, y_te = X_train_f, X_test_f, y_train_f, y_test_f
        else:
            X_tr, X_te, y_tr, y_te = X_train_g, X_test_g, y_train_g, y_test_g

        try:
            exp_res = train_and_evaluate_experiment(
                config=exp_config,
                X_train=X_tr,
                y_train=y_tr,
                X_test=X_te,
                y_test=y_te,
                complaint_ids_test=test_ids,
                classes=np.array(zones_canonical),
                save_artifact=True,
            )
            end_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Training log
            training_log_records.append({
                "experiment_id": exp_id,
                "start_time": start_ts,
                "end_time": end_ts,
                "duration": exp_res["total_duration_seconds"],
                "model": model_name,
                "feature_set": feat_set,
                "class_weight": class_wt if class_wt is not None else "None",
                "status": "SUCCESS",
                "error_message": "",
            })

            cv_m = exp_res["cv_metrics"]
            test_m = exp_res["test_metrics"]

            # Comparison record
            comparison_records.append({
                "experiment_id": exp_id,
                "model": model_name,
                "feature_set": feat_set,
                "class_weight": class_wt if class_wt is not None else "None",
                "cv_accuracy_mean": cv_m["cv_accuracy_mean"],
                "cv_accuracy_std": cv_m["cv_accuracy_std"],
                "cv_balanced_accuracy_mean": cv_m["cv_balanced_accuracy_mean"],
                "cv_macro_f1_mean": cv_m["cv_macro_f1_mean"],
                "cv_macro_f1_std": cv_m["cv_macro_f1_std"],
                "cv_weighted_f1_mean": cv_m["cv_weighted_f1_mean"],
                "test_accuracy": test_m["accuracy"],
                "test_balanced_accuracy": test_m["balanced_accuracy"],
                "test_macro_precision": test_m["precision_macro"],
                "test_macro_recall": test_m["recall_macro"],
                "test_macro_f1": test_m["f1_macro"],
                "test_weighted_f1": test_m["f1_weighted"],
                "test_roc_auc_macro": test_m["roc_auc_macro"],
                "test_roc_auc_weighted": test_m["roc_auc_weighted"],
                "training_time_seconds": exp_res["training_time_seconds"],
            })

            # Save Confusion Matrix Plot
            cm = np.array(test_m["confusion_matrix"])
            cm_path = CONFUSION_MATRICES_DIR / f"{exp_id}.png"
            save_confusion_matrix_plot(
                cm=cm,
                classes=zones_canonical,
                output_path=cm_path,
                title=f"{model_name} [{feat_set}, cw={class_wt}]",
            )

            # Save Per-Zone Metrics
            pz_records = []
            for z in zones_canonical:
                pz = test_m["per_class"].get(z, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0})
                pz_records.append({
                    "withdrawal_zone": z,
                    "precision": pz["precision"],
                    "recall": pz["recall"],
                    "f1_score": pz["f1_score"],
                    "support": pz["support"],
                })
            pz_df = pd.DataFrame(pz_records)
            pz_path = PER_ZONE_METRICS_DIR / f"{exp_id}.csv"
            pz_df.to_csv(pz_path, index=False)

            # Save Probability Outputs and Predictions
            y_pred = exp_res["y_test_pred"]
            y_prob = exp_res["y_test_prob"]

            if y_prob is not None:
                max_probs = np.max(y_prob, axis=1)
                pred_dict = {
                    "complaint_id": test_ids.values,
                    "actual_zone": y_te.values,
                    "predicted_zone": y_pred,
                    "prediction_confidence": np.round(max_probs, 4),
                }
                for z_idx, z in enumerate(zones_canonical):
                    pred_dict[f"prob_{z.lower()}"] = np.round(y_prob[:, z_idx], 4)

                pred_df = pd.DataFrame(pred_dict)
                pred_path = PREDICTIONS_DIR / f"{exp_id}_predictions.csv"
                pred_df.to_csv(pred_path, index=False)

                # Probability diagnostics
                is_correct = (pred_df["actual_zone"] == pred_df["predicted_zone"])
                conf_correct = max_probs[is_correct]
                conf_incorrect = max_probs[~is_correct]

                diagnostics_records.append({
                    "experiment_id": exp_id,
                    "model": model_name,
                    "feature_set": feat_set,
                    "class_weight": class_wt if class_wt is not None else "None",
                    "confidence_mean": round(float(np.mean(max_probs)), 4),
                    "confidence_std": round(float(np.std(max_probs)), 4),
                    "confidence_min": round(float(np.min(max_probs)), 4),
                    "confidence_max": round(float(np.max(max_probs)), 4),
                    "avg_confidence_correct": round(float(np.mean(conf_correct)), 4) if len(conf_correct) > 0 else 0.0,
                    "avg_confidence_incorrect": round(float(np.mean(conf_incorrect)), 4) if len(conf_incorrect) > 0 else 0.0,
                    "calibration_status": "Raw uncalibrated probabilities",
                })

            # Save Feature Importance if applicable
            pipeline = exp_res["pipeline"]
            feature_names = exp_res["feature_names"]
            clf = pipeline.named_steps["classifier"]

            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                imp_df = pd.DataFrame({
                    "feature_name": feature_names,
                    "importance": np.round(imp, 6),
                }).sort_values(by="importance", ascending=False).reset_index(drop=True)
                imp_df["rank"] = imp_df.index + 1
                imp_path = FEATURE_IMPORTANCE_DIR / f"{exp_id}_importance.csv"
                imp_df.to_csv(imp_path, index=False)
            elif hasattr(clf, "coef_"):
                # Multiclass coefficients: compute mean absolute magnitude across classes
                coef_mag = np.mean(np.abs(clf.coef_), axis=0)
                imp_df = pd.DataFrame({
                    "feature_name": feature_names,
                    "mean_abs_coefficient": np.round(coef_mag, 6),
                }).sort_values(by="mean_abs_coefficient", ascending=False).reset_index(drop=True)
                imp_df["rank"] = imp_df.index + 1
                imp_path = FEATURE_IMPORTANCE_DIR / f"{exp_id}_importance.csv"
                imp_df.to_csv(imp_path, index=False)

            # Model Registry Entry
            model_registry["models"][exp_id] = {
                "experiment_id": exp_id,
                "model_name": model_name,
                "algorithm": algorithm,
                "feature_set": feat_set,
                "class_weight": class_wt if class_wt is not None else "None",
                "hyperparameters": exp_config["hyperparameters"],
                "feature_count": exp_res["feature_count"],
                "training_time_seconds": exp_res["training_time_seconds"],
                "cv_metrics": cv_m,
                "test_metrics": {
                    "accuracy": test_m["accuracy"],
                    "balanced_accuracy": test_m["balanced_accuracy"],
                    "f1_macro": test_m["f1_macro"],
                    "f1_weighted": test_m["f1_weighted"],
                    "roc_auc_macro": test_m["roc_auc_macro"],
                    "roc_auc_weighted": test_m["roc_auc_weighted"],
                },
                "artifact_path": exp_res["artifact_path"],
            }

            print(f"  -> CV Macro F1: {cv_m['cv_macro_f1_mean']:.4f} (±{cv_m['cv_macro_f1_std']:.4f}) | Test Acc: {test_m['accuracy']:.4f} | Test Macro F1: {test_m['f1_macro']:.4f}")

        except Exception as e:
            end_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"Experiment {exp_id} failed: {e}", exc_info=True)
            training_log_records.append({
                "experiment_id": exp_id,
                "start_time": start_ts,
                "end_time": end_ts,
                "duration": 0.0,
                "model": model_name,
                "feature_set": feat_set,
                "class_weight": class_wt if class_wt is not None else "None",
                "status": "FAILED",
                "error_message": str(e),
            })

    # ==============================================================
    # 4. Generate Synthesis Reports
    # ==============================================================

    # A. Training Log
    log_df = pd.DataFrame(training_log_records)
    log_path = REPORTS_DIR / "training_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nSaved training log: {log_path}")

    # B. Model Comparison
    comp_df = pd.DataFrame(comparison_records)
    comp_path = REPORTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved model comparison table: {comp_path}")

    # C. Probability Diagnostics Summary
    if diagnostics_records:
        diag_df = pd.DataFrame(diagnostics_records)
        diag_path = PROBABILITY_DIAGNOSTICS_DIR / "probability_diagnostics_summary.csv"
        diag_df.to_csv(diag_path, index=False)
        print(f"Saved probability diagnostics summary: {diag_path}")

    # D. Overfitting Diagnostics
    overfit_records = []
    for r in comparison_records:
        cv_f1 = r["cv_macro_f1_mean"]
        test_f1 = r["test_macro_f1"]
        diff = round(cv_f1 - test_f1, 4)
        # Threshold: diff > 0.05 indicates substantial drop in generalization
        flag = "OVERFITTING_WARNING" if diff > 0.05 else ("UNDERFITTING_WARNING" if diff < -0.05 else "NORMAL")
        overfit_records.append({
            "experiment_id": r["experiment_id"],
            "model": r["model"],
            "feature_set": r["feature_set"],
            "class_weight": r["class_weight"],
            "cv_macro_f1": cv_f1,
            "test_macro_f1": test_f1,
            "difference": diff,
            "overfitting_flag": flag,
        })
    overfit_df = pd.DataFrame(overfit_records)
    overfit_path = REPORTS_DIR / "overfitting_diagnostics.csv"
    overfit_df.to_csv(overfit_path, index=False)
    print(f"Saved overfitting diagnostics: {overfit_path}")

    # E. Geographic Signal Comparison
    geo_comp_records = []
    # Match full vs blind for each model & class_weight
    models = sorted(list(set(r["model"] for r in comparison_records)))
    for m in models:
        for cw in ["None", "balanced", "NOT_SUPPORTED"]:
            full_r = next((r for r in comparison_records if r["model"] == m and r["feature_set"] == "full" and r["class_weight"] == cw), None)
            blind_r = next((r for r in comparison_records if r["model"] == m and r["feature_set"] == "geographic_blind" and r["class_weight"] == cw), None)
            if full_r and blind_r:
                geo_comp_records.append({
                    "model": m,
                    "class_weight": cw,
                    "full_cv_macro_f1": full_r["cv_macro_f1_mean"],
                    "blind_cv_macro_f1": blind_r["cv_macro_f1_mean"],
                    "diff_cv_macro_f1": round(full_r["cv_macro_f1_mean"] - blind_r["cv_macro_f1_mean"], 4),
                    "full_test_macro_f1": full_r["test_macro_f1"],
                    "blind_test_macro_f1": blind_r["test_macro_f1"],
                    "diff_test_macro_f1": round(full_r["test_macro_f1"] - blind_r["test_macro_f1"], 4),
                    "full_test_accuracy": full_r["test_accuracy"],
                    "blind_test_accuracy": blind_r["test_accuracy"],
                    "diff_test_accuracy": round(full_r["test_accuracy"] - blind_r["test_accuracy"], 4),
                    "full_test_balanced_acc": full_r["test_balanced_accuracy"],
                    "blind_test_balanced_acc": blind_r["test_balanced_accuracy"],
                    "diff_test_balanced_acc": round(full_r["test_balanced_accuracy"] - blind_r["test_balanced_accuracy"], 4),
                })
    geo_df = pd.DataFrame(geo_comp_records)
    geo_path = REPORTS_DIR / "geographic_signal_comparison.csv"
    geo_df.to_csv(geo_path, index=False)
    print(f"Saved geographic signal comparison: {geo_path}")

    # F. Class Weight Comparison
    cw_comp_records = []
    # Compare None vs balanced for each model & feature_set
    for m in models:
        for fs in ["full", "geographic_blind"]:
            none_r = next((r for r in comparison_records if r["model"] == m and r["feature_set"] == fs and r["class_weight"] == "None"), None)
            bal_r = next((r for r in comparison_records if r["model"] == m and r["feature_set"] == fs and r["class_weight"] == "balanced"), None)
            if none_r and bal_r:
                # Load Zone_09 metrics for both
                pz_none_df = pd.read_csv(PER_ZONE_METRICS_DIR / f"{none_r['experiment_id']}.csv")
                pz_bal_df = pd.read_csv(PER_ZONE_METRICS_DIR / f"{bal_r['experiment_id']}.csv")

                z09_none = pz_none_df[pz_none_df["withdrawal_zone"] == "Zone_09"].iloc[0]
                z09_bal = pz_bal_df[pz_bal_df["withdrawal_zone"] == "Zone_09"].iloc[0]

                cw_comp_records.append({
                    "model": m,
                    "feature_set": fs,
                    "none_test_macro_f1": none_r["test_macro_f1"],
                    "balanced_test_macro_f1": bal_r["test_macro_f1"],
                    "diff_macro_f1": round(bal_r["test_macro_f1"] - none_r["test_macro_f1"], 4),
                    "none_test_balanced_acc": none_r["test_balanced_accuracy"],
                    "balanced_test_balanced_acc": bal_r["test_balanced_accuracy"],
                    "diff_balanced_acc": round(bal_r["test_balanced_accuracy"] - none_r["test_balanced_accuracy"], 4),
                    "none_zone09_recall": z09_none["recall"],
                    "balanced_zone09_recall": z09_bal["recall"],
                    "diff_zone09_recall": round(z09_bal["recall"] - z09_none["recall"], 4),
                    "none_zone09_f1": z09_none["f1_score"],
                    "balanced_zone09_f1": z09_bal["f1_score"],
                    "diff_zone09_f1": round(z09_bal["f1_score"] - z09_none["f1_score"], 4),
                })
    cw_df = pd.DataFrame(cw_comp_records)
    cw_path = REPORTS_DIR / "class_weight_comparison.csv"
    cw_df.to_csv(cw_path, index=False)
    print(f"Saved class weight comparison: {cw_path}")

    # G. Save Model Registry JSON
    reg_path = LOCATION_CLASSIFIER_DIR / "model_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(model_registry, f, indent=2)
    print(f"Saved model registry: {reg_path} ({len(model_registry['models'])} models registered)")

    print("\n" + "=" * 70)
    print("PHASE 5 LOCATION MODEL TRAINING & EXPERIMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_phase5()
