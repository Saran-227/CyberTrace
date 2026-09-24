"""Training orchestrator for CyberTrace location classification models.

Executes:
- 5-fold Stratified Cross-Validation on the training partition (N = 16,000)
- End-to-end fit on full training partition
- Unseen evaluation on untouched test partition (N = 4,000)
- Metric aggregation across all 14 configured experiments
- Pipeline artifact serialization via joblib
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    LOCATION_CLASSIFIER_DIR,
    CANONICAL_COMPLAINTS_SHA256,
    EXPERIMENT_CONFIGS,
    TARGET_COLUMN,
)
from src.preprocessing.pipeline import (
    prepare_dataset,
    build_preprocessor,
    split_data,
    get_feature_names,
)
from src.models.evaluate import evaluate_classifier
from src.utils.logging import get_logger

logger = get_logger("ModelTraining")


def create_classifier(algorithm: str, hyperparameters: Dict[str, Any]):
    """Instantiate a classifier with the specified algorithm and hyperparameters."""
    params = dict(hyperparameters)
    # Remove metadata keys if present
    params.pop("class_weight_label", None)

    if algorithm == "LogisticRegression":
        return LogisticRegression(**params)
    elif algorithm == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)
    elif algorithm == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)
    elif algorithm == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")


def build_experiment_pipeline(
    feature_set: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
) -> Pipeline:
    """Construct an end-to-end sklearn Pipeline containing preprocessing and classifier."""
    preprocessor = build_preprocessor(feature_set=feature_set)
    estimator = create_classifier(algorithm, hyperparameters)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def cross_validate_pipeline(
    feature_set: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Perform 5-fold Stratified Cross-Validation on the training partition.

    Each fold re-fits the ColumnTransformer and the estimator to guarantee zero leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.sort(np.unique(y_train))

    fold_accuracies = []
    fold_balanced_accs = []
    fold_macro_precisions = []
    fold_macro_recalls = []
    fold_macro_f1s = []
    fold_weighted_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe = build_experiment_pipeline(feature_set, algorithm, hyperparameters)
        pipe.fit(X_tr, y_tr)
        y_val_pred = pipe.predict(X_val)

        metrics = evaluate_classifier(y_val, y_val_pred, classes=classes)
        fold_accuracies.append(metrics["accuracy"])
        fold_balanced_accs.append(metrics["balanced_accuracy"])
        fold_macro_precisions.append(metrics["precision_macro"])
        fold_macro_recalls.append(metrics["recall_macro"])
        fold_macro_f1s.append(metrics["f1_macro"])
        fold_weighted_f1s.append(metrics["f1_weighted"])

    return {
        "cv_accuracy_mean": round(float(np.mean(fold_accuracies)), 4),
        "cv_accuracy_std": round(float(np.std(fold_accuracies)), 4),
        "cv_balanced_accuracy_mean": round(float(np.mean(fold_balanced_accs)), 4),
        "cv_balanced_accuracy_std": round(float(np.std(fold_balanced_accs)), 4),
        "cv_macro_precision_mean": round(float(np.mean(fold_macro_precisions)), 4),
        "cv_macro_recall_mean": round(float(np.mean(fold_macro_recalls)), 4),
        "cv_macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
        "cv_macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
        "cv_weighted_f1_mean": round(float(np.mean(fold_weighted_f1s)), 4),
        "cv_weighted_f1_std": round(float(np.std(fold_weighted_f1s)), 4),
    }


def train_and_evaluate_experiment(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    complaint_ids_test: pd.Series,
    classes: np.ndarray,
    save_artifact: bool = True,
) -> Dict[str, Any]:
    """Execute complete CV, training, test evaluation, and artifact generation for one experiment."""
    exp_id = config["experiment_id"]
    model_name = config["model_name"]
    algorithm = config["algorithm"]
    feature_set = config["feature_set"]
    class_weight = config.get("class_weight", None)
    hyperparameters = config["hyperparameters"]

    logger.info(f"=== Starting Experiment: {exp_id} ({model_name}, {feature_set}, cw={class_weight}) ===")
    start_time = time.time()

    # 1. 5-Fold Stratified Cross-Validation on Training partition
    cv_metrics = cross_validate_pipeline(
        feature_set=feature_set,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        random_state=42,
    )

    # 2. Final Fit on Full Training partition
    fit_start = time.time()
    final_pipeline = build_experiment_pipeline(feature_set, algorithm, hyperparameters)
    final_pipeline.fit(X_train, y_train)
    training_time = round(time.time() - fit_start, 2)

    # 3. Predict on Untouched Test partition
    y_test_pred = final_pipeline.predict(X_test)
    y_test_prob = final_pipeline.predict_proba(X_test) if hasattr(final_pipeline, "predict_proba") else None

    # 4. Compute Test Metrics
    test_metrics = evaluate_classifier(
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        classes=classes,
    )

    # 5. Extract Feature Names from fitted preprocessor
    preprocessor = final_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    # 6. Save Pipeline Artifact
    artifact_path = None
    if save_artifact:
        artifact_path = LOCATION_CLASSIFIER_DIR / f"{exp_id}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, artifact_path)
        logger.info(f"Saved complete pipeline artifact to {artifact_path}")

    total_duration = round(time.time() - start_time, 2)

    return {
        "experiment_id": exp_id,
        "model_name": model_name,
        "algorithm": algorithm,
        "feature_set": feature_set,
        "class_weight": class_weight if class_weight is not None else "None",
        "hyperparameters": hyperparameters,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
        "complaint_ids_test": complaint_ids_test,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_time_seconds": training_time,
        "total_duration_seconds": total_duration,
        "pipeline": final_pipeline,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "status": "SUCCESS",
        "error_message": "",
    }


def train_candidate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate baseline candidate models using stratified split (backward compatible interface)."""
    from sklearn.model_selection import train_test_split
    from src.config import SUPPORTED_MODELS, LOCATION_CLASSIFIER_DIR
    from src.models.location_model import LocationClassifier

    logger.info("Executing stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: Dict[str, Any] = {
        "models": {},
        "metrics_summary": [],
        "best_model_name": None,
        "best_metric_value": -1.0,
    }

    for model_name in SUPPORTED_MODELS:
        logger.info(f"--- Training candidate model: {model_name} ---")
        classifier = LocationClassifier(model_name=model_name)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        metrics = evaluate_classifier(y_test, y_pred, y_prob, classifier.classes_)

        results["models"][model_name] = classifier
        metric_record = {"model": model_name, **metrics}
        results["metrics_summary"].append(metric_record)

        if metrics["f1_weighted"] > results["best_metric_value"]:
            results["best_metric_value"] = metrics["f1_weighted"]
            results["best_model_name"] = model_name

        if save_artifacts:
            safe_name = model_name.lower().replace(" ", "_")
            filepath = LOCATION_CLASSIFIER_DIR / f"{safe_name}_pipeline.joblib"
            classifier.save(filepath)

    return results
