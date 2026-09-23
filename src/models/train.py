"""Training orchestrator for candidate location classification models."""

from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SUPPORTED_MODELS, LOCATION_CLASSIFIER_DIR
from src.models.location_model import LocationClassifier
from src.models.evaluate import evaluate_classifier
from src.utils.logging import get_logger

logger = get_logger("ModelTraining")

def train_candidate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate all candidate classifiers using stratified split.

    Returns dictionary containing evaluation metrics, models, and best model identification.
    """
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

        # Track best model by weighted F1-score
        if metrics["f1_weighted"] > results["best_metric_value"]:
            results["best_metric_value"] = metrics["f1_weighted"]
            results["best_model_name"] = model_name

        if save_artifacts:
            safe_name = model_name.lower().replace(" ", "_")
            filepath = LOCATION_CLASSIFIER_DIR / f"{safe_name}_pipeline.joblib"
            classifier.save(filepath)

    logger.info(
        f"Model comparison complete. Best candidate: {results['best_model_name']} "
        f"with Weighted F1: {results['best_metric_value']}"
    )
    return results
