"""Model evaluation module for computing comprehensive performance metrics."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from src.utils.logging import get_logger

logger = get_logger("ModelEvaluation")

def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute required multi-metric evaluation dictionary."""
    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Multiclass ROC-AUC if probabilities and classes are available
    if y_prob is not None and classes is not None and len(classes) > 1:
        try:
            # One-vs-Rest ROC-AUC calculation
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            metrics["roc_auc_weighted"] = round(float(auc), 4)
        except Exception as e:
            logger.debug(f"ROC-AUC computation skipped: {e}")
            metrics["roc_auc_weighted"] = None
    else:
        metrics["roc_auc_weighted"] = None

    logger.info(
        f"Evaluation: Acc={metrics['accuracy']}, Macro F1={metrics['f1_macro']}, "
        f"Weighted F1={metrics['f1_weighted']}"
    )
    return metrics
