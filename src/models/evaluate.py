"""Model evaluation module for computing comprehensive performance metrics.

Provides functions for:
- Standard multiclass metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1)
- Multiclass One-vs-Rest ROC-AUC (Macro & Weighted)
- Per-class metrics (Precision, Recall, F1, Support)
- Confusion matrix generation and visualization saving
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from src.utils.logging import get_logger

logger = get_logger("ModelEvaluation")


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multi-metric evaluation dictionary.

    Parameters:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        y_prob: Optional predicted class probability matrix (N, n_classes).
        classes: Optional array of unique class labels in canonical order.

    Returns:
        Dictionary containing overall and per-class performance metrics.
    """
    if classes is None:
        classes = np.sort(np.unique(np.concatenate([y_true, y_pred])))

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }

    # Per-class metrics breakdown
    report_dict = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class = {}
    for cls in classes:
        cls_str = str(cls)
        if cls_str in report_dict:
            per_class[cls_str] = {
                "precision": round(float(report_dict[cls_str]["precision"]), 4),
                "recall": round(float(report_dict[cls_str]["recall"]), 4),
                "f1_score": round(float(report_dict[cls_str]["f1-score"]), 4),
                "support": int(report_dict[cls_str]["support"]),
            }
        else:
            per_class[cls_str] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}
    metrics["per_class"] = per_class

    # Multiclass ROC-AUC if probabilities and classes are available
    if y_prob is not None and classes is not None and len(classes) > 1:
        try:
            # Macro One-vs-Rest ROC-AUC
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=classes)
            metrics["roc_auc_macro"] = round(float(auc_macro), 4)
        except Exception as e:
            logger.debug(f"ROC-AUC macro computation skipped: {e}")
            metrics["roc_auc_macro"] = None

        try:
            # Weighted One-vs-Rest ROC-AUC
            auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=classes)
            metrics["roc_auc_weighted"] = round(float(auc_weighted), 4)
        except Exception as e:
            logger.debug(f"ROC-AUC weighted computation skipped: {e}")
            metrics["roc_auc_weighted"] = None
    else:
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_weighted"] = None

    logger.info(
        f"Evaluation: Acc={metrics['accuracy']}, BalAcc={metrics['balanced_accuracy']}, "
        f"Macro F1={metrics['f1_macro']}, Weighted F1={metrics['f1_weighted']}"
    )
    return metrics


def save_confusion_matrix_plot(
    cm: np.ndarray,
    classes: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save dual (raw counts + normalized) confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw count matrix
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title(f"{title} (Raw Counts)", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes, fontsize=9)
    axes[0].set_xlabel("Predicted Zone", fontsize=10)
    axes[0].set_ylabel("True Zone", fontsize=10)

    thresh0 = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh0 else "black",
                fontsize=8,
            )

    # Normalized matrix
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    axes[1].set_title(f"{title} (Normalized Recall)", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(classes, fontsize=9)
    axes[1].set_xlabel("Predicted Zone", fontsize=10)
    axes[1].set_ylabel("True Zone", fontsize=10)

    thresh1 = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh1 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix visualization to: {output_path}")
