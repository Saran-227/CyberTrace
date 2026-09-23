"""Inference engine for predicting cash-out withdrawal zones."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.models.location_model import LocationClassifier
from src.utils.logging import get_logger

logger = get_logger("PredictEngine")

def format_prediction_output(
    classes: np.ndarray,
    probabilities: np.ndarray,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Format raw probabilities into the standard CyberTrace output schema.

    Schema:
    {
        "predicted_zone": str,
        "confidence": float,
        "top_predictions": [
            {"zone": str, "probability": float},
            ...
        ]
    }
    """
    sorted_indices = np.argsort(probabilities)[::-1]
    best_idx = sorted_indices[0]

    predicted_zone = str(classes[best_idx])
    confidence = round(float(probabilities[best_idx]), 4)

    top_predictions: List[Dict[str, Any]] = []
    for idx in sorted_indices[:top_k]:
        top_predictions.append({
            "zone": str(classes[idx]),
            "probability": round(float(probabilities[idx]), 4),
        })

    return {
        "predicted_zone": predicted_zone,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }

def predict_withdrawal_zone(
    input_data: pd.DataFrame,
    model: Optional[LocationClassifier] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Predict likely cash-out withdrawal zone for a single complaint record."""
    if model is None or not model.is_fitted:
        raise RuntimeError(
            "Prediction model is not trained or loaded. "
            "In Phase 1, ML models must be trained via `train_candidate_models` before running inference."
        )

    probs = model.predict_proba(input_data)[0]
    return format_prediction_output(model.classes_, probs, top_k=top_k)
