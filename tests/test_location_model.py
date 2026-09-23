"""Unit tests for location classifier interface and output schema contracts."""

import pytest
import numpy as np
import pandas as pd
from src.models.location_model import LocationClassifier
from src.models.predict import format_prediction_output, predict_withdrawal_zone

def test_format_prediction_output_schema():
    """Verify inference output matches required structured JSON format."""
    classes = np.array(["Zone_01", "Zone_02", "Zone_03"])
    probs = np.array([0.15, 0.75, 0.10])
    output = format_prediction_output(classes, probs, top_k=3)

    assert output["predicted_zone"] == "Zone_02"
    assert output["confidence"] == 0.75
    assert len(output["top_predictions"]) == 3
    assert output["top_predictions"][0]["zone"] == "Zone_02"
    assert output["top_predictions"][0]["probability"] == 0.75
    assert output["top_predictions"][1]["zone"] == "Zone_01"

def test_unfitted_model_raises_error():
    """Verify calling predict on unfitted model raises RuntimeError."""
    classifier = LocationClassifier(model_name="Random Forest")
    dummy_input = pd.DataFrame([{"amount": 10000, "bank": "HDFC"}])
    with pytest.raises(RuntimeError):
        predict_withdrawal_zone(dummy_input, model=classifier)
