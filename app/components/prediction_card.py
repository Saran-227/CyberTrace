"""Prediction display card component for CyberTrace Phase 7."""

from typing import Dict, Any, Optional
import streamlit as st


def render_prediction_card(prediction_result: Optional[Dict[str, Any]] = None) -> None:
    """Render structured model prediction card with predicted zone, margin, and confidence tier."""
    st.markdown("### 🎯 Location Prediction")

    if not prediction_result:
        st.info("No prediction generated. Please provide case details and click **Analyze Case**.")
        return

    predicted_zone = prediction_result.get("predicted_zone", "N/A")
    confidence = float(prediction_result.get("prediction_confidence", prediction_result.get("confidence", 0.0)))
    second_zone = prediction_result.get("second_best_zone")
    margin = float(prediction_result.get("probability_margin", 0.0))
    tier = str(prediction_result.get("confidence_tier", "HIGH")).upper()
    model_id = prediction_result.get("model_id", "random_forest_full_none")

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.metric(
            label="Predicted Withdrawal Zone",
            value=predicted_zone,
            help="Statistical geographic prediction from trained supervised ML model. Never confirmed location.",
        )
    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.1f}%",
            delta=f"Tier: {tier}",
            delta_color="normal" if tier == "HIGH" else "inverse",
        )
    with col3:
        st.metric(
            label="Margin (Δ)",
            value=f"{margin:.2f}",
            help="Difference between top and second-best zone probabilities.",
        )

    if second_zone and second_zone != predicted_zone:
        st.caption(f"**Second-Best Zone:** `{second_zone}` | **Active Model:** `{model_id}`")
    else:
        st.caption(f"**Active Model:** `{model_id}`")
