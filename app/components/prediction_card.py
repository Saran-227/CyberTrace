"""Prediction display card component."""

from typing import Dict, Any, Optional
import streamlit as st

def render_prediction_card(prediction_result: Optional[Dict[str, Any]] = None) -> None:
    """Render structured model prediction card with zone and confidence."""
    st.markdown("### 🎯 Model Prediction")

    if not prediction_result:
        st.info("No prediction generated. Please provide case details and click **Analyze Case**.")
        return

    predicted_zone = prediction_result.get("predicted_zone", "N/A")
    confidence = prediction_result.get("confidence", 0.0)
    top_preds = prediction_result.get("top_predictions", [])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(label="Predicted Withdrawal Zone", value=predicted_zone)
    with col2:
        st.metric(label="Model Confidence", value=f"{confidence * 100:.1f}%")

    if len(top_preds) > 1:
        alt = top_preds[1]
        st.caption(f"**Top Alternative:** {alt['zone']} ({alt['probability'] * 100:.1f}%)")
