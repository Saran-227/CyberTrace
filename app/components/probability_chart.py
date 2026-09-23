"""Probability distribution chart component."""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd

def render_probability_chart(top_predictions: List[Dict[str, Any]]) -> None:
    """Render probability distribution bar chart across alternative candidate zones."""
    if not top_predictions:
        return

    st.markdown("#### 📊 Zone Probability Distribution")
    chart_data = pd.DataFrame({
        "Zone": [p["zone"] for p in top_predictions],
        "Probability": [p["probability"] for p in top_predictions],
    }).set_index("Zone")

    st.bar_chart(chart_data)
