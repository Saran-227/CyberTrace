"""Probability distribution chart component for CyberTrace."""

from typing import List, Dict, Any, Union
import streamlit as st
import pandas as pd


def render_probability_chart(data: Union[List[Dict[str, Any]], Dict[str, float]]) -> None:
    """Render probability distribution bar chart across zones."""
    if not data:
        return

    st.markdown("#### 📊 Zone Probability Distribution")

    if isinstance(data, dict):
        chart_data = pd.DataFrame({
            "Zone": list(data.keys()),
            "Probability": [float(v) for v in data.values()],
        }).set_index("Zone")
    elif isinstance(data, list):
        chart_data = pd.DataFrame({
            "Zone": [p.get("zone", f"Zone_{i}") for i, p in enumerate(data)],
            "Probability": [float(p.get("probability", 0.0)) for p in data],
        }).set_index("Zone")
    else:
        return

    st.bar_chart(chart_data)
