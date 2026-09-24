"""Probability distribution horizontal chart component for CyberTrace (Phase 10).

Renders clean horizontal probability bars using Altair, sorted descending,
with primary predicted sector highlighted in cyan/blue and secondary sectors in amber/slate.
"""

from typing import List, Dict, Any, Union, Optional
import streamlit as st
import pandas as pd
import altair as alt


def render_probability_chart(
    data: Union[List[Dict[str, Any]], Dict[str, float]],
    predicted_zone: Optional[str] = None,
    second_zone: Optional[str] = None,
) -> None:
    """Render a clean horizontal probability distribution bar chart across zones."""
    if not data:
        return

    st.markdown("#### 📊 Sector Probability Distribution")

    rows = []
    if isinstance(data, dict):
        for z, p in data.items():
            prob_pct = float(p) * 100 if float(p) <= 1.0 else float(p)
            rows.append({"Zone": z, "Probability": round(prob_pct, 1)})
    elif isinstance(data, list):
        for item in data:
            z = item.get("zone", "Zone_01")
            p = float(item.get("probability", 0.0))
            prob_pct = p * 100 if p <= 1.0 else p
            rows.append({"Zone": z, "Probability": round(prob_pct, 1)})
    else:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Assign category labels for semantic coloring
    if not predicted_zone:
        predicted_zone = df.sort_values("Probability", ascending=False).iloc[0]["Zone"]

    def get_status(z: str) -> str:
        if z == predicted_zone:
            return "Primary Predicted"
        elif second_zone and z == second_zone:
            return "Secondary Contender"
        return "Alternative"

    df["Status"] = df["Zone"].apply(get_status)

    domain = ["Primary Predicted", "Secondary Contender", "Alternative"]
    range_colors = ["#0284c7", "#f59e0b", "#94a3b8"]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            x=alt.X("Probability:Q", title="Posterior Likelihood (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Zone:N", sort="-x", title=""),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=alt.Legend(title="Candidate Role", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Zone:N", title="Operational Zone"),
                alt.Tooltip("Probability:Q", title="Probability (%)", format=".1f"),
                alt.Tooltip("Status:N", title="Sector Designation"),
            ],
        )
        .properties(height=240)
    )

    st.altair_chart(chart, use_container_width=True)
