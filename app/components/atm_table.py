"""ATM candidates ranking table component for CyberTrace Phase 7."""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd


def render_atm_table(ranked_atms: List[Dict[str, Any]]) -> None:
    """Render structured candidate ATMs with multi-criteria scores, evidence flags, and disclaimers."""
    st.markdown("### 🏧 Candidate ATM Rankings")

    if not ranked_atms:
        st.info("No candidate ATMs identified yet for the selected operational zone.")
        return

    st.caption(
        "⚠️ **Investigative Notice:** Candidate rankings are probabilistic mathematical prioritizations "
        "combining zone probability, proximity, bank compatibility, and simulated operational activity. "
        "They do not establish confirmed physical withdrawal without official banking and CCTV evidence."
    )

    display_data = []
    for atm in ranked_atms:
        overall = atm.get("overall_score", round(float(atm.get("candidate_score", 0.0)) * 100, 1))
        flags = atm.get("evidence_flags", [])
        flag_str = ", ".join(flags[:3]) if flags else "STANDARD"

        display_data.append({
            "Rank": atm.get("rank", 1),
            "ATM ID": atm.get("atm_id", "Unknown"),
            "Bank / Operator": atm.get("bank", "Unknown"),
            "City": atm.get("city", "Unknown"),
            "Zone": atm.get("zone", "N/A"),
            "Distance (km)": f"{float(atm.get('distance_km', 0.0)):.1f}",
            "Score": f"{float(overall):.1f}/100",
            "Bank Match": f"{float(atm.get('bank_score', 50.0)):.0f}%",
            "Activity": f"{float(atm.get('activity_score', 50.0)):.0f}%",
            "Key Indicators": flag_str,
        })

    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed inspection expander
    with st.expander("🔍 Detailed Candidate Score Breakdown & Explanations"):
        for atm in ranked_atms[:5]:
            rank = atm.get("rank", 1)
            atm_id = atm.get("atm_id", "Unknown")
            bank = atm.get("bank", "Unknown")
            overall = atm.get("overall_score", 0.0)
            exp = atm.get("explanation", "Candidate evaluated within target zone.")

            st.markdown(f"**#{rank} {atm_id} ({bank}) — Score: `{overall:.1f}/100`**")
            st.caption(f"ℹ️ {exp}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.caption(f"Zone Prob: **{atm.get('zone_probability_score', 0.0):.1f}**")
            c2.caption(f"Spatial: **{atm.get('spatial_score', 0.0):.1f}**")
            c3.caption(f"Bank: **{atm.get('bank_score', 0.0):.1f}**")
            c4.caption(f"Time: **{atm.get('time_score', 0.0):.1f}**")
            c5.caption(f"Activity: **{atm.get('activity_score', 0.0):.1f}**")
            c6.caption(f"Amount: **{atm.get('amount_compatibility_score', 0.0):.1f}**")
            st.markdown("---")
