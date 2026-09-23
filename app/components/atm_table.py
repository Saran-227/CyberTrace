"""ATM candidates ranking table component."""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd

def render_atm_table(ranked_atms: List[Dict[str, Any]]) -> None:
    """Render structured candidate ATMs with distance and candidate score."""
    st.markdown("### 🏧 Candidate ATM Rankings")

    if not ranked_atms:
        st.info("No candidate ATMs identified yet for the selected operational zone.")
        return

    st.caption("⚠️ **Notice:** Candidate ranking is based on proximity and bank compatibility. It does not establish that an ATM was confirmed in actual withdrawal.")

    display_data = []
    for atm in ranked_atms:
        display_data.append({
            "Rank / Status": atm.get("designation", "Candidate"),
            "ATM ID": atm.get("atm_id"),
            "Bank / Operator": atm.get("bank"),
            "Distance (km)": f"{atm.get('distance_km', 0.0):.2f}",
            "Candidate Score": f"{atm.get('candidate_score', 0.0):.2f}",
            "Source": atm.get("source", "OpenStreetMap"),
        })

    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
