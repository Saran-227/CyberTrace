"""Report summary card component."""

from typing import Dict, Any
import streamlit as st

def render_report_card(explanation: Dict[str, Any]) -> None:
    """Render executive intelligence highlights and recommended investigative actions."""
    st.markdown("### 📋 Investigative Highlights")

    st.info(explanation.get("zone_summary", ""))

    with st.expander("Key Contributing Signals", expanded=True):
        for sig in explanation.get("key_signals", []):
            st.markdown(f"- {sig}")

    with st.expander("Recommended Investigative Actions", expanded=True):
        for act in explanation.get("recommended_focus", []):
            st.markdown(f"- {act}")

    st.caption(f"🛡️ {explanation.get('evidence_disclaimer', '')}")
