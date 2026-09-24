"""Premium Floating Navigation Bar for CyberTrace (Phase 10).

Renders a light, modern, floating pill navigation container inspired by Reference Style 1 (Haven).
Includes live engine readiness checking, navigation pills, and official reporting redirect.
"""

from typing import Optional
from pathlib import Path
import streamlit as st

from src.config import (
    CYBERCRIME_PORTAL_URL,
    FINANCIAL_FRAUD_HELPLINE,
    LOCATION_CLASSIFIER_DIR,
    ATM_LOCATIONS_PROCESSED_PATH,
)


def check_system_readiness() -> bool:
    """Verify core model pipeline and ATM geographic datasets are ready for inference."""
    try:
        model_exists = any(LOCATION_CLASSIFIER_DIR.glob("*.joblib"))
        atm_exists = ATM_LOCATIONS_PROCESSED_PATH.exists()
        return model_exists and atm_exists
    except Exception:
        return False


def render_navbar() -> str:
    """Render the floating pill navigation bar and return the active selected page."""
    is_ready = check_system_readiness()

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Overview"

    # Floating Container Wrapper
    st.markdown(
        """
        <div class="cyber-nav-container">
            <div class="cyber-nav-brand">
                <span class="cyber-nav-icon">🛡️</span>
                <span class="cyber-nav-logo">CYBERTRACE</span>
                <span class="cyber-nav-tag">ACADEMIC INTELLIGENCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3-Column Navigation Grid
    c_brand_spacer, c_pills, c_status = st.columns([1, 2.4, 1.4], vertical_alignment="center")

    with c_brand_spacer:
        # Subtle quick-stat or breadcrumb
        has_case = "case_analysis" in st.session_state and st.session_state["case_analysis"]
        if has_case:
            case_id = st.session_state["case_analysis"].get("case_id", "ACTIVE")
            st.markdown(f'<span class="nav-active-case">Case: <strong>{case_id}</strong></span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="nav-active-case" style="color:#94a3b8;">No active case</span>', unsafe_allow_html=True)

    with c_pills:
        pages = ["Overview", "Investigate", "Analytics", "Reports"]
        current = st.session_state.get("current_page", "Overview")
        if current not in pages:
            current = "Overview"

        selected = st.pills(
            "Platform Navigation",
            options=pages,
            default=current,
            key="global_nav_pills",
            label_visibility="collapsed",
        )
        if selected and selected != st.session_state.get("current_page"):
            st.session_state["current_page"] = selected
            st.rerun()

    with c_status:
        if is_ready:
            st.markdown(
                """
                <div style="text-align: right; display: flex; justify-content: flex-end; align-items: center; gap: 8px;">
                    <span class="status-pill-green">
                        <span class="pulse-dot-green"></span>
                        System Ready
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align: right; display: flex; justify-content: flex-end; align-items: center; gap: 8px;">
                    <span class="status-pill-amber">
                        <span class="pulse-dot-amber"></span>
                        Artifacts Awaiting
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    return st.session_state.get("current_page", "Overview")
