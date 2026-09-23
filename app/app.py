"""CyberTrace - Main Application Shell.

A supervised machine-learning based cybercrime cash-withdrawal location intelligence platform.
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CYBERCRIME_PORTAL_URL, FINANCIAL_FRAUD_HELPLINE
from app.pages.dashboard import render_dashboard
from app.pages.investigation import render_investigation
from app.pages.analytics import render_analytics
from app.pages.report import render_report_page

# Configure Streamlit page
st.set_page_config(
    page_title="CyberTrace | Location Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
css_file = ROOT_DIR / "app" / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Application Header
st.markdown(
    """
    <div class="platform-header">
        <div>
            <h1 class="platform-title">CYBERTRACE</h1>
            <div class="platform-subtitle">Cybercrime Cash-Withdrawal Location Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation & Official Notice
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    selected_page = st.radio(
        "Platform Modules",
        options=["Dashboard", "Investigation", "Analytics", "Reports"],
        index=1,  # Default to Investigation for immediate case review
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🚨 Emergency Redirection")
    st.markdown(
        f"""
        <div style="background:#1e1b4b; border:1px solid #4338ca; border-radius:6px; padding:12px; margin-bottom:12px;">
            <strong style="color:#a5b4fc; font-size:12px;">NATIONAL HELPLINE</strong><br>
            <span style="color:#ffffff; font-size:18px; font-weight:800;">📞 {FINANCIAL_FRAUD_HELPLINE}</span><br>
            <small style="color:#cbd5e1; font-size:11px;">Citizen Financial Cyber Fraud</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(
        label="🌐 cybercrime.gov.in",
        url=CYBERCRIME_PORTAL_URL,
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("CyberTrace Architecture v0.1.0 (Phase 1 Ready)")
    st.caption("⚠️ Academic Research Platform. All incident metadata is synthetic.")

# Page Routing
if selected_page == "Dashboard":
    render_dashboard()
elif selected_page == "Investigation":
    render_investigation()
elif selected_page == "Analytics":
    render_analytics()
elif selected_page == "Reports":
    render_report_page()
