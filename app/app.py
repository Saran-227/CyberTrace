"""CyberTrace - Main Application Shell (Phase 10).

A supervised machine-learning based cybercrime cash-withdrawal location intelligence platform.
Polished with Apple/Haven-inspired calm minimalism, floating navigation, and full dynamic state integration.
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CYBERCRIME_PORTAL_URL, FINANCIAL_FRAUD_HELPLINE
from app.components.navbar import render_navbar
from app.pages.dashboard import render_dashboard
from app.pages.investigation import render_investigation
from app.pages.analytics import render_analytics
from app.pages.report import render_report_page

# Configure Streamlit page layout
st.set_page_config(
    page_title="CyberTrace | Location Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom light-theme CSS design system
css_file = ROOT_DIR / "app" / "assets" / "css" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Floating Pill Navbar
selected_page = render_navbar()

# Sidebar: Secondary Official Information & Disclaimers
with st.sidebar:
    st.markdown("### 🏛️ Official Cybercrime Redirection")
    st.markdown(
        f"""
        <div class="helpline-box">
            <div class="helpline-title">Citizen Financial Cyber Fraud</div>
            <div class="helpline-number">📞 {FINANCIAL_FRAUD_HELPLINE}</div>
            <small style="color:#94a3b8; font-size:11px;">National Helpline 24x7</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.link_button(
        label="🌐 Report at cybercrime.gov.in",
        url=CYBERCRIME_PORTAL_URL,
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:11.5px; color:#64748b; line-height:1.5;">
            <strong>ACADEMIC PROTOTYPE NOTICE</strong><br>
            CyberTrace is an academic research platform. All complaint records and historical activity are synthetic.
            OpenStreetMap provides geographic infrastructure under ODbL.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Version 1.0 • Phase 10 Production Integration")

# Page Routing
if selected_page == "Overview":
    render_dashboard()
elif selected_page == "Investigate":
    render_investigation()
elif selected_page == "Analytics":
    render_analytics()
elif selected_page == "Reports":
    render_report_page()
else:
    render_dashboard()
