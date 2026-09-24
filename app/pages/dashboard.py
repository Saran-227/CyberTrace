"""Dashboard / Home Page for CyberTrace (Phase 10).

Delivers an Apple/Haven-inspired landing experience, abstract geospatial network visualization,
live active investigation summary, system metrics, and workflow cards.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import (
    LOCATION_CLASSIFIER_DIR,
    ATM_LOCATIONS_PROCESSED_PATH,
    GENERATED_REPORTS_DIR,
)
from src.geographic.zones import get_all_zones
from src.data.loader import get_dataset_status


def render_dashboard() -> None:
    # =========================================================================
    # 1. HERO SECTION (Reference Style 1 - Haven Inspired)
    # =========================================================================
    st.markdown(
        """
        <div style="text-align: center; padding: 24px 0 16px 0; max-width: 820px; margin: 0 auto;">
            <div class="hero-badge">✦ Academic Location Intelligence Platform</div>
            <h1 class="hero-title">Cybercrime location intelligence, without the guesswork.</h1>
            <p class="hero-subtitle" style="margin: 0 auto 28px auto;">
                Analyze complaint metadata. Model likely withdrawal zones. Rank nearby ATM candidates using verified OpenStreetMap infrastructure.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA Buttons in Center
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.2, 1.5])
    with col_c2:
        if st.button("🚀 Start an Investigation", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Investigate"
            st.rerun()

    # =========================================================================
    # 2. ABSTRACT GEOSPATIAL NETWORK VISUALIZATION
    # =========================================================================
    st.markdown(
        """
        <div class="network-graphic-container">
            <svg width="680" height="150" viewBox="0 0 680 150" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width:100%; height:auto;">
                <!-- Connecting lines -->
                <path d="M120 75 C 200 40, 260 40, 340 75" stroke="#0ea5e9" stroke-width="2" stroke-dasharray="4 4" opacity="0.6"/>
                <path d="M340 75 C 420 110, 480 110, 560 75" stroke="#10b981" stroke-width="2" stroke-dasharray="4 4" opacity="0.6"/>
                <path d="M340 75 L 530 40" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.4"/>
                <path d="M340 75 L 540 110" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.4"/>

                <!-- Node 1: Incident Origin -->
                <circle cx="120" cy="75" r="24" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
                <circle cx="120" cy="75" r="10" fill="#ef4444"/>
                <text x="120" y="118" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="700" font-family="-apple-system, sans-serif">COMPLAINT ORIGIN</text>
                <text x="120" y="132" text-anchor="middle" fill="#64748b" font-size="9" font-family="-apple-system, sans-serif">Reported Incident</text>

                <!-- Node 2: Predicted Sector -->
                <rect x="300" y="45" width="80" height="60" rx="14" fill="#f0f9ff" stroke="#0284c7" stroke-width="2.5" stroke-dasharray="4 3"/>
                <text x="340" y="74" text-anchor="middle" fill="#0369a1" font-size="12" font-weight="800" font-family="-apple-system, sans-serif">PREDICTED</text>
                <text x="340" y="88" text-anchor="middle" fill="#0369a1" font-size="10" font-weight="700" font-family="-apple-system, sans-serif">ZONE</text>
                <text x="340" y="122" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="700" font-family="-apple-system, sans-serif">LOCATION MODEL</text>
                <text x="340" y="136" text-anchor="middle" fill="#64748b" font-size="9" font-family="-apple-system, sans-serif">Supervised ML Probability</text>

                <!-- Node 3: Top ATM Candidate -->
                <circle cx="560" cy="75" r="22" fill="#ecfdf5" stroke="#10b981" stroke-width="2.5"/>
                <circle cx="560" cy="75" r="9" fill="#10b981"/>
                <text x="560" y="118" text-anchor="middle" fill="#0f172a" font-size="11" font-weight="700" font-family="-apple-system, sans-serif">TOP ATM CANDIDATE</text>
                <text x="560" y="132" text-anchor="middle" fill="#64748b" font-size="9" font-family="-apple-system, sans-serif">Multi-Criteria Score</text>

                <!-- Alternative ATM Nodes -->
                <circle cx="530" cy="40" r="7" fill="#f59e0b"/>
                <circle cx="540" cy="110" r="7" fill="#f59e0b"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. CURRENT INVESTIGATION STATUS (OR INTENTIONAL EMPTY STATE)
    # =========================================================================
    has_analysis = "case_analysis" in st.session_state and st.session_state["case_analysis"]

    if has_analysis:
        analysis = st.session_state["case_analysis"]
        case = analysis.get("case_data", {})
        pred = analysis.get("prediction", {})
        ranking = analysis.get("ranking", {})
        ranked_atms = ranking.get("ranked_atms", [])
        top_atm = ranked_atms[0] if ranked_atms else {}

        conf_pct = float(pred.get("prediction_confidence", 0.0)) * 100
        tier = str(pred.get("confidence_tier", "HIGH")).upper()
        top_label = f"{top_atm.get('bank', 'N/A')} ({top_atm.get('atm_id', 'N/A')})" if top_atm else "No OSM ATMs cataloged"
        top_score = f"{top_atm.get('overall_score', 0.0):.1f} / 100" if top_atm else "N/A"

        st.markdown(
            f"""
            <div class="cyber-card" style="border-left: 5px solid #0284c7; background: #ffffff;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:12px;">
                    <div>
                        <span class="card-label">Active Investigation</span>
                        <div style="font-size:20px; font-weight:800; color:#0f172a;">Case {case.get('complaint_id')}</div>
                        <div style="font-size:12px; color:#64748b;">Reported: {case.get('city')}, {case.get('state')} | INR {case.get('amount', 0):,} ({case.get('bank')})</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="status-pill-green">Active Review</span>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top:14px; padding-top:12px; border-top:1px solid #f1f5f9;">
                    <div>
                        <div class="card-label">Predicted Sector</div>
                        <div style="font-size:16px; font-weight:800; color:#0284c7;">{pred.get('predicted_zone')}</div>
                        <div style="font-size:11px; color:#64748b;">Confidence: {conf_pct:.1f}% ({tier})</div>
                    </div>
                    <div>
                        <div class="card-label">Leading Candidate ATM</div>
                        <div style="font-size:14px; font-weight:700; color:#0f172a;">{top_atm.get('bank', 'N/A') if top_atm else 'None'}</div>
                        <div style="font-size:11px; color:#64748b;">Score: {top_score}</div>
                    </div>
                    <div>
                        <div class="card-label">Dual-Sector Search</div>
                        <div style="font-size:14px; font-weight:700; color:{'#f59e0b' if ranking.get('cross_zone_search') else '#10b981'};">
                            {'ACTIVE (Contested Boundary)' if ranking.get('cross_zone_search') else 'INACTIVE (Single Sector)'}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c_left, col_c_btn, col_c_right = st.columns([1.5, 1.2, 1.5])
        with col_c_btn:
            if st.button("🔍 Open Active Investigation Workspace", use_container_width=True):
                st.session_state["current_page"] = "Investigate"
                st.rerun()
    else:
        st.markdown(
            """
            <div class="cyber-card" style="text-align: center; padding: 32px 24px; background: #ffffff;">
                <div style="font-size: 36px; margin-bottom: 8px;">📍</div>
                <h3 style="margin: 0 0 6px 0; color: #0f172a; font-size: 18px; font-weight: 700;">Your investigation workspace is ready.</h3>
                <p style="margin: 0 auto 16px auto; font-size: 13px; color: #64748b; max-width: 520px;">
                    Enter an incident complaint in the workbench to initiate supervised location classification, probability estimation, and spatial ATM discovery.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. SYSTEM PROVENANCE & METRICS (Reference Style 2 Stat Boxes)
    # =========================================================================
    st.markdown("### 🏛️ Verified System Infrastructure & Provenance")
    
    status = get_dataset_status()
    complaints_count = status.get("processed_complaints", {}).get("records", 20000)

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{complaints_count:,}</div>
                <div class="stat-desc">Synthetic Complaints</div>
                <small style="color:#64748b; font-size:10px;">Zero Target Leakage Enforced</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sm2:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-number">333</div>
                <div class="stat-desc">Verified OSM ATMs</div>
                <small style="color:#64748b; font-size:10px;">14 Study Cities Represented</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sm3:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{len(get_all_zones())}</div>
                <div class="stat-desc">Operational Sectors</div>
                <small style="color:#64748b; font-size:10px;">Punjab / North India Focus</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sm4:
        trained_models_count = len(list(LOCATION_CLASSIFIER_DIR.glob("*.joblib")))
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{trained_models_count}</div>
                <div class="stat-desc">Trained Classifiers</div>
                <small style="color:#10b981; font-size:10px; font-weight:700;">Random Forest & Logistic Reg</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. METHODOLOGY & WORKFLOW (Reference Style 3 Cards)
    # =========================================================================
    st.markdown("### ⚙️ How CyberTrace Works")
    st.markdown("A four-stage supervised pipeline designed to assist non-technical investigators from incident report to actionable brief.")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(
            """
            <div class="stat-box" style="height: 100%;">
                <div style="font-size: 11px; font-weight: 800; color: #0284c7; text-transform: uppercase;">Stage 01</div>
                <h4 style="margin: 6px 0; color: #0f172a; font-size: 15px;">Incident Preprocessing</h4>
                <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin: 0;">
                    Extracts monetary amounts, cyclical temporal indicators, and payment rails with strict zero target leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            """
            <div class="stat-box" style="height: 100%;">
                <div style="font-size: 11px; font-weight: 800; color: #0284c7; text-transform: uppercase;">Stage 02</div>
                <h4 style="margin: 6px 0; color: #0f172a; font-size: 15px;">Location Modeling</h4>
                <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin: 0;">
                    Supervised classifier estimates posterior probabilities across 10 withdrawal zones with margin-based confidence tiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            """
            <div class="stat-box" style="height: 100%;">
                <div style="font-size: 11px; font-weight: 800; color: #0284c7; text-transform: uppercase;">Stage 03</div>
                <h4 style="margin: 6px 0; color: #0f172a; font-size: 15px;">ATM Spatial Scoring</h4>
                <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin: 0;">
                    Discovers OpenStreetMap physical infrastructure and ranks candidates using multi-criteria proximity and bank compatibility.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w4:
        st.markdown(
            """
            <div class="stat-box" style="height: 100%;">
                <div style="font-size: 11px; font-weight: 800; color: #0284c7; text-transform: uppercase;">Stage 04</div>
                <h4 style="margin: 6px 0; color: #0f172a; font-size: 15px;">Intelligence Brief</h4>
                <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin: 0;">
                    Generates 14-section formal HTML and PDF reports with evidentiary disclaimers and recommended law enforcement actions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
