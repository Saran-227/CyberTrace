"""Spotlight Card Component for the #1 Ranked ATM Candidate (Phase 10).

Displays high-priority candidate metrics, multi-criteria score breakdown progress bars,
and non-technical operational explanations without claiming forensic confirmation.
"""

from typing import Dict, Any, Optional, Union, List
import streamlit as st


def render_atm_spotlight_card(top_atm: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> None:
    """Render a dedicated spotlight card for the highest-ranked ATM candidate (#1)."""
    if not top_atm:
        st.info("No candidate ATM available for spotlight review.")
        return

    if isinstance(top_atm, list):
        if not top_atm:
            st.info("No candidate ATM available for spotlight review.")
            return
        top_atm = top_atm[0]

    atm_id = top_atm.get("atm_id", "Unknown")
    raw_bank = str(top_atm.get("bank", "Unknown")).strip()
    bank = "Bank information unavailable" if raw_bank.lower() in ["unknown", "none", "nan", ""] else raw_bank
    city = top_atm.get("city", "Unknown")
    dist = float(top_atm.get("distance_km", 0.0))
    overall = float(top_atm.get("overall_score", 0.0))
    zone = top_atm.get("zone", "N/A")
    explanation = top_atm.get("explanation", "Candidate evaluated within target operational sector.")

    st.markdown(
        f"""
        <div class="cyber-card" style="border: 1.5px solid #10b981; background: #ffffff; padding: 20px 24px; margin-top: 14px; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <span style="background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; font-size:11px; font-weight:700; border-radius:9999px; padding:3px 10px; text-transform:uppercase; letter-spacing:0.04em;">
                        ⭐ #1 Highest-Ranked ATM Candidate
                    </span>
                    <h3 style="margin: 8px 0 2px 0; color:#0f172a; font-size:20px; font-weight:800;">{bank}</h3>
                    <div style="font-size:12px; color:#64748b;">
                        OSM Ref: <code>{atm_id}</code> | Sector: <strong>{zone}</strong> | Municipality: <strong>{city}</strong>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;">Composite Score</div>
                    <div style="font-size:26px; font-weight:800; color:#059669; line-height:1.2;">
                        {overall:.1f} <span style="font-size:13px; color:#64748b;">/ 100</span>
                    </div>
                    <div style="font-size:11.5px; color:#64748b; margin-top:2px;">
                        📍 <strong>{dist:.2f} km</strong> from origin
                    </div>
                </div>
            </div>
            
            <p style="margin: 12px 0 16px 0; font-size: 13px; color: #334155; line-height: 1.5; background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #f1f5f9;">
                💡 <strong>Why this candidate ranked highest:</strong> {explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sub-component Score Bars
    with st.expander("📊 Multi-Criteria Score Breakdown for #1 Candidate"):
        c1, c2, c3 = st.columns(3)
        with c1:
            z_score = float(top_atm.get("zone_probability_score", top_atm.get("zone_score", overall * 0.9)))
            st.metric("Zone Likelihood (30%)", f"{z_score:.1f} / 100")
            st.progress(min(max(z_score / 100.0, 0.0), 1.0))

            b_score = float(top_atm.get("bank_score", 50.0))
            st.metric("Bank Match (20%)", f"{b_score:.1f} / 100")
            st.progress(min(max(b_score / 100.0, 0.0), 1.0))

        with c2:
            s_score = float(top_atm.get("spatial_score", overall * 0.85))
            st.metric("Proximity Decay (25%)", f"{s_score:.1f} / 100")
            st.progress(min(max(s_score / 100.0, 0.0), 1.0))

            t_score = float(top_atm.get("time_score", 75.0))
            st.metric("24x7 Hours (10%)", f"{t_score:.1f} / 100")
            st.progress(min(max(t_score / 100.0, 0.0), 1.0))

        with c3:
            act_score = float(top_atm.get("activity_score", 70.0))
            st.metric("Activity Baseline (10%)", f"{act_score:.1f} / 100")
            st.progress(min(max(act_score / 100.0, 0.0), 1.0))

            amt_score = float(top_atm.get("amount_compatibility_score", top_atm.get("amount_score", 75.0)))
            st.metric("Amount Match (5%)", f"{amt_score:.1f} / 100")
            st.progress(min(max(amt_score / 100.0, 0.0), 1.0))
