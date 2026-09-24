"""Investigation Workbench for CyberTrace (Phase 10).

The core operational product experience:
- Clean, grouped input controls
- Real-time execution with latency measurement
- Synchronized prediction, horizontal probability distribution, and interactive Leaflet map
- Candidate spotlight card with multi-criteria progress bars
- Filterable candidate ATM table
- Non-technical operational brief
- Direct executive intelligence report generation
"""

from datetime import datetime
from typing import Optional
import time
import streamlit as st
import pandas as pd

from src.config import (
    CYBERCRIME_PORTAL_URL,
    FINANCIAL_FRAUD_HELPLINE,
    PRIMARY_MODEL_ID,
    FALLBACK_MODEL_ID,
)
from src.intelligence.case_analysis import analyze_case
from src.intelligence.report import generate_executive_report

from app.components.map import render_investigation_map, render_neutral_map
from app.components.prediction_card import render_prediction_card
from app.components.atm_table import render_atm_table
from app.components.atm_card import render_atm_spotlight_card
from app.components.probability_chart import render_probability_chart

# Complete 15 Study Cities with verified centroids
STUDY_CITIES_COORDS = {
    "Amritsar": (31.6340, 74.8723, "Amritsar", "Punjab"),
    "Jalandhar": (31.3260, 75.5762, "Jalandhar", "Punjab"),
    "Ludhiana": (30.9010, 75.8573, "Ludhiana", "Punjab"),
    "Patiala": (30.3398, 76.3869, "Patiala", "Punjab"),
    "Chandigarh": (30.7333, 76.7794, "Chandigarh", "Chandigarh"),
    "Ambala": (30.3782, 76.7767, "Ambala", "Haryana"),
    "Panipat": (29.3909, 76.9635, "Panipat", "Haryana"),
    "Ghaziabad": (28.6692, 77.4538, "Ghaziabad", "Uttar Pradesh"),
    "Meerut": (28.9845, 77.7064, "Meerut", "Uttar Pradesh"),
    "New Delhi": (28.6139, 77.2090, "New Delhi", "Delhi"),
    "Noida": (28.5355, 77.3910, "Gautam Buddha Nagar", "Uttar Pradesh"),
    "Gurugram": (28.4595, 77.0266, "Gurugram", "Haryana"),
    "Faridabad": (28.4089, 77.3178, "Faridabad", "Haryana"),
    "Alwar": (27.5530, 76.6346, "Alwar", "Rajasthan"),
    "Jaipur": (26.9124, 75.7873, "Jaipur", "Rajasthan"),
}

BANKS_LIST = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Punjab National Bank",
    "Axis Bank",
    "Bank of Baroda",
    "Kotak Mahindra Bank",
    "Canara Bank",
    "Union Bank of India",
    "IndusInd Bank",
    "unknown",
]


def render_investigation() -> None:
    # 1. Official Reporting Directive Banner
    st.markdown(
        f"""
        <div class="cyber-notice" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <strong>OFFICIAL LAW ENFORCEMENT DIRECTIVE:</strong> CyberTrace is an academic intelligence tool and does not process legal police filings.
                Official complaints must be submitted directly to <a href="{CYBERCRIME_PORTAL_URL}" target="_blank" style="color:#b91c1c; font-weight:700;">cybercrime.gov.in</a>.
            </div>
            <div>
                <span class="status-pill-amber" style="background:#fee2e2; border-color:#fecaca; color:#b91c1c;">
                    📞 National Helpline: {FINANCIAL_FRAUD_HELPLINE}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🔍 Investigation Workspace")
    st.caption("Trace a complaint. Model the withdrawal zone. Discover candidate ATM infrastructure.")

    # Two-Column Layout: Case Intake Form (Left) & Intelligence Display (Right)
    col_form, col_intel = st.columns([1.1, 1.9], gap="large")

    with col_form:
        st.markdown("### 📝 Incident Intake")
        st.caption("Start with the reported incident metadata.")

        with st.form("case_intake_form"):
            complaint_id = st.text_input("Complaint Reference ID", value="CT-2026-8812")

            c_amt, c_bank = st.columns(2)
            with c_amt:
                amount = st.number_input("Disputed Amount (₹)", min_value=100, max_value=5000000, value=35000, step=1000)
            with c_bank:
                bank = st.selectbox("Victim Bank", BANKS_LIST, index=1)

            c_loc, c_tx = st.columns(2)
            with c_loc:
                city = st.selectbox("Reported City", list(STUDY_CITIES_COORDS.keys()), index=1)
            with c_tx:
                transaction_type = st.selectbox(
                    "Payment Rail",
                    ["UPI", "IMPS", "NEFT", "Net Banking", "ATM Withdrawal", "AEPS", "CARD"],
                    index=0,
                )

            fraud_type = st.selectbox(
                "Modus Operandi",
                [
                    "OTP Fraud",
                    "Phishing/Smishing",
                    "Lottery/Task Scam",
                    "Identity Theft",
                    "Impersonation",
                    "Investment Scam",
                    "Customer Support Fraud",
                    "Job Scam",
                    "KYC Update",
                    "Loan Fraud",
                ],
                index=1,
            )

            c_date, c_time = st.columns(2)
            with c_date:
                inc_date = st.date_input("Incident Date", value=datetime.today())
            with c_time:
                inc_time = st.time_input("Incident Time", value=datetime.strptime("14:30", "%H:%M").time())

            # City default coordinates lookup
            def_lat, def_lon, district, state = STUDY_CITIES_COORDS[city]
            with st.expander("🌐 Geographic Coordinates (Optional Adjustment)"):
                complaint_lat = st.number_input("Latitude", value=float(def_lat), format="%.4f")
                complaint_lon = st.number_input("Longitude", value=float(def_lon), format="%.4f")

            selected_model = st.selectbox(
                "Inference Model",
                [PRIMARY_MODEL_ID, FALLBACK_MODEL_ID],
                format_func=lambda x: f"Primary: {x}" if "forest" in x else f"Fallback: {x}",
                index=0,
            )

            analyze_submitted = st.form_submit_button("⚡ ANALYZE CASE", type="primary", use_container_width=True)

        if analyze_submitted:
            with st.spinner("Analyzing geographic signal and ranking candidate infrastructure..."):
                t0 = time.time()
                raw_case_dict = {
                    "complaint_id": complaint_id,
                    "complaint_date": str(inc_date),
                    "complaint_time": str(inc_time),
                    "amount": float(amount),
                    "bank": bank,
                    "transaction_type": transaction_type,
                    "fraud_type": fraud_type,
                    "city": city,
                    "state": state,
                    "district": district,
                    "complaint_latitude": complaint_lat,
                    "complaint_longitude": complaint_lon,
                    "hour": inc_time.hour,
                    "day_of_week": inc_date.weekday(),
                }

                # Run live dynamic pipeline
                analysis_result = analyze_case(raw_case_dict, model_id=selected_model)

                # Update session state with dynamic result
                st.session_state["case_analysis"] = analysis_result
                st.session_state["active_case"] = analysis_result["case_data"]
                st.session_state["prediction_result"] = analysis_result["prediction"]
                st.session_state["ranked_atms"] = analysis_result["ranking"]["ranked_atms"]
                # Invalidate old report so user never downloads stale case brief
                st.session_state["report_status"] = "OUTDATED"
                st.session_state["current_report_path"] = None

    # Right Column: Intelligence & Geospatial Visualization
    with col_intel:
        if "case_analysis" in st.session_state and st.session_state["case_analysis"]:
            analysis = st.session_state["case_analysis"]
            case = analysis["case_data"]
            pred = analysis["prediction"]
            ranking = analysis["ranking"]
            raw_ranked = ranking["ranked_atms"]
            expl = analysis["explanation"]

            # 1. Operational Status Bar
            cross_zone_active = ranking.get("cross_zone_search", False)
            cross_zone_str = "ACTIVE (Dual-Sector Search)" if cross_zone_active else "INACTIVE (Single Sector)"
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:10px 16px; margin-bottom:14px; font-size:12.5px; color:#334155; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
                    <div>⚙️ <strong>Model:</strong> <code>{analysis.get('model_id')}</code></div>
                    <div>⚡ <strong>Latency:</strong> <strong>{analysis.get('execution_time_ms', 0):.1f} ms</strong></div>
                    <div>🔄 <strong>Cross-Zone Search:</strong> <span style="color:{'#d97706' if cross_zone_active else '#16a34a'}; font-weight:700;">{cross_zone_str}</span></div>
                    <div>🏧 <strong>Evaluated ATMs:</strong> {ranking.get('total_atms_evaluated', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Prediction Card and Horizontal Probability Chart
            c_p1, c_p2 = st.columns([1.1, 0.9])
            with c_p1:
                render_prediction_card(pred)
            with c_p2:
                if "zone_probabilities" in pred:
                    render_probability_chart(
                        pred["zone_probabilities"],
                        predicted_zone=pred.get("predicted_zone"),
                        second_zone=pred.get("second_best_zone"),
                    )

            # 3. Interactive Map Filter Controls
            st.markdown("### 🗺️ Geospatial Intelligence Map")
            fc1, fc2, fc3, fc4 = st.columns([1, 1.2, 1.2, 1.4])
            with fc1:
                cand_count = st.selectbox("Display Count", [5, 10, 25], index=1)
            with fc2:
                unique_banks = ["All"] + sorted(list({a.get("bank", "Unknown") for a in raw_ranked if a.get("bank")}))
                bank_filter = st.selectbox("Filter Bank", unique_banks, index=0)
            with fc3:
                unique_zones = ["All"] + sorted(list({a.get("zone", "N/A") for a in raw_ranked if a.get("zone")}))
                zone_filter = st.selectbox("Filter Zone", unique_zones, index=0)
            with fc4:
                atm_choices = ["None (Overview)"] + [f"#{a.get('rank', i+1)}: {a.get('atm_id')} ({a.get('bank')})" for i, a in enumerate(raw_ranked[:cand_count])]
                selected_choice = st.selectbox("Focus ATM", atm_choices, index=0)
                selected_atm_id: Optional[str] = None
                if selected_choice != "None (Overview)":
                    selected_atm_id = selected_choice.split(":")[1].split("(")[0].strip()

            # 4. Render Dynamic Leaflet Map
            render_investigation_map(
                case_analysis=analysis,
                visible_candidate_count=cand_count,
                bank_filter=bank_filter if bank_filter != "All" else None,
                zone_filter=zone_filter if zone_filter != "All" else None,
                selected_atm_id=selected_atm_id,
                height=520,
            )

            # 5. Top Candidate Spotlight Card (#1 Candidate)
            if raw_ranked:
                render_atm_spotlight_card(raw_ranked[0])

            # 6. Filter and Render Ranked ATM Candidates Table
            filtered_ranked = raw_ranked
            if bank_filter != "All":
                filtered_ranked = [a for a in filtered_ranked if bank_filter.lower() in a.get("bank", "").lower() or bank_filter.lower() in a.get("operator", "").lower()]
            if zone_filter != "All":
                filtered_ranked = [a for a in filtered_ranked if a.get("zone") == zone_filter]
            filtered_ranked = filtered_ranked[:cand_count]

            render_atm_table(filtered_ranked)

            # 7. Plain-Language Operational Brief
            with st.expander("📋 Non-Technical Operational Intelligence Brief", expanded=True):
                st.markdown(f"**Zone Assessment:** {expl.get('zone_summary')}")
                st.markdown(f"**ATM Assessment:** {expl.get('atm_summary')}")
                st.markdown("**Key Operational Signals:**")
                for sig in expl.get("key_signals", []):
                    st.markdown(f"- {sig}")
                st.markdown("**Recommended Investigative Actions:**")
                for act in expl.get("recommended_focus", []):
                    st.markdown(f"1. {act}")
                st.caption(f"🛡️ *Disclaimer:* {expl.get('evidence_disclaimer')}")

            # 8. Report Generator Trigger
            st.markdown("---")
            if st.button("📄 Generate Executive Intelligence Report (HTML & PDF)", type="primary", use_container_width=True):
                with st.spinner("Compiling official executive intelligence brief (HTML & PDF)..."):
                    report_path = generate_executive_report(analysis)
                    st.session_state["current_report_path"] = report_path
                    st.session_state["report_status"] = "CURRENT"
                st.success(f"✅ Intelligence Brief successfully compiled: `{report_path.name}`. Access HTML preview and PDF download in the **Reports** section.")

        else:
            # Clean Empty State: Neutral OpenStreetMap View of Target Region
            st.markdown(
                """
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px 24px; margin-bottom:14px;">
                    <div style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:4px;">Ready for Investigation</div>
                    <div style="font-size:13px; color:#64748b;">
                        Enter the incident details on the left and click <strong>ANALYZE CASE</strong> to initiate supervised location classification and spatial ATM discovery.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_neutral_map(height=520)
