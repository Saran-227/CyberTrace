"""Investigation Page: Case information, zone prediction, Leaflet map, and ATM candidates."""

from datetime import datetime
import streamlit as st
import pandas as pd

from src.config import (
    CYBERCRIME_PORTAL_URL,
    FINANCIAL_FRAUD_HELPLINE,
    LOCATION_CLASSIFIER_DIR,
)
from src.geographic.zones import get_all_zones, get_zone_bounding_box, get_zone_centroid
from src.models.location_model import LocationClassifier
from src.models.predict import predict_withdrawal_zone
from src.atm.atm_discovery import discover_candidate_atms
from src.atm.ranking import rank_atm_candidates
from src.intelligence.report import generate_executive_report
from src.app_helpers import get_or_create_session_state

from app.components.map import render_leaflet_map
from app.components.prediction_card import render_prediction_card
from app.components.atm_table import render_atm_table
from app.components.probability_chart import render_probability_chart

def render_investigation() -> None:
    # 1. Official Reporting Banner (National Cyber Crime Reporting Portal & Helpline 1930)
    st.markdown(
        f"""
        <div class="official-reporting-banner">
            <div class="banner-left">
                <span class="banner-title">🛡️ OFFICIAL GOVERNMENT REPORTING DIRECTIVE</span>
                <span class="banner-desc">
                    CyberTrace is an intelligence analysis tool and does NOT register official police complaints.
                    Victims must report incidents directly to law enforcement authorities.
                </span>
            </div>
            <div>
                <a href="{CYBERCRIME_PORTAL_URL}" target="_blank" style="text-decoration:none;">
                    <button style="background:#dc2626; color:#ffffff; border:none; padding:10px 16px; border-radius:6px; font-weight:700; cursor:pointer;">
                        🚨 Report Cybercrime (cybercrime.gov.in)
                    </button>
                </a>
                <span class="helpline-pill" style="margin-left: 8px;">
                    📞 Helpline: {FINANCIAL_FRAUD_HELPLINE}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🔎 Case Investigation Workbench")
    st.caption("Enter case metadata to analyze cash-out probabilities and identify candidate ATM points of interest.")

    # Two column layout: Case Input Form (Left) and Intelligence Display (Right)
    col_form, col_intel = st.columns([1.1, 1.9])

    with col_form:
        st.markdown("### 📝 Case Information")
        with st.form("case_input_form"):
            complaint_id = st.text_input("Complaint Reference ID", value="CYBER-2026-0814")

            c_amt, c_bank = st.columns(2)
            with c_amt:
                amount = st.number_input("Amount (₹)", min_value=500, max_value=5000000, value=42500, step=1000)
            with c_bank:
                bank = st.selectbox(
                    "Victim Bank",
                    ["HDFC", "SBI", "ICICI", "Axis Bank", "Punjab National Bank", "Bank of Baroda", "Kotak Mahindra"],
                    index=0,
                )

            c_loc, c_tx = st.columns(2)
            with c_loc:
                city = st.selectbox("Reported City", ["Jalandhar", "Ludhiana", "Amritsar", "Patiala", "Chandigarh", "Mohali"])
            with c_tx:
                transaction_type = st.selectbox("Transaction Type", ["UPI", "IMPS", "NEFT", "Net Banking", "ATM Withdrawal"])

            fraud_type = st.selectbox(
                "Reported Fraud Modus Operandi",
                [
                    "Phishing/Smishing",
                    "Lottery/Task Scam",
                    "Identity Theft",
                    "Impersonation",
                    "Investment Scam",
                    "Customer Support Fraud",
                ],
            )

            c_date, c_time = st.columns(2)
            with c_date:
                inc_date = st.date_input("Incident Date", value=datetime.today())
            with c_time:
                inc_time = st.time_input("Incident Time", value=datetime.strptime("23:15", "%H:%M").time())

            # Default coordinates lookup based on city
            city_coords = {
                "Jalandhar": (31.3260, 75.5762, "Jalandhar", "Punjab"),
                "Ludhiana": (30.9010, 75.8573, "Ludhiana", "Punjab"),
                "Amritsar": (31.6340, 74.8723, "Amritsar", "Punjab"),
                "Patiala": (30.3398, 76.3869, "Patiala", "Punjab"),
                "Chandigarh": (30.7333, 76.7794, "Chandigarh", "Chandigarh"),
                "Mohali": (30.7046, 76.7179, "SAS Nagar", "Punjab"),
            }
            default_lat, default_lon, district, state = city_coords[city]

            with st.expander("Geographic Coordinates (Optional Adjustment)"):
                complaint_lat = st.number_input("Latitude", value=float(default_lat), format="%.5f")
                complaint_lon = st.number_input("Longitude", value=float(default_lon), format="%.5f")

            analyze_submitted = st.form_submit_button("⚡ Analyze Case", use_container_width=True)

        if analyze_submitted:
            # Prepare case dictionary
            hour = inc_time.hour
            dow = inc_date.weekday()
            is_weekend = int(dow >= 5)
            is_night = int(hour >= 22 or hour <= 5)
            amount_category = "High (25k-50k)" if amount >= 25000 and amount < 50000 else "Medium (5k-25k)"

            case_data = {
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
                "hour": hour,
                "day_of_week": dow,
                "day_name": inc_date.strftime("%A"),
                "is_weekend": is_weekend,
                "is_night": is_night,
                "amount_category": amount_category,
            }
            st.session_state["active_case"] = case_data
            st.session_state["last_case"] = case_data

            # Check if a trained location model pipeline exists on disk
            candidate_model_files = list(LOCATION_CLASSIFIER_DIR.glob("*.joblib"))
            if candidate_model_files:
                try:
                    classifier = LocationClassifier.load(candidate_model_files[0])
                    input_df = pd.DataFrame([case_data])
                    pred_res = predict_withdrawal_zone(input_df, model=classifier)
                    st.session_state["prediction_result"] = pred_res
                except Exception as exc:
                    st.warning(f"Model inference failed: {exc}. Using designated operational zone stub.")
                    st.session_state["prediction_result"] = None
            else:
                # Clean interface notice: model not trained yet (Phase 5)
                # For Phase 1 demonstration of map & ranking architecture, associate to corresponding operational zone
                zone_stub = "Zone_04" if city == "Ludhiana" else "Zone_01"
                st.session_state["prediction_result"] = {
                    "predicted_zone": zone_stub,
                    "confidence": 0.814,
                    "top_predictions": [
                        {"zone": zone_stub, "probability": 0.814},
                        {"zone": "Zone_02" if zone_stub == "Zone_01" else "Zone_03", "probability": 0.102},
                        {"zone": "Zone_05", "probability": 0.051},
                    ],
                    "is_phase1_stub": True,
                }

            # Run ATM Candidate Discovery & Ranking
            active_zone = st.session_state["prediction_result"]["predicted_zone"]
            discovered_atms = discover_candidate_atms(
                predicted_zone=active_zone,
                complaint_lat=complaint_lat,
                complaint_lon=complaint_lon,
                use_synthetic_fallback=True,
            )
            ranked = rank_atm_candidates(
                candidates=discovered_atms,
                complaint_bank=bank,
                complaint_hour=hour,
                top_n=10,
            )
            st.session_state["ranked_atms"] = ranked

    # Display Right: Intelligence & Geographic Visualization
    with col_intel:
        if "active_case" in st.session_state and st.session_state["active_case"]:
            case = st.session_state["active_case"]
            pred = st.session_state.get("prediction_result")
            ranked = st.session_state.get("ranked_atms", [])
            zone_id = pred.get("predicted_zone", "Zone_01") if pred else "Zone_01"

            if pred and pred.get("is_phase1_stub"):
                st.info(
                    "ℹ️ **Phase 1 Architecture Mode:** Location classifier estimators have not been trained yet "
                    "(Phase 5). Showing calibrated test zone to verify map, ranking, and reporting interfaces."
                )

            # Map Visualization
            st.markdown("### 🗺️ Operational Geospatial Intelligence Map")
            c_lat = case.get("complaint_latitude", 31.3260)
            c_lon = case.get("complaint_longitude", 75.5762)
            bbox = get_zone_bounding_box(zone_id)

            render_leaflet_map(
                center_lat=c_lat,
                center_lon=c_lon,
                zoom=12,
                complaint_point=(c_lat, c_lon, f"Complaint {case.get('complaint_id')} (₹{case.get('amount'):,})"),
                predicted_zone_bbox=bbox,
                candidate_atms=ranked,
                height=480,
            )

            # Prediction Card and Top Alternatives
            c_p1, c_p2 = st.columns([1, 1])
            with c_p1:
                render_prediction_card(pred)
            with c_p2:
                if pred and "top_predictions" in pred:
                    render_probability_chart(pred["top_predictions"])

            # ATM Candidates Table
            render_atm_table(ranked)

            # Report Generator Trigger
            st.markdown("---")
            if st.button("📄 Generate Executive Intelligence Report", use_container_width=True):
                report_path = generate_executive_report(case, pred, ranked)
                st.success(f"Report generated: `{report_path.name}`. Access it under the **Reports** page.")
        else:
            st.info("👈 Enter complaint metadata and click **Analyze Case** to initiate intelligence triage.")
