"""Dashboard page: System overview, dataset status, and model health."""

import streamlit as st
import pandas as pd
from src.config import (
    CANDIDATE_FEATURES,
    SUPPORTED_MODELS,
    LOCATION_CLASSIFIER_DIR,
    ATM_LOCATIONS_PROCESSED_PATH,
    COMPLAINTS_RAW_PATH,
    COMPLAINTS_PROCESSED_PATH,
)
from src.geographic.zones import get_all_zones
from src.data.loader import get_dataset_status, load_processed_complaints

def render_dashboard() -> None:
    st.markdown("## 📊 System Overview & Operational Dashboard")
    st.markdown(
        "CyberTrace monitors synthetic complaint streams to isolate geographic cash-out patterns "
        "and prioritize candidate ATM investigations."
    )

    # Dataset Status
    status = get_dataset_status()
    complaints_present = status["processed_complaints"]["exists"] or status["raw_complaints"]["exists"]
    complaints_count = (
        status["processed_complaints"]["records"]
        if status["processed_complaints"]["exists"]
        else status["raw_complaints"]["records"]
    )

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Synthetic Complaints",
            value=f"{complaints_count:,}" if complaints_present else "0 (Pending)",
            delta="Synthetic Dataset",
        )
    with c2:
        st.metric(label="Target Zones", value=len(get_all_zones()), delta="Punjab/North")
    with c3:
        st.metric(label="Monitored Banks", value="7 Banks", delta="Multi-rail")
    with c4:
        trained_models = list(LOCATION_CLASSIFIER_DIR.glob("*.joblib"))
        st.metric(
            label="Trained Models",
            value=len(trained_models),
            delta="Ready for Inference" if trained_models else "Phase 5 Pending",
        )

    st.markdown("---")

    # Detailed Status Blocks
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🗄️ Dataset Pipeline Status")
        dataset_table = []
        for name, info in status.items():
            dataset_table.append({
                "Dataset Layer": name.replace("_", " ").title(),
                "Status": "✅ Present" if info["exists"] else "⏳ Missing / Awaiting Phase",
                "Records": f"{info['records']:,}" if info["exists"] else "0",
            })
        st.dataframe(pd.DataFrame(dataset_table), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### 🤖 Location Classification Status")
        st.markdown("**Evaluated Model Architectures:**")
        for m in SUPPORTED_MODELS:
            safe_name = m.lower().replace(" ", "_")
            exists = (LOCATION_CLASSIFIER_DIR / f"{safe_name}_pipeline.joblib").exists()
            status_text = "🟢 Serialized & Ready" if exists else "⚪ Interface Stub (Awaiting Phase 5 Training)"
            st.markdown(f"- **{m}**: {status_text}")

        st.caption("Training can be executed from the Analytics workbench or using `python -m src.models.train`.")

    st.markdown("---")
    st.markdown("### 🔍 Recent Analysis Session")
    if "last_case" in st.session_state and st.session_state["last_case"]:
        st.success(
            f"Active Session: Case **{st.session_state['last_case'].get('complaint_id')}** "
            f"(₹{st.session_state['last_case'].get('amount', 0):,}, {st.session_state['last_case'].get('bank')}) "
            f"under active review."
        )
    else:
        st.info("No cases analyzed in this session yet. Navigate to **Investigation** to evaluate a complaint.")
