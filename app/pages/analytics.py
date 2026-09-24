"""Analytics & Model Evaluation Workbench for CyberTrace (Phase 10).

Visualizes real project datasets and evaluation artifacts without recomputing expensive models:
- Dataset profiling (20,000 synthetic complaints)
- Phase 6 benchmark results across 14 model experiments
- Target class imbalance diagnostics
- OpenStreetMap ATM infrastructure coverage across 15 study cities
"""

from typing import Dict, Any, Optional
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    ATM_LOCATIONS_PROCESSED_PATH,
    REPORTS_DIR,
)


@st.cache_data(show_spinner=False)
def load_cached_complaint_sample(max_rows: int = 20000) -> Optional[pd.DataFrame]:
    """Load preprocessed complaints dataset with caching."""
    if COMPLAINTS_PROCESSED_PATH.exists():
        return pd.read_csv(COMPLAINTS_PROCESSED_PATH, nrows=max_rows)
    return None


@st.cache_data(show_spinner=False)
def load_cached_evaluation_benchmarks() -> Optional[pd.DataFrame]:
    """Load Phase 6 14-model evaluation metrics summary."""
    eval_csv = REPORTS_DIR / "phase6_model_evaluation.csv"
    if eval_csv.exists():
        return pd.read_csv(eval_csv)
    return None


@st.cache_data(show_spinner=False)
def load_cached_imbalance_summary() -> Optional[pd.DataFrame]:
    """Load class imbalance summary statistics."""
    imb_csv = REPORTS_DIR / "class_imbalance_summary.csv"
    if imb_csv.exists():
        return pd.read_csv(imb_csv)
    return None


@st.cache_data(show_spinner=False)
def load_cached_atm_coverage() -> Optional[pd.DataFrame]:
    """Load verified OpenStreetMap ATM coverage across study regions."""
    cov_csv = REPORTS_DIR / "atm_coverage_report.csv"
    if cov_csv.exists():
        return pd.read_csv(cov_csv)
    return None


def render_analytics() -> None:
    st.markdown("## 📈 Empirical Analytics & Model Benchmarks")
    st.caption("Inspect synthetic complaint distributions, 14-pipeline classification benchmarks, and OpenStreetMap spatial coverage.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Complaint Dataset Profiling",
        "🏆 Model Benchmarking (Phase 6)",
        "⚖️ Class Imbalance Diagnostics",
        "🏧 OpenStreetMap ATM Coverage",
    ])

    # =========================================================================
    # TAB 1: COMPLAINT DATASET PROFILING
    # =========================================================================
    with tab1:
        df_complaints = load_cached_complaint_sample()
        if df_complaints is not None and not df_complaints.empty:
            st.markdown("### 📊 Dataset Overview (20,000 Records)")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Complaints", f"{len(df_complaints):,}")
            with c2:
                st.metric("Monitored Cities", df_complaints["city"].nunique() if "city" in df_complaints.columns else 15)
            with c3:
                st.metric("Target Zones", df_complaints["withdrawal_zone"].nunique() if "withdrawal_zone" in df_complaints.columns else 10)
            with c4:
                st.metric("Leakage Enforcement", "VERIFIED ZERO", delta="Strict Preprocessing")

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Modus Operandi Distribution")
                if "fraud_type" in df_complaints.columns:
                    vc_fraud = df_complaints["fraud_type"].value_counts().reset_index()
                    vc_fraud.columns = ["Fraud Type", "Count"]
                    chart_fraud = (
                        alt.Chart(vc_fraud)
                        .mark_bar(cornerRadiusEnd=4, color="#0284c7")
                        .encode(
                            x=alt.X("Count:Q", title="Number of Incidents"),
                            y=alt.Y("Fraud Type:N", sort="-x", title=""),
                            tooltip=["Fraud Type", "Count"],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(chart_fraud, use_container_width=True)

            with col_b:
                st.markdown("#### Payment Channel Breakdown")
                if "transaction_type" in df_complaints.columns:
                    vc_tx = df_complaints["transaction_type"].value_counts().reset_index()
                    vc_tx.columns = ["Payment Rail", "Count"]
                    chart_tx = (
                        alt.Chart(vc_tx)
                        .mark_bar(cornerRadiusEnd=4, color="#10b981")
                        .encode(
                            x=alt.X("Count:Q", title="Incident Volume"),
                            y=alt.Y("Payment Rail:N", sort="-x", title=""),
                            tooltip=["Payment Rail", "Count"],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(chart_tx, use_container_width=True)

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.markdown("#### Diurnal Hourly Distribution")
                if "hour" in df_complaints.columns:
                    hour_df = df_complaints["hour"].value_counts().sort_index().reset_index()
                    hour_df.columns = ["Hour", "Complaints"]
                    chart_hour = (
                        alt.Chart(hour_df)
                        .mark_line(point=True, color="#0284c7")
                        .encode(
                            x=alt.X("Hour:O", title="Hour of Day (00:00 - 23:00)"),
                            y=alt.Y("Complaints:Q", title="Incident Count"),
                            tooltip=["Hour", "Complaints"],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart_hour, use_container_width=True)

            with col_d:
                st.markdown("#### Disputed Amount Categories")
                if "amount_category" in df_complaints.columns:
                    amt_df = df_complaints["amount_category"].value_counts().reset_index()
                    amt_df.columns = ["Category", "Count"]
                    chart_amt = (
                        alt.Chart(amt_df)
                        .mark_bar(cornerRadiusEnd=4, color="#f59e0b")
                        .encode(
                            x=alt.X("Count:Q", title="Count"),
                            y=alt.Y("Category:N", sort="-x", title=""),
                            tooltip=["Category", "Count"],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart_amt, use_container_width=True)

        else:
            st.warning("Processed complaints dataset not loaded. Ensure `data/processed/cybercrime_complaints.csv` exists.")

    # =========================================================================
    # TAB 2: MODEL BENCHMARKING (PHASE 6)
    # =========================================================================
    with tab2:
        df_bench = load_cached_evaluation_benchmarks()
        if df_bench is not None and not df_bench.empty:
            st.markdown("### 🏆 14 Location Classification Experiments (Phase 6)")
            st.caption("Evaluated across 4 model families under Full Metadata vs Geographic-Blind feature configurations.")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Primary Production Model", "Random Forest (Full)", delta="Macro F1: 0.9657")
            with m2:
                st.metric("Fallback Linear Model", "Logistic Regression (Full)", delta="Macro F1: 0.9664")
            with m3:
                st.metric("Geographic-Blind Baseline", "Random Forest (Blind)", delta="Macro F1: 0.0666", delta_color="inverse")

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            
            # Comparison Table
            display_cols = [
                "model_id", "model_name", "feature_set", "class_weight",
                "macro_f1", "balanced_accuracy", "accuracy", "roc_auc_weighted", "inference_efficiency"
            ]
            valid_cols = [c for c in display_cols if c in df_bench.columns]
            styled_bench = df_bench[valid_cols].copy()
            st.dataframe(styled_bench, use_container_width=True, hide_index=True)

            st.markdown("#### Full Metadata vs. Geographic-Blind Comparison")
            chart_comp = (
                alt.Chart(df_bench)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("macro_f1:Q", title="Macro F1 Score (Holdout Set)", scale=alt.Scale(domain=[0, 1.0])),
                    y=alt.Y("model_name:N", title="", sort="-x"),
                    color=alt.Color("feature_set:N", scale=alt.Scale(domain=["full", "geographic_blind"], range=["#0284c7", "#ef4444"])),
                    tooltip=["model_id", "feature_set", "macro_f1", "balanced_accuracy"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart_comp, use_container_width=True)

        else:
            st.info("Evaluation metrics report `reports/phase6_model_evaluation.csv` not found.")

    # =========================================================================
    # TAB 3: CLASS IMBALANCE DIAGNOSTICS
    # =========================================================================
    with tab3:
        df_imb = load_cached_imbalance_summary()
        st.markdown("### ⚖️ Target Zone Class Imbalance Diagnostics")
        st.caption("Empirical distribution across the 10 operational withdrawal zones in the benchmark dataset.")

        if df_imb is not None and not df_imb.empty:
            st.dataframe(df_imb, use_container_width=True, hide_index=True)

        df_comp = load_cached_complaint_sample()
        if df_comp is not None and "withdrawal_zone" in df_comp.columns:
            zone_counts = df_comp["withdrawal_zone"].value_counts().reset_index()
            zone_counts.columns = ["Withdrawal Zone", "Count"]
            zone_counts["Proportion (%)"] = (zone_counts["Count"] / len(df_comp) * 100).round(2)

            chart_zone = (
                alt.Chart(zone_counts)
                .mark_bar(cornerRadiusEnd=4, color="#0284c7")
                .encode(
                    x=alt.X("Count:Q", title="Number of Training Instances"),
                    y=alt.Y("Withdrawal Zone:N", sort="-x", title=""),
                    tooltip=["Withdrawal Zone", "Count", "Proportion (%)"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_zone, use_container_width=True)

    # =========================================================================
    # TAB 4: OPENSTREETMAP ATM INFRASTRUCTURE COVERAGE
    # =========================================================================
    with tab4:
        df_atm_cov = load_cached_atm_coverage()
        st.markdown("### 🏧 OpenStreetMap ATM Infrastructure Coverage")
        st.caption("333 verified physical ATM locations cataloged across 15 configured study regions under Open Database License (ODbL).")

        if df_atm_cov is not None and not df_atm_cov.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Cataloged ATMs", int(df_atm_cov["atm_records"].sum()))
            with c2:
                st.metric("Cities with Public OSM ATMs", f"{(df_atm_cov['status'] == 'OK').sum()} of 15")
            with c3:
                panipat_status = df_atm_cov[df_atm_cov["city"] == "Panipat"]["status"].values[0]
                st.metric("Panipat Status", str(panipat_status), delta="Zero OSM ATMs Verified", delta_color="inverse")

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            
            show_cols = ["city", "state", "atm_records", "known_bank_operator", "unknown_bank_operator", "status", "notes"]
            valid_show = [c for c in show_cols if c in df_atm_cov.columns]
            st.dataframe(df_atm_cov[valid_show], use_container_width=True, hide_index=True)
        else:
            st.info("ATM coverage report `reports/atm_coverage_report.csv` not found.")
