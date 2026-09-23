"""Analytics Page: Dataset statistics, model evaluation benchmarks, and zone distributions."""

import streamlit as st
import pandas as pd
import numpy as np

from src.data.loader import load_processed_complaints, load_raw_complaints
from src.data.validator import validate_complaints_data
from src.preprocessing.pipeline import prepare_xy
from src.models.train import train_candidate_models
from src.config import SUPPORTED_MODELS, LOCATION_CLASSIFIER_DIR

def render_analytics() -> None:
    st.markdown("## 📈 Analytics & Model Evaluation Workbench")
    st.markdown("Inspect synthetic dataset characteristics, distribution skews, and benchmark ML classifier performances.")

    tab1, tab2, tab3 = st.tabs(["📊 Dataset Profiling", "🤖 Model Evaluation & Metrics", "🗺️ Geographic Distributions"])

    # Load complaints if available
    df = load_processed_complaints()
    if df is None:
        df = load_raw_complaints()

    with tab1:
        st.markdown("### 📋 Complaint Dataset Health & Statistics")
        if df is not None and not df.empty:
            v_res = validate_complaints_data(df)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Records", f"{v_res['total_records']:,}")
            with c2:
                st.metric("Total Features", v_res["total_columns"])
            with c3:
                st.metric("Duplicate Rows", v_res["duplicate_rows_count"])
            with c4:
                st.metric("Columns with Nulls", len(v_res["missing_values_by_column"]))

            st.markdown("#### Sample Records")
            st.dataframe(df.head(10), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Fraud Types Breakdown")
                if "fraud_type" in df.columns:
                    st.bar_chart(df["fraud_type"].value_counts())
            with col_b:
                st.markdown("#### Transaction Amounts by Category")
                if "amount_category" in df.columns:
                    st.bar_chart(df["amount_category"].value_counts())
        else:
            st.warning("Complaints dataset not found in `data/processed/` or `data/raw/`.")
            st.info("Place `cybercrime_complaints.csv` into `data/processed/` or use the data generator.")

    with tab2:
        st.markdown("### 🏆 Location Classifier Benchmarking")
        st.caption("Compare Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest using stratified holdout.")

        if df is not None and not df.empty:
            if st.button("🚀 Train & Benchmark All 4 Candidate Models"):
                with st.spinner("Executing stratified train-test split and evaluating candidate pipelines..."):
                    try:
                        X, y = prepare_xy(df)
                        # Train on sample for fast UI interactive response if large
                        sample_size = min(len(X), 5000)
                        results = train_candidate_models(X.iloc[:sample_size], y.iloc[:sample_size], save_artifacts=True)
                        st.session_state["benchmark_results"] = results
                        st.success(f"Benchmarking complete! Top model: **{results['best_model_name']}**")
                    except Exception as err:
                        st.error(f"Training failed: {err}")

            if "benchmark_results" in st.session_state:
                b_res = st.session_state["benchmark_results"]
                summary_df = pd.DataFrame(b_res["metrics_summary"])
                cols_to_show = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_weighted"]
                st.markdown("#### 📊 Quantitative Benchmark Results")
                st.dataframe(summary_df[[c for c in cols_to_show if c in summary_df.columns]], use_container_width=True, hide_index=True)

                st.markdown("#### Confusion Matrix (Best Model)")
                best_model_record = [r for r in b_res["metrics_summary"] if r["model"] == b_res["best_model_name"]][0]
                cm = np.array(best_model_record["confusion_matrix"])
                st.dataframe(pd.DataFrame(cm), use_container_width=True)
            else:
                st.info("Click the button above to run the 4-model evaluation suite.")
        else:
            st.info("Dataset required to run model training benchmark.")

    with tab3:
        st.markdown("### 📍 Operational Zone Distribution")
        if df is not None and not df.empty and "withdrawal_zone" in df.columns:
            st.bar_chart(df["withdrawal_zone"].value_counts())
        else:
            st.info("Zone distribution available after loading complaints dataset.")
