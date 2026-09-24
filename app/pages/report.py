"""Reports Page: View, preview, and download generated executive intelligence reports.

Consumes the authoritative st.session_state["case_analysis"] object to dynamically
generate and deliver professional HTML and PDF executive intelligence briefs.
"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from src.config import GENERATED_REPORTS_DIR
from src.intelligence.report import generate_executive_report, generate_executive_report_bundle


def render_report_page() -> None:
    st.markdown("## 📑 Executive Intelligence Reports")
    st.markdown("Access structured intelligence briefs prepared for non-technical investigators, prosecutors, and executive leadership.")

    # 1. Active Case Analysis Check
    has_analysis = "case_analysis" in st.session_state and st.session_state["case_analysis"]

    if has_analysis:
        analysis = st.session_state["case_analysis"]
        case = analysis.get("case_data", {})
        pred = analysis.get("prediction", {})
        ranking = analysis.get("ranking", {})
        ranked_atms = ranking.get("ranked_atms", [])
        
        top_atm = ranked_atms[0] if ranked_atms else {}
        top_label = f"{top_atm.get('bank', 'Unknown')} ({top_atm.get('atm_id', 'N/A')})" if top_atm else "No verified OSM ATMs in sector"
        top_score = f"{top_atm.get('overall_score', 0.0):.1f} / 100" if top_atm else "N/A"
        conf_pct = float(pred.get("prediction_confidence", 0.0)) * 100
        tier = str(pred.get("confidence_tier", "HIGH")).upper()

        st.markdown("### 🎯 Active Investigation Case Brief")
        
        # Status Card
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Case Reference", str(case.get("complaint_id", "N/A")))
            st.caption(f"📍 {case.get('city')}, {case.get('state')}")
        with sc2:
            st.metric("Predicted Zone", str(pred.get("predicted_zone", "N/A")))
            st.caption(f"⚡ Confidence: {conf_pct:.1f}% ({tier})")
        with sc3:
            st.metric("Disputed Amount", f"₹{case.get('amount', 0):,}")
            st.caption(f"💳 {case.get('bank')} / {case.get('transaction_type')}")
        with sc4:
            st.metric("Top ATM Candidate", str(top_atm.get("bank", "N/A")) if top_atm else "None")
            st.caption(f"⭐ Score: {top_score}")

        # Check report currency
        report_status = st.session_state.get("report_status", "OUTDATED")
        current_report = st.session_state.get("current_report_path", None)

        col_gen, col_stat = st.columns([1.5, 1])
        with col_gen:
            gen_btn = st.button("⚡ Generate Intelligence Report (HTML & PDF)", type="primary", use_container_width=True)
            if gen_btn:
                with st.spinner("Compiling formal intelligence brief with probability charts and score breakdowns..."):
                    bundle = generate_executive_report_bundle(analysis)
                    st.session_state["current_report_path"] = bundle["html_path"]
                    st.session_state["report_status"] = "CURRENT"
                    current_report = bundle["html_path"]
                    report_status = "CURRENT"
                st.success("✅ Report successfully compiled!")
                st.rerun()

        with col_stat:
            if report_status == "CURRENT" and current_report and current_report.exists():
                st.info(f"🟢 **Status:** Active Case Brief Compiled (`{current_report.name}`)")
            else:
                st.warning("⚠️ **Status:** Outdated or Not Yet Compiled. Click button to generate.")

        # Download and Preview Controls for Current Report
        if current_report and current_report.exists() and report_status == "CURRENT":
            st.markdown("---")
            st.markdown("### 📥 Download Current Case Report")
            c_dl1, c_dl2 = st.columns(2)
            
            with c_dl1:
                with open(current_report, "r", encoding="utf-8") as f:
                    html_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="⬇️ Download Executive HTML Report",
                    data=html_bytes,
                    file_name=current_report.name,
                    mime="text/html",
                    use_container_width=True,
                )

            with c_dl2:
                pdf_path = current_report.with_suffix(".pdf")
                if pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Formal PDF Brief",
                        data=pdf_bytes,
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("PDF version compiling or unavailable.")

            st.markdown("---")
            st.markdown("### 👁️ Live Report Preview")
            with open(current_report, "r", encoding="utf-8") as f:
                html_preview = f.read()
            components.html(html_preview, height=900, scrolling=True)

    else:
        st.info("ℹ️ **No active case analysis found.** Please enter complaint metadata and execute analysis in the **Investigation** workbench before generating an intelligence brief.")
        if st.button("🔍 Go to Investigation Workbench"):
            st.session_state["current_page"] = "investigation"
            st.rerun()

    # 2. Historical Generated Reports Archive
    st.markdown("---")
    with st.expander("📁 Generated Intelligence Reports Archive", expanded=not has_analysis):
        report_files = sorted(
            list(GENERATED_REPORTS_DIR.glob("*.html")) + list(GENERATED_REPORTS_DIR.glob("*.pdf")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not report_files:
            st.caption("No historical reports in archive.")
            return

        col_select, col_actions = st.columns([2, 1])
        with col_select:
            selected_file_name = st.selectbox(
                "Select Historical Report",
                options=[p.name for p in report_files],
                index=0,
            )
        with col_actions:
            selected_path = GENERATED_REPORTS_DIR / selected_file_name
            with open(selected_path, "rb") as f:
                file_bytes = f.read()

            mime_type = "text/html" if selected_path.suffix == ".html" else "application/pdf"
            st.download_button(
                label=f"⬇️ Download {selected_path.suffix.upper()[1:]}",
                data=file_bytes,
                file_name=selected_file_name,
                mime=mime_type,
                use_container_width=True,
            )

        if selected_path.suffix == ".html":
            st.caption(f"Previewing `{selected_path.name}`:")
            with open(selected_path, "r", encoding="utf-8") as f:
                hist_html = f.read()
            components.html(hist_html, height=700, scrolling=True)
        else:
            st.info(f"PDF Report `{selected_path.name}` ready for download above.")
