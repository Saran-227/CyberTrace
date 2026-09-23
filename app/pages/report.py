"""Reports Page: View, preview, and download generated executive intelligence reports."""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from src.config import GENERATED_REPORTS_DIR

def render_report_page() -> None:
    st.markdown("## 📑 Executive Intelligence Reports")
    st.markdown("Access structured intelligence briefs prepared for non-technical investigators and executive decision-makers.")

    # List all files in reports/generated/
    report_files = sorted(
        list(GENERATED_REPORTS_DIR.glob("*.html")) + list(GENERATED_REPORTS_DIR.glob("*.pdf")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not report_files:
        st.info("No reports generated yet. Conduct an analysis in **Investigation** and click **Generate Executive Intelligence Report**.")
        return

    col_select, col_actions = st.columns([2, 1])
    with col_select:
        selected_file_name = st.selectbox(
            "Select Intelligence Report",
            options=[p.name for p in report_files],
            index=0,
        )
    with col_actions:
        selected_path = GENERATED_REPORTS_DIR / selected_file_name
        with open(selected_path, "rb") as f:
            file_bytes = f.read()

        mime_type = "text/html" if selected_path.suffix == ".html" else "application/pdf"
        st.download_button(
            label="⬇️ Download Report",
            data=file_bytes,
            file_name=selected_file_name,
            mime=mime_type,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 👁️ Report Preview")

    if selected_path.suffix == ".html":
        with open(selected_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.info(f"PDF Report `{selected_path.name}` ready for download above.")
