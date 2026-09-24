# Executive Intelligence Reporting Engine (Phase 9)

## 1. Objective & Architectural Role
The CyberTrace Executive Intelligence Reporting Engine translates the live dynamic outputs of the location model, probability diagnostics, geospatial discovery, and multi-criteria ATM candidate ranking engines into formal, human-readable intelligence briefs available in **HTML** and **PDF** formats.

The reporting engine operates strictly as a downstream presentation layer:
- **No Independent Inference**: Consumes `case_analysis` produced by `src/intelligence/case_analysis.py`.
- **No Independent Ranking**: Slices and formats candidate ATMs already scored by `src/atm/ranking.py`.
- **Zero Static Demo Data**: 100% dynamically compiled from the investigator's active case submission.

---

## 2. Dynamic Pipeline Integration

```
USER INCIDENT INPUT
          ↓
  analyze_case() [Phase 7]
          ↓
  Authoritative Result (st.session_state["case_analysis"])
          ↓
  prepare_report_data() [Phase 9 data normalization]
          ↓
  ├── HTML Engine (Jinja2 + reports/templates/executive_report_template.html)
  └── PDF Engine  (ReportLab 5.0 + src/intelligence/report_pdf.py)
          ↓
  Saved Artifacts in reports/generated/
  ├── CYBERTRACE_<CASE_ID>_<TIMESTAMP>.html
  └── CYBERTRACE_<CASE_ID>_<TIMESTAMP>.pdf
          ↓
  Streamlit Report Workbench (Preview & User Download)
```

---

## 3. Mandatory 14 Report Sections

Every generated executive intelligence brief includes 14 structured intelligence sections:

1. **Executive Summary**: High-level overview card featuring Case ID, Reported Location, Financial Disputed Amount, Predicted Cash-Out Zone, Model Confidence % & Tier, and Highest-Ranked ATM Candidate (#1) with composite score.
2. **Case & Incident Profile**: Structured breakdown of victim report, financial value, payment channel, issuing institution, jurisdiction, and modus operandi.
3. **Location Prediction & Probability Analysis**: Primary predicted withdrawal zone, second-best sector, probability margin ($\Delta$), active model ID, sorted table of all 10 evaluated sectors, and high-resolution probability distribution horizontal bar chart.
4. **Confidence & Uncertainty Diagnostic**: Plain-language explanation of the Phase 6 threshold framework:
   - **HIGH ($\Delta \ge 0.30$)**: Decisive probability lead.
   - **MEDIUM ($0.15 \le \Delta < 0.30$)**: Contested multi-sector boundary (e.g. NCR zones).
   - **LOW ($\Delta < 0.15$)**: Ambiguous classification requiring broader intelligence corroboration.
5. **Geographic Assessment & Cross-Zone Search**: Geographic origin coordinates, target sector boundaries, and cross-boundary status. Activates a dedicated warning section when `cross_zone_search == True`.
6. **Candidate ATM Assessment (Top 10)**: Prioritized table of OpenStreetMap physical ATM infrastructure with distance, composite score, bank brand, network operator, and compatibility flags.
7. **Highest-Ranked ATM Candidate Spotlight (#1)**: Dedicated intelligence card for Candidate #1 featuring complete 6-criteria score breakdown (Zone Likelihood, Proximity Decay, Bank Match, 24x7 Hours, Activity Simulation, Amount Match).
8. **Analytical Reasoning & Contributing Signals**: Bulleted breakdown of influencing operational factors and recommended priority actions from `src/intelligence/explanation.py`.
9. **Evidence Hierarchy & Legal Separation**: Clear 3-tier categorization:
   - *Level 1 (Model Prediction)*: Statistical likelihood.
   - *Level 2 (Analytical Candidate)*: Multi-criteria physical infrastructure score.
   - *Level 3 (Authoritative Forensic Evidence)*: Banking interchange electronic journal logs, physical CCTV recordings, suspect CDR data.
10. **Data Sources & Provenance**: Explicitly delineates real external crowdsourced data (OpenStreetMap ODbL) from synthetic research baselines (complaints, simulated activity) and ML artifacts.
11. **Methodological Limitations**: 10-point comprehensive limitations disclosure.
12. **Recommended Law Enforcement Verification Steps**: 6 actionable, legally compliant investigative steps for authorized officers.
13. **Official Cybercrime Reporting**: Official contact details directing victims to `cybercrime.gov.in` and Citizen Financial Cyber Fraud Helpline `1930`.
14. **System Audit Metadata**: Exact model artifact reference, ranking engine version (`v1.0`), report version (`v1.0`), and generation timestamp.

---

## 4. Evidentiary Language & Claim Boundaries

The reporting engine strictly prohibits terms implying criminal guilt or physical transaction confirmation:
- ❌ **Prohibited**: *"Confirmed ATM"*, *"Fraud ATM"*, *"Criminal cash-out location"*, *"Actual withdrawal site"*, *"Perpetrator location"*.
- ✅ **Permitted**: *"Predicted withdrawal zone"*, *"Highest-ranked ATM candidate"*, *"Model-supported candidate"*, *"Analytical proximity"*.

---

## 5. Dual-Format Generation Engine

### HTML Report Engine
- **Template**: `reports/templates/executive_report_template.html`.
- **Renderer**: Jinja2.
- **Design**: CyberTrace Dark Intelligence theme with responsive horizontal CSS meters, visual status chips, print-friendly `@media print` typography, and embedded Base64 probability charts.

### PDF Report Engine
- **Module**: `src/intelligence/report_pdf.py`.
- **Renderer**: ReportLab 5.0 Platypus (`SimpleDocTemplate`, `Table`, `Paragraph`, `Image`, `HRFlowable`).
- **Features**: Two-pass `NumberedCanvas` delivering dynamic "Page X of Y" footers, running headers on pages $> 1$, auto-wrapping table cells preventing column overflow, embedded high-DPI horizontal probability bar charts, and colored warning callout boxes.
