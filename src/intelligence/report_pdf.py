"""PDF Executive Intelligence Report Generator for CyberTrace (Phase 9).

Produces professional, multi-page PDF intelligence briefs using ReportLab.
Adheres strictly to academic synthetic data disclosures and evidence boundaries.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from src.utils.logging import get_logger

logger = get_logger("ReportPDFGenerator")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count and professional headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 762, "CYBERTRACE EXECUTIVE INTELLIGENCE BRIEF")
            self.drawRightString(576, 762, f"RESTRICTED / LAW ENFORCEMENT ADVISORY")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)

        self.setFont("Helvetica", 7.5)
        self.drawString(36, 30, "CyberTrace Location Intelligence Platform | Map data © OpenStreetMap contributors")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_text)
        self.restoreState()


def create_probability_chart_image(
    zone_probabilities: Dict[str, float],
    predicted_zone: str,
    second_zone: Optional[str] = None,
) -> io.BytesIO:
    """Generate a high-resolution horizontal bar chart of all 10 zone probabilities."""
    sorted_items = sorted(zone_probabilities.items(), key=lambda x: x[1], reverse=False)
    zones = [k for k, v in sorted_items]
    vals = [v * 100 for k, v in sorted_items]

    bar_colors = []
    for z in zones:
        if z == predicted_zone:
            bar_colors.append("#0284c7")  # Cyan/blue primary
        elif second_zone and z == second_zone:
            bar_colors.append("#f59e0b")  # Amber secondary
        else:
            bar_colors.append("#94a3b8")  # Neutral slate

    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=180)
    bars = ax.barh(zones, vals, color=bar_colors, height=0.6)
    ax.set_xlabel("Model Probability (%)", fontsize=8, fontweight="bold", color="#334155")
    ax.set_title("Withdrawal Zone Probability Distribution (10 Evaluated Sectors)", fontsize=9, fontweight="bold", color="#0f172a", pad=8)
    
    max_val = max(vals) if vals else 1.0
    ax.set_xlim(0, max(max_val * 1.18, 12))

    for bar, val in zip(bars, vals):
        if val >= 0.1:
            ax.text(
                bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#0f172a",
                fontweight="bold",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=7.5)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Build and write a formal multi-page PDF intelligence brief."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
    )
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#991b1b"),
    )
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    case = report_data.get("case_data", {})
    pred = report_data.get("prediction", {})
    ranking = report_data.get("ranking", {})
    ranked_atms = ranking.get("ranked_atms", [])
    expl = report_data.get("explanation", {})
    prov = report_data.get("provenance", {})
    metadata = report_data.get("metadata", {})

    case_id = case.get("complaint_id", report_data.get("case_id", "CASE-UNKNOWN"))
    gen_time = metadata.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # =========================================================================
    # HEADER BANNER
    # =========================================================================
    header_data = [
        [
            Paragraph("CYBERTRACE", title_style),
            Paragraph("<font color='#0284c7'><b>OFFICIAL ANALYTICAL ADVISORY</b></font><br/><font size='7' color='#64748b'>RESTRICTED DISSEMINATION</font>", ParagraphStyle("Badge", alignment=2, fontName="Helvetica", leading=10)),
        ],
        [
            Paragraph("EXECUTIVE INTELLIGENCE BRIEF", subtitle_style),
            Paragraph(f"<b>Case Ref:</b> {case_id}<br/><b>Generated:</b> {gen_time}", ParagraphStyle("SubMeta", alignment=2, fontName="Helvetica", fontSize=7.5, leading=10, textColor=colors.HexColor("#475569"))),
        ],
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))

    # =========================================================================
    # ACADEMIC SYNTHETIC RESEARCH NOTICE
    # =========================================================================
    notice_text = (
        "<b>ACADEMIC SYNTHETIC RESEARCH NOTICE & EVIDENCE BOUNDARIES:</b><br/>"
        "All complaint records, victim profiles, and ATM activity histories used by CyberTrace are <b>synthetic</b>. "
        "Predicted withdrawal zones and ATM rankings represent mathematical, geographic, and simulated operational likelihoods. "
        "<b>ATM candidates are NOT confirmed locations of criminal withdrawal.</b> Official forensic confirmation strictly "
        "requires authorized banking interchange logs and physical CCTV surveillance verification."
    )
    notice_table = Table([[Paragraph(notice_text, disclaimer_style)]], colWidths=[540])
    notice_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ef4444")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(notice_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Summary", sec_heading))
    
    top_atm = ranked_atms[0] if ranked_atms else {}
    top_atm_label = f"{top_atm.get('atm_id', 'N/A')} ({top_atm.get('bank', 'N/A')})" if top_atm else "No verified OSM ATMs found"
    top_score_label = f"{top_atm.get('overall_score', 0.0):.1f} / 100" if top_atm else "N/A"
    conf_pct = float(pred.get("prediction_confidence", 0.0)) * 100
    tier = str(pred.get("confidence_tier", "HIGH")).upper()

    summary_cards = [
        [
            Paragraph("<b>Case Reference:</b>", body_style), Paragraph(str(case_id), body_bold),
            Paragraph("<b>Predicted Cash-Out Zone:</b>", body_style), Paragraph(f"<font color='#0284c7'><b>{pred.get('predicted_zone', 'N/A')}</b></font>", body_bold),
        ],
        [
            Paragraph("<b>Reported Location:</b>", body_style), Paragraph(f"{case.get('city', 'N/A')}, {case.get('state', 'N/A')}", body_style),
            Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"<b>{conf_pct:.1f}%</b> ({tier})", body_style),
        ],
        [
            Paragraph("<b>Reported Disputed Amount:</b>", body_style), Paragraph(f"INR {case.get('amount', 0):,} ({case.get('amount_category', 'Standard')})", body_style),
            Paragraph("<b>Top ATM Candidate (#1):</b>", body_style), Paragraph(top_atm_label, body_style),
        ],
        [
            Paragraph("<b>Reporting Bank & Rail:</b>", body_style), Paragraph(f"{case.get('bank', 'N/A')} / {case.get('transaction_type', 'N/A')}", body_style),
            Paragraph("<b>Candidate Score:</b>", body_style), Paragraph(f"<b>{top_score_label}</b>", body_bold),
        ],
    ]
    sum_table = Table(summary_cards, colWidths=[120, 150, 130, 140])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 2. CASE INFORMATION & INCIDENT PROFILE
    # =========================================================================
    story.append(Paragraph("2. Case Information & Incident Profile", sec_heading))

    incident_narrative = (
        f"CyberTrace ingested the submitted incident metadata (Complaint Reference: <b>{case_id}</b>) and evaluated "
        f"geographic, institutional, and temporal indicators to estimate the statistical probability distribution "
        f"of withdrawal cash-out zones. The reported fraud incident involves <b>INR {case.get('amount', 0):,}</b> "
        f"via <b>{case.get('transaction_type', 'N/A')}</b> originating from <b>{case.get('bank', 'N/A')}</b> in "
        f"<b>{case.get('city', 'N/A')}</b> (District: {case.get('district', 'N/A')}, State: {case.get('state', 'N/A')}). "
        f"The incident modus operandi was classified as <b>{case.get('fraud_type', 'Cyber Fraud')}</b>."
    )
    story.append(Paragraph(incident_narrative, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 3. LOCATION PREDICTION & PROBABILITY ANALYSIS
    # =========================================================================
    story.append(Paragraph("3. Location Prediction & Probability Analysis", sec_heading))

    margin = float(pred.get("probability_margin", 0.0))
    second_zone = pred.get("second_best_zone", "None")
    active_model = pred.get("model_id", "random_forest_full_none")

    pred_summary = (
        f"Primary Location Classifier (<code>{active_model}</code>) predicted <b>{pred.get('predicted_zone')}</b> "
        f"with <b>{conf_pct:.1f}%</b> posterior probability. Second-ranked sector is <b>{second_zone}</b> with a probability margin "
        f"of <b>Δ {margin:.3f}</b>, placing this analysis in the <b>{tier} CONFIDENCE</b> diagnostic tier."
    )
    story.append(Paragraph(pred_summary, body_style))
    story.append(Spacer(1, 6))

    # Embed Probability Chart Image
    zone_probs = pred.get("zone_probabilities", {})
    if zone_probs:
        chart_buf = create_probability_chart_image(zone_probs, pred.get("predicted_zone"), second_zone)
        chart_img = Image(chart_buf, width=480, height=205)
        story.append(chart_img)
        story.append(Spacer(1, 6))

    # All 10 zone probability table (sorted)
    sorted_probs = sorted(zone_probs.items(), key=lambda x: x[1], reverse=True)
    prob_rows = [
        [Paragraph("Zone", th_style), Paragraph("Probability", th_style), Paragraph("Status", th_style),
         Paragraph("Zone", th_style), Paragraph("Probability", th_style), Paragraph("Status", th_style)]
    ]
    half = (len(sorted_probs) + 1) // 2
    for i in range(half):
        z1, p1 = sorted_probs[i]
        status1 = "PRIMARY" if z1 == pred.get("predicted_zone") else ("SECONDARY" if z1 == second_zone else "Alternative")
        if i + half < len(sorted_probs):
            z2, p2 = sorted_probs[i + half]
            status2 = "PRIMARY" if z2 == pred.get("predicted_zone") else ("SECONDARY" if z2 == second_zone else "Alternative")
        else:
            z2, p2, status2 = "", 0.0, ""
        
        prob_rows.append([
            Paragraph(f"<b>{z1}</b>", td_style), Paragraph(f"{p1*100:.1f}%", td_bold if status1 == "PRIMARY" else td_style), Paragraph(status1, td_style),
            Paragraph(f"<b>{z2}</b>" if z2 else "", td_style), Paragraph(f"{p2*100:.1f}%" if z2 else "", td_style), Paragraph(status2, td_style),
        ])
    prob_table = Table(prob_rows, colWidths=[70, 70, 130, 70, 70, 130])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 4. CONFIDENCE & UNCERTAINTY FRAMEWORK
    # =========================================================================
    story.append(Paragraph("4. Confidence & Uncertainty Diagnostic", sec_heading))
    if tier == "HIGH":
        conf_expl = (
            "<b>High Confidence (Δ ≥ 0.30):</b> The leading zone has a decisive probability lead over all competing sectors. "
            "Downstream ATM candidate discovery focuses predominantly on infrastructure within the primary predicted zone."
        )
    elif tier == "MEDIUM":
        conf_expl = (
            "<b>Medium Confidence (0.15 ≤ Δ &lt; 0.30):</b> The location classifier shows competitive probabilities across multiple sectors "
            f"(such as {pred.get('predicted_zone')} and {second_zone}). Dual-sector cross-zone search was automatically activated to avoid boundary misses."
        )
    else:
        conf_expl = (
            "<b>Low Confidence (Δ &lt; 0.15):</b> High uncertainty across multiple zones. The prediction does not exhibit a strong separation. "
            "Investigators should treat sector recommendations as exploratory and cross-verify with device geolocation or IP data."
        )
    story.append(Paragraph(conf_expl, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 5. GEOGRAPHIC ASSESSMENT & CROSS-ZONE SEARCH
    # =========================================================================
    story.append(Paragraph("5. Geographic Assessment", sec_heading))
    cross_zone_active = ranking.get("cross_zone_search", False)
    cand_zones_str = ", ".join(ranking.get("candidate_zones", [pred.get("predicted_zone", "N/A")]))
    
    geo_narrative = (
        f"Complaint Origin Coordinates: <b>Lat {case.get('complaint_latitude', 0.0):.4f}, Lon {case.get('complaint_longitude', 0.0):.4f}</b> "
        f"({case.get('city', 'Reported City')}). Evaluated Operational Sectors: <b>{cand_zones_str}</b>. "
        f"Cross-Boundary Dual-Sector Evaluation: <b>{'ACTIVE' if cross_zone_active else 'INACTIVE'}</b>."
    )
    story.append(Paragraph(geo_narrative, body_style))

    if cross_zone_active:
        story.append(Spacer(1, 4))
        cross_box = (
            "<b>CROSS-ZONE CANDIDATE ASSESSMENT ACTIVE:</b><br/>"
            f"Because of contested probability between primary sector {pred.get('predicted_zone')} and secondary sector {second_zone}, "
            "ATM candidates were simultaneously evaluated and ranked across both administrative zones. "
            "This reflects statistical uncertainty between adjacent jurisdictions (e.g. NCR boundary zones) and must not be interpreted as physical movement evidence."
        )
        cross_table = Table([[Paragraph(cross_box, body_style)]], colWidths=[540])
        cross_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(cross_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 6. CANDIDATE ATM ASSESSMENT (TOP 10)
    # =========================================================================
    story.append(Paragraph("6. Candidate ATM Infrastructure Assessment", sec_heading))
    
    if not ranked_atms:
        no_atm_box = (
            "<b>DATA COVERAGE NOTICE:</b> No verified OpenStreetMap ATM points of interest are currently cataloged "
            f"for the evaluated sector ({case.get('city', 'Target City')} / {pred.get('predicted_zone')}). "
            "CyberTrace utilizes crowdsourced OpenStreetMap data without commercial API dependencies. "
            "Physical ATM verification must proceed via direct banking queries."
        )
        no_table = Table([[Paragraph(no_atm_box, body_style)]], colWidths=[540])
        no_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(no_table)
    else:
        story.append(Paragraph(
            "The multi-criteria ranking engine evaluated physical ATM infrastructure sourced from OpenStreetMap against 6 weighted criteria: "
            "Zone Likelihood (30%), Proximity Decay (25%), Bank Compatibility (20%), 24x7 Hours (10%), Activity Compatibility (10%), Amount Match (5%).",
            body_style
        ))
        story.append(Spacer(1, 4))

        atm_rows = [
            [
                Paragraph("Rank", th_style),
                Paragraph("ATM ID", th_style),
                Paragraph("Bank / Operator", th_style),
                Paragraph("City / District", th_style),
                Paragraph("Zone", th_style),
                Paragraph("Dist (km)", th_style),
                Paragraph("Score", th_style),
                Paragraph("Bank Match", th_style),
            ]
        ]
        for atm in ranked_atms[:10]:
            is_top = (atm.get("rank") == 1)
            flags = atm.get("evidence_flags", {})
            atm_rows.append([
                Paragraph(f"<b>#{atm.get('rank', 1)}</b>", td_bold if is_top else td_style),
                Paragraph(str(atm.get("atm_id", ""))[:14], td_style),
                Paragraph(str(atm.get("bank", "Unknown"))[:16], td_bold if is_top else td_style),
                Paragraph(f"{atm.get('city', '')} ({atm.get('district', '')})"[:18], td_style),
                Paragraph(str(atm.get("zone", "")), td_style),
                Paragraph(f"{atm.get('distance_km', 0.0):.2f}", td_style),
                Paragraph(f"<b>{atm.get('overall_score', 0.0):.1f}</b>", td_bold if is_top else td_style),
                Paragraph(str(flags.get("bank_match", "N/A"))[:12], td_style),
            ])

        atm_table = Table(atm_rows, colWidths=[35, 75, 110, 110, 50, 50, 50, 60])
        atm_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f0fdf4")),  # Highlight top candidate in soft green
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(atm_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 7. HIGHEST-RANKED ATM CANDIDATE SPOTLIGHT & SCORE BREAKDOWN
    # =========================================================================
    if ranked_atms:
        story.append(Paragraph("7. Highest-Ranked ATM Candidate Spotlight (#1)", sec_heading))
        t1 = ranked_atms[0]
        spotlight_data = [
            [
                Paragraph("<b>Designation:</b>", body_style), Paragraph("<font color='#10b981'><b>HIGHEST-RANKED CANDIDATE</b></font>", body_bold),
                Paragraph("<b>Composite Score:</b>", body_style), Paragraph(f"<b>{t1.get('overall_score', 0.0):.2f} / 100</b>", body_bold),
            ],
            [
                Paragraph("<b>ATM ID:</b>", body_style), Paragraph(str(t1.get("atm_id")), body_style),
                Paragraph("<b>Physical Coordinates:</b>", body_style), Paragraph(f"Lat {t1.get('latitude', 0.0):.5f}, Lon {t1.get('longitude', 0.0):.5f}", body_style),
            ],
            [
                Paragraph("<b>Bank / Operator:</b>", body_style), Paragraph(f"{t1.get('bank', 'Unknown')} ({t1.get('operator', 'Unknown')})", body_style),
                Paragraph("<b>Haversine Distance:</b>", body_style), Paragraph(f"<b>{t1.get('distance_km', 0.0):.2f} km</b> from sector origin", body_style),
            ],
            [
                Paragraph("<b>Assigned Sector:</b>", body_style), Paragraph(str(t1.get("zone")), body_style),
                Paragraph("<b>Operating Accessibility:</b>", body_style), Paragraph(str(t1.get("evidence_flags", {}).get("operating_hours", "Standard")), body_style),
            ],
        ]
        spot_table = Table(spotlight_data, colWidths=[110, 160, 110, 160])
        spot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#10b981")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbf7d0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(spot_table)
        story.append(Spacer(1, 6))

        # Detailed 6-component score table
        breakdown_rows = [
            [
                Paragraph("Zone Prob (30%)", th_style),
                Paragraph("Spatial Prox (25%)", th_style),
                Paragraph("Bank Match (20%)", th_style),
                Paragraph("24x7 Hours (10%)", th_style),
                Paragraph("Activity Sim (10%)", th_style),
                Paragraph("Amount Match (5%)", th_style),
                Paragraph("Overall (100%)", th_style),
            ],
            [
                Paragraph(f"{t1.get('zone_score', 0.0):.1f}", td_style),
                Paragraph(f"{t1.get('spatial_score', 0.0):.1f}", td_style),
                Paragraph(f"{t1.get('bank_score', 0.0):.1f}", td_style),
                Paragraph(f"{t1.get('time_score', 0.0):.1f}", td_style),
                Paragraph(f"{t1.get('activity_score', 0.0):.1f}", td_style),
                Paragraph(f"{t1.get('amount_score', 0.0):.1f}", td_style),
                Paragraph(f"<b>{t1.get('overall_score', 0.0):.2f}</b>", td_bold),
            ]
        ]
        bd_table = Table(breakdown_rows, colWidths=[77, 77, 77, 77, 77, 77, 78])
        bd_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(bd_table)
        story.append(Spacer(1, 8))

    # =========================================================================
    # 8. ANALYTICAL REASONING & CONTRIBUTING SIGNALS
    # =========================================================================
    story.append(Paragraph("8. Analytical Reasoning & Contributing Signals", sec_heading))
    for sig in expl.get("key_signals", []):
        story.append(Paragraph(f"• {sig}", body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 9. EVIDENCE HIERARCHY & METHODOLOGICAL PRINCIPLE
    # =========================================================================
    story.append(Paragraph("9. Evidence Hierarchy & Legal Separation", sec_heading))
    hierarchy_rows = [
        [Paragraph("Evidence Level", th_style), Paragraph("Nature of Intelligence", th_style), Paragraph("CyberTrace Output / Status", th_style)],
        [
            Paragraph("<b>LEVEL 1: Model Prediction</b>", td_bold),
            Paragraph("Supervised machine learning probability estimation across geographic zones.", td_style),
            Paragraph("<b>PRODUCED:</b> Statistical zone likelihood (e.g. Zone_02 88.4%)", td_style),
        ],
        [
            Paragraph("<b>LEVEL 2: Candidate Ranking</b>", td_bold),
            Paragraph("Multi-criteria scoring of open public ATM infrastructure.", td_style),
            Paragraph("<b>PRODUCED:</b> Prioritized OpenStreetMap candidate list (Rank #1-#N)", td_style),
        ],
        [
            Paragraph("<b>LEVEL 3: Physical Evidence</b>", td_bold),
            Paragraph("Bank interchange reconciliation logs, cardholder switch records, physical CCTV surveillance.", td_style),
            Paragraph("<font color='#991b1b'><b>NOT PRODUCED:</b> Requires authorized law enforcement subpoena</font>", td_style),
        ],
    ]
    hier_table = Table(hierarchy_rows, colWidths=[130, 220, 190])
    hier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(hier_table)
    story.append(Spacer(1, 8))

    # =========================================================================
    # 10. DATA SOURCES & PROVENANCE
    # =========================================================================
    story.append(Paragraph("10. Data Sources & Provenance", sec_heading))
    sources_text = (
        "• <b>Synthetic Cybercrime Complaint Dataset:</b> Academic baseline of 20,000 synthetic records with verified zero target leakage.<br/>"
        "• <b>OpenStreetMap Geographic Data:</b> 333 verified ATM locations ingested via Overpass API under Open Database License (ODbL).<br/>"
        "• <b>Synthetic ATM Operational Activity:</b> 719,280 hourly synthetic records simulating baseline diurnal footfall without fraud labels.<br/>"
        f"• <b>Location Model Artifact:</b> Scikit-Learn Pipeline (<code>{prov.get('model_artifact', 'random_forest_full_none.joblib')}</code>)."
    )
    story.append(Paragraph(sources_text, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 11. METHODOLOGICAL LIMITATIONS (10 POINTS)
    # =========================================================================
    story.append(Paragraph("11. Methodological Limitations", sec_heading))
    limitations = [
        "1. Complaint data is synthetic and designed for academic supervised-learning research.",
        "2. ATM historical activity is synthetic and reflects simulated operational baselines, not real bank transaction logs.",
        "3. OpenStreetMap ATM coverage is crowdsourced and may have coverage gaps (e.g. Panipat catalog has 0 OSM ATMs).",
        "4. Bank and operator metadata may be unrecorded in public OSM records, receiving default neutral compatibility scores.",
        "5. Candidate ranking is an analytical prioritization tool, not forensic proof of cash-out occurrence.",
        "6. Zone boundaries are statistical spatial bounding representations derived from training centroids.",
        "7. Model predictions reflect the distribution of the training dataset and may exhibit geographical bias.",
        "8. Actual banking transaction interchange logs are confidential and were not accessed by CyberTrace.",
        "9. CCTV recordings, device MAC addresses, and telecommunications records are required to corroborate candidates.",
        "10. High prediction uncertainty occurs in boundary regions (e.g. NCR sectors) requiring dual-zone candidate inspection.",
    ]
    for lim in limitations:
        story.append(Paragraph(lim, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 12. RECOMMENDED INVESTIGATIVE VERIFICATION STEPS
    # =========================================================================
    story.append(Paragraph("12. Recommended Investigative Verification Steps", sec_heading))
    recommendations = [
        "1. Issue formal legal notices to the acquiring bank for ATM electronic journal logs and cash-out timestamps.",
        "2. Request surveillance CCTV footage for top candidate ATMs covering a ±30 minute window around incident timing.",
        "3. Cross-reference suspect cellular tower CDR (Call Detail Record) data against candidate ATM coordinates.",
        "4. Request NPCI (National Payments Corporation of India) switch interchange logs for transaction tracing.",
        "5. Verify operational status and physical cash replenishment records for candidate ATMs during the incident window.",
        "6. File official intelligence documentation under the victim's National Cyber Crime Reporting Portal reference.",
    ]
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # 13. OFFICIAL CYBERCRIME REPORTING & AUDIT METADATA
    # =========================================================================
    story.append(Paragraph("13. Official Cybercrime Reporting & System Audit", sec_heading))
    reporting_info = (
        "<b>National Cyber Crime Reporting Portal:</b> https://www.cybercrime.gov.in/<br/>"
        "<b>National Financial Cyber Fraud Helpline:</b> <b>1930</b><br/>"
        "<i>Notice: CyberTrace is an academic research prototype and does not file official law enforcement complaints.</i>"
    )
    story.append(Paragraph(reporting_info, body_style))
    story.append(Spacer(1, 6))

    audit_box = [
        [
            Paragraph(f"<b>System:</b> CyberTrace {metadata.get('report_version', 'v1.0')}", meta_style),
            Paragraph(f"<b>Model:</b> {active_model}", meta_style),
            Paragraph(f"<b>Map Engine:</b> Leaflet / OSM Carto", meta_style),
            Paragraph(f"<b>Status:</b> ANALYSIS_COMPLETED", meta_style),
        ]
    ]
    audit_table = Table(audit_box, colWidths=[135, 135, 135, 135])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(audit_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"Generated PDF executive report successfully: {output_path}")
    return output_path
