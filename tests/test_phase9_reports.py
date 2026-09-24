"""Comprehensive Unit and Integration Tests for Phase 9: Executive Intelligence Report Generator.

Verifies the 30 requirements specified in CyberTrace Phase 9:
- Report contract conformity
- Dynamic parameter reflection (zero static examples)
- All 14 mandatory sections (Executive Summary to Reporting info)
- HTML and PDF file generation and content completeness
- Cross-zone conditional sections
- Panipat zero-ATM handling
- Synthetic data disclosures and claim boundary enforcement
"""

import pytest
from pathlib import Path
import re

from src.intelligence.case_analysis import analyze_case
from src.intelligence.report import (
    generate_executive_report,
    generate_executive_report_bundle,
    prepare_report_data,
    REPORT_VERSION,
)
from src.config import GENERATED_REPORTS_DIR


@pytest.fixture(scope="module")
def jalandhar_analysis():
    """Execute live analysis for standard high-confidence Jalandhar case."""
    return analyze_case({
        "complaint_id": "CT-TEST-JAL-001",
        "city": "Jalandhar",
        "amount": 15000.0,
        "bank": "HDFC",
        "transaction_type": "UPI",
        "fraud_type": "Phishing/Smishing",
    })


@pytest.fixture(scope="module")
def gurugram_analysis():
    """Execute live analysis for medium-confidence cross-zone NCR case."""
    return analyze_case({
        "complaint_id": "CT-TEST-GUR-002",
        "city": "Gurugram",
        "amount": 50000.0,
        "bank": "Axis",
        "transaction_type": "NEFT",
        "fraud_type": "Investment Scam",
    })


@pytest.fixture(scope="module")
def panipat_analysis():
    """Execute live analysis for zero-OSM ATM Panipat case."""
    return analyze_case({
        "complaint_id": "CT-TEST-PAN-003",
        "city": "Panipat",
        "amount": 10000.0,
        "bank": "PNB",
        "transaction_type": "IMPS",
        "fraud_type": "KYC Update",
    })


# 1. Report contract validation
def test_01_report_contract_validation(jalandhar_analysis):
    report_data = prepare_report_data(jalandhar_analysis)
    assert "case_id" in report_data
    assert "case_data" in report_data
    assert "prediction" in report_data
    assert "ranking" in report_data
    assert "ranked_atms" in report_data
    assert "candidate_zones" in report_data
    assert "explanation" in report_data
    assert "provenance" in report_data
    assert "metadata" in report_data
    assert "sorted_probabilities" in report_data
    assert len(report_data["sorted_probabilities"]) == 10


# 2. Report generation from valid case analysis
def test_02_report_generation_from_valid_case_analysis(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_jalandhar.html")
    assert report_path.exists()
    assert report_path.suffix == ".html"


# 3. Dynamic case ID appears
def test_03_dynamic_case_id_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_case_id.html")
    content = report_path.read_text(encoding="utf-8")
    assert "CT-TEST-JAL-001" in content


# 4. Dynamic predicted zone appears
def test_04_dynamic_predicted_zone_appears(jalandhar_analysis):
    pred_zone = jalandhar_analysis["prediction"]["predicted_zone"]
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_pred_zone.html")
    content = report_path.read_text(encoding="utf-8")
    assert pred_zone in content


# 5. Dynamic confidence appears
def test_05_dynamic_confidence_appears(jalandhar_analysis):
    conf_pct = f"{jalandhar_analysis['prediction']['prediction_confidence'] * 100:.1f}%"
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_conf.html")
    content = report_path.read_text(encoding="utf-8")
    assert conf_pct in content


# 6. All 10 zone probabilities appear
def test_06_all_10_zone_probabilities_appear(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_10zones.html")
    content = report_path.read_text(encoding="utf-8")
    for i in range(1, 11):
        assert f"Zone_{i:02d}" in content


# 7. Top ATM candidate appears
def test_07_top_atm_candidate_appears(jalandhar_analysis):
    top_atm = jalandhar_analysis["ranking"]["ranked_atms"][0]
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_top_atm.html")
    content = report_path.read_text(encoding="utf-8")
    assert top_atm["atm_id"] in content
    assert top_atm["bank"] in content


# 8. Top ATM score matches case_analysis
def test_08_top_atm_score_matches_case_analysis(jalandhar_analysis):
    top_atm = jalandhar_analysis["ranking"]["ranked_atms"][0]
    score_str = f"{top_atm['overall_score']:.2f}"
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_score.html")
    content = report_path.read_text(encoding="utf-8")
    assert score_str in content


# 9. Candidate rankings match Phase 7
def test_09_candidate_rankings_match_phase7(jalandhar_analysis):
    ranked = jalandhar_analysis["ranking"]["ranked_atms"]
    report_data = prepare_report_data(jalandhar_analysis)
    report_atms = report_data["ranked_atms"]
    assert len(report_atms) == len(ranked)
    for i in range(min(5, len(ranked))):
        assert report_atms[i]["atm_id"] == ranked[i]["atm_id"]
        assert report_atms[i]["overall_score"] == pytest.approx(ranked[i]["overall_score"], abs=0.01)


# 10. No hardcoded example case appears
def test_10_no_hardcoded_example_case_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_no_hardcode.html")
    content = report_path.read_text(encoding="utf-8")
    # Verify values from hypothetical other cases don't appear in Jalandhar case
    assert "Gurugram" not in content
    assert "Faridabad" not in content


# 11. Synthetic activity disclaimer appears
def test_11_synthetic_activity_disclaimer_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_synth_disclaimer.html")
    content = report_path.read_text(encoding="utf-8")
    assert "ACADEMIC SYNTHETIC RESEARCH NOTICE" in content
    assert "synthetic" in content.lower()


# 12. ATM candidate disclaimer appears
def test_12_atm_candidate_disclaimer_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_atm_disclaimer.html")
    content = report_path.read_text(encoding="utf-8")
    assert "NOT confirmed locations of criminal withdrawal" in content or "NOT confirmed locations of withdrawal" in content


# 13. OSM attribution appears
def test_13_osm_attribution_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_osm.html")
    content = report_path.read_text(encoding="utf-8")
    assert "OpenStreetMap" in content
    assert "contributors" in content


# 14. Evidence hierarchy appears
def test_14_evidence_hierarchy_appears(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_hier.html")
    content = report_path.read_text(encoding="utf-8")
    assert "LEVEL 1: Model Prediction" in content
    assert "LEVEL 2: Analytical Candidate" in content
    assert "LEVEL 3: Physical Evidence" in content


# 15. Limitations appear
def test_15_limitations_appear(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_limits.html")
    content = report_path.read_text(encoding="utf-8")
    assert "Methodological Limitations" in content
    assert "crowdsourced" in content.lower()
    assert "CCTV" in content


# 16. Verification recommendations appear
def test_16_verification_recommendations_appear(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_recs.html")
    content = report_path.read_text(encoding="utf-8")
    assert "Recommended Investigative Verification Steps" in content
    assert "CCTV" in content
    assert "interchange" in content.lower()


# 17. HTML file is generated
def test_17_html_file_is_generated(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_exists.html")
    assert report_path.exists()
    assert report_path.is_file()


# 18. HTML is non-empty
def test_18_html_is_non_empty(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_nonempty.html")
    assert report_path.stat().st_size > 1000


# 19. PDF file is generated
def test_19_pdf_file_is_generated(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_pdf.html", generate_pdf=True)
    pdf_path = report_path.with_suffix(".pdf")
    assert pdf_path.exists()
    assert pdf_path.is_file()


# 20. PDF is non-empty
def test_20_pdf_is_non_empty(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_pdf_nonempty.html", generate_pdf=True)
    pdf_path = report_path.with_suffix(".pdf")
    assert pdf_path.stat().st_size > 1000


# 21. Invalid case analysis fails gracefully
def test_21_invalid_case_analysis_fails_gracefully():
    # Incomplete dictionary with empty keys should not raise unhandled exception
    empty_analysis = {"case_id": "CT-EMPTY", "case_data": {}, "prediction": {}, "ranking": {}}
    report_path = generate_executive_report(empty_analysis, output_filename="test_p9_empty.html")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "CT-EMPTY" in content


# 22. Missing ranked ATMs are handled
def test_22_missing_ranked_atms_are_handled(jalandhar_analysis):
    no_atm_analysis = dict(jalandhar_analysis)
    no_atm_analysis["ranking"] = {"candidate_zones": ["Zone_02"], "cross_zone_search": False, "ranked_atms": []}
    report_path = generate_executive_report(no_atm_analysis, output_filename="test_p9_no_atms.html")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Zero OSM ATM Records Found" in content or "No verified OSM ATMs" in content


# 23. Missing map data is handled
def test_23_missing_map_data_is_handled(jalandhar_analysis):
    no_map_analysis = dict(jalandhar_analysis)
    no_map_analysis.pop("map_data", None)
    report_path = generate_executive_report(no_map_analysis, output_filename="test_p9_no_map.html")
    assert report_path.exists()


# 24. Panipat/zero-ATM case is handled
def test_24_panipat_zero_atm_case_is_handled(panipat_analysis):
    report_path = generate_executive_report(panipat_analysis, output_filename="test_p9_panipat.html")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Panipat" in content


# 25. Cross-zone report section appears when active
def test_25_cross_zone_report_section_appears_when_active(gurugram_analysis):
    report_path = generate_executive_report(gurugram_analysis, output_filename="test_p9_cross_active.html")
    content = report_path.read_text(encoding="utf-8")
    if gurugram_analysis["ranking"].get("cross_zone_search"):
        assert "Cross-Zone Candidate Assessment Active" in content


# 26. Cross-zone section does not appear when inactive
def test_26_cross_zone_section_does_not_appear_when_inactive(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_cross_inactive.html")
    content = report_path.read_text(encoding="utf-8")
    if not jalandhar_analysis["ranking"].get("cross_zone_search"):
        assert "Cross-Zone Candidate Assessment Active" not in content


# 27. New case invalidates old report
def test_27_new_case_invalidates_old_report():
    case_a = analyze_case({"city": "Jalandhar", "amount": 15000, "bank": "HDFC", "complaint_id": "CASE-A"})
    case_b = analyze_case({"city": "Ludhiana", "amount": 25000, "bank": "SBI", "complaint_id": "CASE-B"})
    
    rep_a = generate_executive_report(case_a, output_filename="test_p9_case_a.html")
    rep_b = generate_executive_report(case_b, output_filename="test_p9_case_b.html")
    
    content_a = rep_a.read_text(encoding="utf-8")
    content_b = rep_b.read_text(encoding="utf-8")
    
    assert "CASE-A" in content_a
    assert "CASE-B" not in content_a
    assert "CASE-B" in content_b
    assert "CASE-A" not in content_b


# 28. Report output is deterministic for identical inputs
def test_28_report_output_is_deterministic_for_identical_inputs(jalandhar_analysis):
    d1 = prepare_report_data(jalandhar_analysis)
    d2 = prepare_report_data(jalandhar_analysis)
    assert d1["case_id"] == d2["case_id"]
    assert d1["prediction"]["predicted_zone"] == d2["prediction"]["predicted_zone"]
    assert d1["ranked_atms"][0]["atm_id"] == d2["ranked_atms"][0]["atm_id"]
    assert d1["ranked_atms"][0]["overall_score"] == d2["ranked_atms"][0]["overall_score"]


# 29. No hidden cash-out coordinates appear
def test_29_no_hidden_cashout_coordinates_appear(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_leakage.html")
    content = report_path.read_text(encoding="utf-8")
    assert "cashout_lat" not in content.lower()
    assert "cashout_lon" not in content.lower()
    assert "synthetic_cashout" not in content.lower()


# 30. No fraud activity claims are generated from synthetic activity
def test_30_no_fraud_activity_claims_are_generated_from_synthetic_activity(jalandhar_analysis):
    report_path = generate_executive_report(jalandhar_analysis, output_filename="test_p9_claims.html")
    content = report_path.read_text(encoding="utf-8")
    assert "confirmed criminal" not in content.lower()
    assert "fraud atm" not in content.lower()
    assert "confirmed withdrawal atm" not in content.lower()
