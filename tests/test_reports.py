"""Unit tests for non-technical executive intelligence report generation."""

import pytest
from pathlib import Path
from src.intelligence.explanation import generate_investigative_explanation
from src.intelligence.report import generate_executive_report

def test_explanation_non_technical_vocabulary():
    """Verify explanations avoid technical ML jargon and use proper disclaimers."""
    case = {"amount": 25000, "bank": "HDFC", "city": "Jalandhar", "hour": 14}
    pred = {"predicted_zone": "Zone_04", "confidence": 0.814}
    atms = [{"atm_id": "ATM-01", "bank": "HDFC", "distance_km": 1.2, "candidate_score": 0.88, "designation": "Highest-ranked candidate"}]

    explanation = generate_investigative_explanation(case, pred, atms)
    assert "Zone_04" in explanation["zone_summary"]
    assert "highest-probability" in explanation["zone_summary"]
    # Ensure no raw jargon like 'RandomForestClassifier' or 'class 4'
    assert "Random Forest predicted" not in explanation["zone_summary"]
    assert "evidence" in explanation["evidence_disclaimer"].lower()

def test_generate_report_file_creation(tmp_path):
    """Verify report generation creates an HTML file with required sections."""
    case = {
        "complaint_id": "TEST-CASE-99",
        "amount": 35000.0,
        "amount_category": "High (25k-50k)",
        "bank": "HDFC",
        "transaction_type": "UPI",
        "fraud_type": "Phishing",
        "city": "Jalandhar",
        "district": "Jalandhar",
        "state": "Punjab",
    }
    pred = {"predicted_zone": "Zone_01", "confidence": 0.85, "top_predictions": []}
    atms = [{"atm_id": "ATM001", "bank": "HDFC", "distance_km": 1.5, "candidate_score": 0.9, "designation": "Highest-ranked candidate"}]

    out_file = "test_exec_report.html"
    report_path = generate_executive_report(case, pred, atms, output_filename=out_file)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "CYBERTRACE EXECUTIVE INTELLIGENCE BRIEF" in content
    assert "TEST-CASE-99" in content
    assert "Zone_01" in content
    assert "ACADEMIC SYNTHETIC RESEARCH NOTICE" in content
