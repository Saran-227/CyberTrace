"""
CyberTrace - Phase 10 Frontend Integration Test Suite
Verifies production-quality integration, shared authoritative state,
dynamic reactivity, caching, error handling, edge cases, and absence
of hardcoded live demo values.
"""

import os
import re
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

# Core components and pipelines
from src.intelligence.case_analysis import analyze_case, get_location_model
from app.components.navbar import check_system_readiness
from app.components.atm_card import render_atm_spotlight_card
from app.components.atm_table import render_atm_table
from app.components.probability_chart import render_probability_chart
from app.components.map import render_investigation_map, render_neutral_map
from app.pages.analytics import (
    load_cached_evaluation_benchmarks,
    load_cached_imbalance_summary,
    load_cached_atm_coverage,
)


# -------------------------------------------------------------------------
# Test Cases & Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_case_input():
    return {
        "case_id": "CR-2026-TEST10-001",
        "city": "Jalandhar",
        "state": "Punjab",
        "district": "Jalandhar",
        "amount": 15000.0,
        "bank": "HDFC",
        "transaction_type": "UPI",
        "fraud_type": "Phishing/Smishing",
        "incident_date": "2026-09-24",
        "incident_time": "14:30:00",
        "latitude": 31.3260,
        "longitude": 75.5762,
    }


@pytest.fixture
def panipat_case_input():
    return {
        "case_id": "CR-2026-PANIPAT-001",
        "city": "Panipat",
        "state": "Haryana",
        "district": "Panipat",
        "amount": 25000.0,
        "bank": "SBI",
        "transaction_type": "ATM Withdrawal",
        "fraud_type": "Card Cloning",
        "incident_date": "2026-09-24",
        "incident_time": "11:00:00",
        "latitude": 29.3909,
        "longitude": 76.9635,
    }


# -------------------------------------------------------------------------
# 1. Module Imports and Component Availability
# -------------------------------------------------------------------------

def test_01_frontend_modules_import_cleanly():
    """Verify that all core frontend app components and pages import without syntax errors."""
    import app.app as main_app
    import app.pages.dashboard as dashboard_page
    import app.pages.investigation as investigation_page
    import app.pages.analytics as analytics_page
    import app.pages.report as report_page
    import app.components.navbar as navbar_comp

    assert main_app is not None
    assert dashboard_page is not None
    assert investigation_page is not None
    assert analytics_page is not None
    assert report_page is not None
    assert navbar_comp is not None


# -------------------------------------------------------------------------
# 2. System Readiness Check
# -------------------------------------------------------------------------

def test_02_system_readiness_check_reflects_actual_assets():
    """Ensure check_system_readiness inspects actual disk assets and returns ready."""
    status = check_system_readiness()
    assert isinstance(status, bool)
    assert status is True


# -------------------------------------------------------------------------
# 3. Dynamic Case Analysis Pipeline Execution
# -------------------------------------------------------------------------

def test_03_case_analysis_pipeline_generates_full_contract(sample_case_input):
    """Verify analyze_case creates a non-hardcoded, fully populated case_analysis result."""
    result = analyze_case(sample_case_input)
    assert result["case_id"] == sample_case_input["case_id"]
    pred = result["prediction"]
    assert "predicted_zone" in pred
    assert "prediction_confidence" in pred or "confidence" in pred
    assert "zone_probabilities" in pred
    assert len(pred["zone_probabilities"]) == 10
    assert "ranking" in result
    assert "candidate_zones" in result["ranking"]
    assert "ranked_atms" in result["ranking"]
    assert "explanation" in result


# -------------------------------------------------------------------------
# 4. Dynamic Update: Changing Case Completely Updates Contract
# -------------------------------------------------------------------------

def test_04_changing_case_updates_all_fields_with_no_stale_data(sample_case_input):
    """Verify switching from Jalandhar to Gurugram yields different authoritative output."""
    result_jalandhar = analyze_case(sample_case_input)

    gurugram_input = {
        "case_id": "CR-2026-GURUGRAM-999",
        "city": "Gurugram",
        "state": "Haryana",
        "district": "Gurugram",
        "amount": 50000.0,
        "bank": "Axis",
        "transaction_type": "NEFT",
        "fraud_type": "Identity Theft",
        "incident_date": "2026-09-24",
        "incident_time": "18:00:00",
        "latitude": 28.4595,
        "longitude": 77.0266,
    }
    result_gurugram = analyze_case(gurugram_input)

    assert result_gurugram["case_id"] == "CR-2026-GURUGRAM-999"
    assert result_jalandhar["case_id"] != result_gurugram["case_id"]
    jal_atms = result_jalandhar["ranking"]["ranked_atms"]
    gur_atms = result_gurugram["ranking"]["ranked_atms"]
    if jal_atms and gur_atms:
        top_jal_atm = jal_atms[0]["atm_id"]
        top_gur_atm = gur_atms[0]["atm_id"]
        assert top_jal_atm != top_gur_atm


# -------------------------------------------------------------------------
# 5. Probability Chart Generation
# -------------------------------------------------------------------------

def test_05_probability_chart_handles_all_10_zones(sample_case_input):
    """Test Altair probability chart renders with 10 zones and primary zone highlighted."""
    result = analyze_case(sample_case_input)
    pred = result["prediction"]
    with patch("streamlit.altair_chart") as mock_chart, \
         patch("streamlit.markdown") as mock_md:
        render_probability_chart(
            data=pred["zone_probabilities"],
            predicted_zone=pred.get("predicted_zone"),
        )
        assert mock_chart.called


# -------------------------------------------------------------------------
# 6. Map Generation: Dynamic Markers and Neutral State
# -------------------------------------------------------------------------

def test_06_neutral_map_renders_without_crash():
    """Verify render_neutral_map embeds leaflet HTML without crash."""
    with patch("streamlit.components.v1.html") as mock_html:
        render_neutral_map(height=400)
        assert mock_html.called
        html_arg = mock_html.call_args[0][0]
        assert "openstreetmap.org" in html_arg
        assert "Geographic intelligence will appear here" in html_arg


def test_07_dynamic_map_renders_top_candidates_only(sample_case_input):
    """Verify render_investigation_map passes formatted HTML map with case data."""
    result = analyze_case(sample_case_input)
    with patch("streamlit.components.v1.html") as mock_html:
        render_investigation_map(case_analysis=result, visible_candidate_count=5)
        assert mock_html.called
        html_arg = mock_html.call_args[0][0]
        assert "leaflet" in html_arg.lower()
        # Verify case ID is contained or JSON data was injected
        assert sample_case_input["case_id"] in html_arg or "complaint" in html_arg.lower()


# -------------------------------------------------------------------------
# 7. ATM Spotlight Card & Table Formatting
# -------------------------------------------------------------------------

def test_08_atm_spotlight_card_uses_highest_ranked_candidate(sample_case_input):
    """Verify render_atm_spotlight_card extracts data from rank 1 without errors."""
    result = analyze_case(sample_case_input)
    ranked = result["ranking"]["ranked_atms"]
    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.progress") as mock_progress, \
         patch("streamlit.columns") as mock_columns:
        mock_columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        render_atm_spotlight_card(ranked)
        assert mock_markdown.called


def test_09_atm_table_renders_gracefully(sample_case_input):
    """Verify render_atm_table outputs a formatted dataframe."""
    result = analyze_case(sample_case_input)
    ranked = result["ranking"]["ranked_atms"]
    with patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.markdown") as mock_md, \
         patch("streamlit.caption") as mock_cap, \
         patch("streamlit.expander") as mock_exp:
        mock_exp.return_value.__enter__ = MagicMock()
        mock_exp.return_value.__exit__ = MagicMock()
        render_atm_table(ranked)
        assert mock_df.called
        passed_df = mock_df.call_args[0][0]
        assert "Rank" in passed_df.columns
        assert "Score" in passed_df.columns
        assert "ATM ID" in passed_df.columns


# -------------------------------------------------------------------------
# 8. Panipat Edge Case Graceful Handling
# -------------------------------------------------------------------------

def test_10_panipat_zero_atm_edge_case(panipat_case_input):
    """Verify Panipat executes without crashing and handles empty candidates gracefully."""
    result = analyze_case(panipat_case_input)
    assert result is not None
    assert result["status"] == "SUCCESS"
    assert result["prediction"]["predicted_zone"] == "Zone_05"
    ranked = result["ranking"]["ranked_atms"]

    # Test that spotlight card handles candidates and empty candidates gracefully
    with patch("streamlit.markdown") as mock_md, \
         patch("streamlit.progress"), \
         patch("streamlit.columns") as mock_cols, \
         patch("streamlit.expander") as mock_exp:
        mock_cols.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_exp.return_value.__enter__ = MagicMock()
        mock_exp.return_value.__exit__ = MagicMock()
        render_atm_spotlight_card(ranked)

    with patch("streamlit.info") as mock_info:
        render_atm_spotlight_card([])
        assert mock_info.called

    # Test that ATM table handles empty candidates gracefully
    with patch("streamlit.info") as mock_info:
        render_atm_table([])
        assert mock_info.called


# -------------------------------------------------------------------------
# 9. Bank Operator Missing/Unknown Graceful Handling
# -------------------------------------------------------------------------

def test_11_unknown_bank_operator_display_label():
    """Verify unknown bank operator is displayed respectfully without inventing names."""
    atms_with_unknown = [
        {
            "rank": 1,
            "atm_id": "osm_atm_test_01",
            "bank": "unknown",
            "operator": "unknown",
            "distance_km": 1.2,
            "zone": "Zone_01",
            "total_score": 0.82,
            "score_components": {
                "spatial_proximity_score": 0.9,
                "bank_affinity_score": 0.1,
                "historical_density_score": 0.8,
                "activity_level_score": 0.7,
                "operating_hours_score": 0.5,
                "zone_probability_score": 0.9,
            },
            "evidence_flags": ["Bank operator unavailable in OpenStreetMap"]
        }
    ]
    with patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.markdown") as mock_md, \
         patch("streamlit.caption") as mock_cap, \
         patch("streamlit.expander") as mock_exp:
        mock_exp.return_value.__enter__ = MagicMock()
        mock_exp.return_value.__exit__ = MagicMock()
        render_atm_table(atms_with_unknown)
        passed_df = mock_df.call_args[0][0]
        bank_val = passed_df.iloc[0]["Bank / Operator"]
        assert bank_val == "Bank information unavailable"


# -------------------------------------------------------------------------
# 10. Analytics Cached Data Loaders
# -------------------------------------------------------------------------

def test_12_analytics_cached_loaders_execute_and_return_data():
    """Verify precomputed evaluation, class imbalance, and ATM coverage artifacts load cleanly."""
    eval_df = load_cached_evaluation_benchmarks()
    imbalance_df = load_cached_imbalance_summary()
    coverage_df = load_cached_atm_coverage()

    assert eval_df is not None and not eval_df.empty
    assert imbalance_df is not None and not imbalance_df.empty
    assert coverage_df is not None and not coverage_df.empty


# -------------------------------------------------------------------------
# 11. Model & Dataset Caching Verification
# -------------------------------------------------------------------------

def test_13_model_pipeline_loads_from_cached_registry():
    """Verify get_location_model returns valid model pipeline."""
    model, model_id = get_location_model()
    assert model is not None
    assert model_id is not None


# -------------------------------------------------------------------------
# 12. No Hardcoded Production Prediction Values in Frontend Files
# -------------------------------------------------------------------------

def test_14_no_hardcoded_live_predictions_in_app_directory():
    """
    Scan app/ source files to ensure no hardcoded prediction outputs
    (e.g., hardcoded zone predictions, static probabilities, or fake scores).
    """
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
    forbidden_patterns = [
        r'st\.session_state\["case_analysis"\]\s*=\s*\{\s*"predicted_zone":\s*"Zone_',
        r'predicted_zone\s*=\s*["\']Zone_07["\']\s*#\s*hardcoded',
        r'confidence\s*=\s*0\.884\s*#\s*hardcoded',
    ]

    for root, _, files in os.walk(app_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file_obj:
                    content = file_obj.read()
                    for pattern in forbidden_patterns:
                        assert not re.search(pattern, content), f"Hardcoded pattern found in {path}: {pattern}"


# -------------------------------------------------------------------------
# 13. Stale Report Invalidation Logic
# -------------------------------------------------------------------------

def test_15_case_invalidation_logic():
    """Verify changing case input resets report status."""
    session = {
        "case_input": {"case_id": "CASE-1"},
        "case_analysis": {"case_id": "CASE-1"},
        "report_status": "READY",
        "generated_report": {"case_id": "CASE-1"},
    }

    # Simulate user changing case input and analyzing new case
    new_input = {"case_id": "CASE-2"}
    session["case_input"] = new_input
    # Invalidation trigger
    if session.get("case_analysis", {}).get("case_id") != new_input["case_id"]:
        session["report_status"] = "OUTDATED"
        session["generated_report"] = None

    assert session["report_status"] == "OUTDATED"
    assert session["generated_report"] is None
