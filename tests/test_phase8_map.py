"""Comprehensive Unit and Verification Test Suite for CyberTrace Phase 8:
Interactive Leaflet / OpenStreetMap Geospatial Intelligence Map Pipeline.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

from src.config import ATM_LOCATIONS_PROCESSED_PATH
from src.geographic.map_data import (
    load_zone_bounding_boxes,
    calculate_geographic_bounds,
    prepare_map_data,
)
from src.geographic.zones import ZONE_DEFINITIONS
from src.intelligence.case_analysis import analyze_case


@pytest.fixture(scope="module")
def sample_analysis_jalandhar():
    """Execute live case analysis for Jalandhar (single-zone high confidence)."""
    return analyze_case({
        "complaint_id": "TEST_MAP_JAL",
        "amount": 25000,
        "bank": "HDFC Bank",
        "city": "Jalandhar",
        "hour": 14,
    })


@pytest.fixture(scope="module")
def sample_analysis_ncr():
    """Execute live case analysis for Noida (medium confidence / cross-zone candidate search)."""
    return analyze_case({
        "complaint_id": "TEST_MAP_NCR",
        "amount": 40000,
        "bank": "State Bank of India",
        "city": "Noida",
        "hour": 23,
    })


@pytest.fixture(scope="module")
def sample_analysis_panipat():
    """Execute live case analysis for Panipat (Zone_05, 0 OSM ATMs edge case)."""
    return analyze_case({
        "complaint_id": "TEST_MAP_PAN",
        "amount": 20000,
        "bank": "Punjab National Bank",
        "city": "Panipat",
        "hour": 10,
    })


def test_01_map_contract_is_valid(sample_analysis_jalandhar):
    """1. Verify that prepare_map_data outputs a complete, validated contract dictionary."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    assert isinstance(map_data, dict)
    required_keys = [
        "case_id",
        "model_id",
        "complaint_point",
        "predicted_zone",
        "candidate_zones",
        "zone_probabilities",
        "ranked_atms",
        "cross_zone_search",
        "confidence_tier",
        "map_bounds",
        "map_center",
        "map_zoom",
        "osm_attribution",
    ]
    for key in required_keys:
        assert key in map_data, f"Missing required key '{key}' in map data contract!"


def test_02_complaint_coordinates_are_valid(sample_analysis_jalandhar):
    """2. Verify complaint origin coordinates are bounded [-90, 90] and [-180, 180]."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    cp = map_data["complaint_point"]
    assert cp is not None
    assert -90.0 <= cp["latitude"] <= 90.0
    assert -180.0 <= cp["longitude"] <= 180.0
    assert "Complaint Location" in cp["label"]


def test_03_predicted_zone_exists(sample_analysis_jalandhar):
    """3. Verify predicted zone is identified and has bounding box."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    pz = map_data["predicted_zone"]
    assert pz["zone_id"].startswith("Zone_")
    assert 0.0 <= pz["probability"] <= 100.0
    assert "bbox" in pz
    b = pz["bbox"]
    assert b["min_lat"] < b["max_lat"]
    assert b["min_lon"] < b["max_lon"]


def test_04_candidate_zones_exist_in_project_zone_definitions(sample_analysis_jalandhar):
    """4. Verify candidate zones are members of recognized canonical zones."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    valid_zones = [f"Zone_{i:02d}" for i in range(1, 11)]
    for cz in map_data["candidate_zones"]:
        assert cz["zone_id"] in valid_zones


def test_05_all_atm_markers_contain_valid_coordinates(sample_analysis_jalandhar):
    """5. Verify every candidate ATM marker has non-null, valid coordinates."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    for atm in map_data["ranked_atms"]:
        assert -90.0 <= atm["latitude"] <= 90.0
        assert -180.0 <= atm["longitude"] <= 180.0


def test_06_ranked_atm_ids_exist_in_the_atm_dataset(sample_analysis_jalandhar):
    """6. Verify all visualized ATM IDs exist in the verified OSM dataset."""
    atms_df = pd.read_csv(ATM_LOCATIONS_PROCESSED_PATH)
    valid_ids = set(atms_df["atm_id"].unique())
    map_data = prepare_map_data(sample_analysis_jalandhar)
    for atm in map_data["ranked_atms"]:
        assert atm["atm_id"] in valid_ids


def test_07_top_candidate_is_rank_1(sample_analysis_jalandhar):
    """7. Verify the first candidate in ranked_atms is marked as rank 1 and is_top True."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    atms = map_data["ranked_atms"]
    if atms:
        assert atms[0]["rank"] == 1
        assert atms[0]["is_top"] is True
        assert atms[0]["marker_color"] == "#10b981"  # Emerald green


def test_08_atm_markers_preserve_ranking(sample_analysis_jalandhar):
    """8. Verify candidate ATM markers preserve strict score ordering."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    atms = map_data["ranked_atms"]
    scores = [a["overall_score"] for a in atms]
    assert scores == sorted(scores, reverse=True)


def test_09_all_10_zone_probabilities_are_preserved(sample_analysis_jalandhar):
    """9. Verify all 10 canonical withdrawal zone probabilities are retained."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    probs = map_data["zone_probabilities"]
    assert len(probs) == 10
    expected = [f"Zone_{i:02d}" for i in range(1, 11)]
    for z in expected:
        assert z in probs
        assert 0.0 <= probs[z] <= 1.0


def test_10_candidate_zone_probabilities_are_valid(sample_analysis_jalandhar):
    """10. Verify candidate zone probability values are percentages in [0.0, 100.0]."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    for cz in map_data["candidate_zones"]:
        assert 0.0 <= cz["probability"] <= 100.0


def test_11_map_bounds_can_be_generated():
    """11. Verify calculate_geographic_bounds generates bounding box, center, and zoom."""
    sample_points = [(31.32, 75.57), (31.62, 74.87)]
    bounds, center, zoom = calculate_geographic_bounds(sample_points)
    assert len(bounds) == 2
    assert bounds[0][0] < bounds[1][0]
    assert bounds[0][1] < bounds[1][1]
    assert len(center) == 2
    assert 5 <= zoom <= 15


def test_12_single_zone_case_produces_valid_bounds(sample_analysis_jalandhar):
    """12. Verify single-zone case produces valid tight bounding box around the city."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    bounds = map_data["map_bounds"]
    assert len(bounds) == 2
    lat_span = bounds[1][0] - bounds[0][0]
    lon_span = bounds[1][1] - bounds[0][1]
    assert 0.1 <= lat_span <= 2.0
    assert 0.1 <= lon_span <= 2.0


def test_13_cross_zone_case_produces_valid_bounds(sample_analysis_ncr):
    """13. Verify cross-zone case generates encompassing bounds covering both sectors."""
    map_data = prepare_map_data(sample_analysis_ncr)
    bounds = map_data["map_bounds"]
    assert len(bounds) == 2
    assert bounds[0][0] < bounds[1][0]
    assert bounds[0][1] < bounds[1][1]


def test_14_low_confidence_multi_zone_case_produces_valid_bounds():
    """14. Verify low-confidence synthesized case generates valid regional bounds."""
    mock_case = {
        "case_id": "MOCK_LOW",
        "model_id": "random_forest_full_none",
        "case_data": {"complaint_latitude": 28.6, "complaint_longitude": 77.2, "city": "Delhi"},
        "prediction": {
            "predicted_zone": "Zone_07",
            "prediction_confidence": 0.40,
            "confidence_tier": "LOW",
            "probability_margin": 0.05,
            "second_best_zone": "Zone_08",
            "zone_probabilities": {"Zone_07": 0.40, "Zone_08": 0.35, "Zone_06": 0.15, "Zone_05": 0.10},
        },
        "ranking": {
            "candidate_zones": ["Zone_07", "Zone_08", "Zone_06"],
            "cross_zone_search": True,
            "ranked_atms": [],
        },
    }
    map_data = prepare_map_data(mock_case)
    bounds = map_data["map_bounds"]
    assert len(bounds) == 2
    assert bounds[0][0] < bounds[1][0]


def test_15_missing_complaint_coordinates_are_handled():
    """15. Verify case with missing coordinates defaults gracefully without crashing."""
    mock_case = {
        "case_id": "MISSING_COORDS",
        "model_id": "random_forest_full_none",
        "case_data": {"complaint_latitude": None, "complaint_longitude": None, "city": "Unknown"},
        "prediction": {
            "predicted_zone": "Zone_01",
            "prediction_confidence": 0.90,
            "confidence_tier": "HIGH",
            "probability_margin": 0.85,
            "zone_probabilities": {"Zone_01": 0.90},
        },
        "ranking": {"candidate_zones": ["Zone_01"], "ranked_atms": []},
    }
    map_data = prepare_map_data(mock_case)
    assert map_data["complaint_point"] is None
    assert len(map_data["map_bounds"]) == 2


def test_16_empty_atm_candidate_list_is_handled():
    """16. Verify map data handles zero candidate ATMs cleanly."""
    mock_case = {
        "case_id": "EMPTY_ATMS",
        "model_id": "random_forest_full_none",
        "case_data": {"complaint_latitude": 31.32, "complaint_longitude": 75.57, "city": "Jalandhar"},
        "prediction": {"predicted_zone": "Zone_02", "zone_probabilities": {"Zone_02": 1.0}},
        "ranking": {"candidate_zones": ["Zone_02"], "ranked_atms": []},
    }
    map_data = prepare_map_data(mock_case)
    assert map_data["ranked_atms"] == []
    assert map_data["total_atms_visible"] == 0


def test_17_panipat_zero_atm_case_does_not_crash(sample_analysis_panipat):
    """17. Verify Panipat (Zone_05, 0 OSM ATMs) prepares map data cleanly without errors."""
    map_data = prepare_map_data(sample_analysis_panipat)
    assert map_data["case_id"] == "TEST_MAP_PAN"
    assert map_data["predicted_zone"]["zone_id"] == "Zone_05"
    assert len(map_data["map_bounds"]) == 2


def test_18_no_hidden_cashout_coordinates_are_introduced(sample_analysis_jalandhar):
    """18. Verify no forbidden hidden cash-out coordinates enter map data."""
    forbidden = ["synthetic_cashout_latitude", "synthetic_cashout_longitude", "cashout_latitude"]
    for k in sample_analysis_jalandhar["case_data"]:
        assert k not in forbidden
    map_data = prepare_map_data(sample_analysis_jalandhar)
    for k in map_data:
        assert k not in forbidden


def test_19_no_atm_ranking_logic_is_duplicated_in_the_map_layer(sample_analysis_jalandhar):
    """19. Verify map data merely extracts scores already computed by Phase 7."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    phase7_scores = [a["overall_score"] for a in sample_analysis_jalandhar["ranking"]["ranked_atms"][:10]]
    map_scores = [a["overall_score"] for a in map_data["ranked_atms"]]
    assert phase7_scores == map_scores


def test_20_osm_attribution_is_present_in_the_map_configuration(sample_analysis_jalandhar):
    """20. Verify required OpenStreetMap copyright attribution is present."""
    map_data = prepare_map_data(sample_analysis_jalandhar)
    assert "OpenStreetMap" in map_data["osm_attribution"]
    assert "copyright" in map_data["osm_attribution"].lower()
