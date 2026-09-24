"""Comprehensive Verification and Unit Test Suite for CyberTrace Phase 7:
ATM Candidate Ranking Engine & Dynamic Live Investigation Pipeline.
"""

import math
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ATM_LOCATIONS_PROCESSED_PATH,
    ATM_ACTIVITY_PROCESSED_PATH,
    ATM_RANKING_WEIGHTS,
    CANDIDATE_SEARCH_CONFIG,
    PRIMARY_MODEL_ID,
    FALLBACK_MODEL_ID,
)
from src.atm.loader import (
    load_atm_locations,
    load_atm_activity,
    get_atm_activity_profile,
    load_zone_centroids,
)
from src.atm.ranking import (
    classify_bank_compatibility,
    classify_time_compatibility,
    calculate_amount_compatibility,
    calculate_spatial_score,
    calculate_zone_probability_score,
    select_candidate_zones,
    rank_atms_for_case,
    calculate_candidate_score,
)
from src.intelligence.case_analysis import (
    analyze_case,
    validate_and_normalize_case_input,
)


@pytest.fixture(scope="module")
def atms_data():
    """Load verified ATM locations DataFrame."""
    return load_atm_locations()


@pytest.fixture(scope="module")
def activity_data():
    """Load synthetic historical ATM activity DataFrame."""
    return load_atm_activity()


def test_01_atm_dataset_loads_correctly(atms_data):
    """1. Verify that verified ATM locations dataset loads and contains 333 records."""
    assert isinstance(atms_data, pd.DataFrame)
    assert len(atms_data) == 333
    assert "atm_id" in atms_data.columns
    assert "latitude" in atms_data.columns
    assert "longitude" in atms_data.columns
    assert "city" in atms_data.columns
    assert "source" in atms_data.columns


def test_02_atm_ids_are_unique(atms_data):
    """2. Verify that all ATM IDs are distinct and non-duplicated."""
    assert atms_data["atm_id"].nunique() == len(atms_data)
    assert not atms_data["atm_id"].duplicated().any()


def test_03_activity_dataset_loads_correctly(activity_data):
    """3. Verify that synthetic historical ATM activity dataset loads with expected shape."""
    assert isinstance(activity_data, pd.DataFrame)
    assert len(activity_data) == 719280
    assert "activity_score" in activity_data.columns
    assert "average_amount" in activity_data.columns
    assert "hour" in activity_data.columns


def test_04_activity_has_no_forbidden_fraud_fields(activity_data):
    """4. Verify that ATM activity contains zero fraud-specific or complaint target variables."""
    forbidden_terms = ["fraud", "withdrawal_zone", "complaint", "victim", "synthetic_cashout"]
    for col in activity_data.columns:
        for term in forbidden_terms:
            assert term not in col.lower(), f"Forbidden term '{term}' found in column '{col}'!"


def test_05_ranking_weights_sum_to_one():
    """5. Verify that scoring weights sum strictly to 1.0."""
    weight_sum = sum(ATM_RANKING_WEIGHTS.values())
    assert abs(weight_sum - 1.0) < 1e-6
    # Ensure all components have positive weights
    for component, weight in ATM_RANKING_WEIGHTS.items():
        assert weight > 0.0, f"Component '{component}' must have positive weight"


def test_06_score_components_remain_within_documented_ranges():
    """6. Verify all individual score components produce values in [0.0, 100.0]."""
    # Spatial score
    assert 0.0 <= calculate_spatial_score(0.0) <= 100.0
    assert 0.0 <= calculate_spatial_score(50.0) <= 100.0
    assert 0.0 <= calculate_spatial_score(500.0) <= 100.0

    # Zone probability score
    probs = {"Zone_01": 0.85, "Zone_02": 0.15}
    assert 0.0 <= calculate_zone_probability_score(probs, "Zone_01", "Zone_02") <= 100.0
    assert 0.0 <= calculate_zone_probability_score(probs, "Zone_03", "Zone_04") <= 100.0

    # Bank score
    for bank in ["HDFC Bank", "SBI", "unknown", "Axis Bank"]:
        _, b_score = classify_bank_compatibility("HDFC Bank", bank)
        assert 0.0 <= b_score <= 100.0

    # Time score
    for is_247 in [True, False, None]:
        for h in [2, 14, None]:
            _, t_score = classify_time_compatibility(is_247, h)
            assert 0.0 <= t_score <= 100.0

    # Amount compatibility score
    profile = {"average_amount": 3500.0, "high_value_withdrawal_count": 1.0, "estimated_cash_volume": 20000.0}
    for amt in [2000.0, 15000.0, 50000.0, -100.0]:
        _, a_score = calculate_amount_compatibility(amt, profile)
        assert 0.0 <= a_score <= 100.0


def test_07_overall_score_remains_within_documented_range(atms_data):
    """7. Verify that overall candidate score is bounded strictly in [0.0, 100.0]."""
    case = {
        "complaint_id": "TEST_CASE",
        "amount": 35000.0,
        "bank": "HDFC Bank",
        "hour": 14,
    }
    pred = {
        "predicted_zone": "Zone_01",
        "confidence_tier": "HIGH",
        "probability_margin": 0.90,
        "second_best_zone": "Zone_02",
        "zone_probabilities": {"Zone_01": 0.95, "Zone_02": 0.05},
    }
    res = rank_atms_for_case(case, pred, atms_df=atms_data, top_n=10)
    for cand in res["ranked_atms"]:
        assert 0.0 <= cand["overall_score"] <= 100.0


def test_08_exact_bank_match_receives_higher_bank_compatibility_than_mismatch():
    """8. Verify exact bank match scores higher than mismatch."""
    _, s_exact = classify_bank_compatibility("HDFC Bank", "HDFC Bank")
    _, s_mismatch = classify_bank_compatibility("HDFC Bank", "State Bank of India")
    assert s_exact == 100.0
    assert s_mismatch == 20.0
    assert s_exact > s_mismatch


def test_09_unknown_bank_does_not_receive_a_fabricated_match():
    """9. Verify unknown bank receives neutral score without fabricated match."""
    label1, s_unk1 = classify_bank_compatibility("HDFC Bank", "unknown")
    label2, s_unk2 = classify_bank_compatibility("unknown", "ICICI Bank")
    assert label1 == "UNKNOWN"
    assert label2 == "UNKNOWN"
    assert s_unk1 == 50.0
    assert s_unk2 == 50.0


def test_10_higher_zone_probability_increases_zone_contribution():
    """10. Verify higher zone probability yields strictly higher zone_probability_score."""
    probs_high = {"Zone_07": 0.90, "Zone_08": 0.10}
    probs_low = {"Zone_07": 0.20, "Zone_08": 0.80}

    s_high = calculate_zone_probability_score(probs_high, "Zone_07")
    s_low = calculate_zone_probability_score(probs_low, "Zone_07")
    assert s_high > s_low


def test_11_closer_candidate_receives_stronger_spatial_contribution():
    """11. Verify proximity decay awards higher spatial score to closer ATMs."""
    score_close = calculate_spatial_score(1.5)
    score_far = calculate_spatial_score(35.0)
    assert score_close > score_far
    assert score_close > 0.0


def test_12_cross_zone_candidates_are_included_for_medium_confidence():
    """12. Verify medium-confidence cases activate cross-zone search and include second zone."""
    pred = {
        "predicted_zone": "Zone_07",
        "second_best_zone": "Zone_08",
        "probability_margin": 0.18,
        "confidence_tier": "MEDIUM",
        "zone_probabilities": {"Zone_07": 0.58, "Zone_08": 0.40, "Zone_06": 0.02},
    }
    candidate_zones, cross_zone_active = select_candidate_zones(pred)
    assert cross_zone_active is True
    assert "Zone_07" in candidate_zones
    assert "Zone_08" in candidate_zones


def test_13_high_confidence_cases_restrict_candidate_zones_appropriately():
    """13. Verify high-confidence predictions restrict search to primary zone."""
    pred = {
        "predicted_zone": "Zone_01",
        "second_best_zone": "Zone_02",
        "probability_margin": 0.95,
        "confidence_tier": "HIGH",
        "zone_probabilities": {"Zone_01": 0.98, "Zone_02": 0.02},
    }
    candidate_zones, cross_zone_active = select_candidate_zones(pred)
    assert cross_zone_active is False
    assert candidate_zones == ["Zone_01"]


def test_14_low_confidence_cases_include_sufficient_probability_mass():
    """14. Verify low-confidence predictions accumulate zones up to 80% probability mass."""
    pred = {
        "predicted_zone": "Zone_07",
        "second_best_zone": "Zone_08",
        "probability_margin": 0.05,
        "confidence_tier": "LOW",
        "zone_probabilities": {
            "Zone_07": 0.35,
            "Zone_08": 0.30,
            "Zone_06": 0.20,
            "Zone_05": 0.10,
            "Zone_04": 0.05,
        },
    }
    candidate_zones, cross_zone_active = select_candidate_zones(pred)
    assert cross_zone_active is True
    assert len(candidate_zones) >= 3
    # Sum of probabilities in candidate zones should be >= 0.80
    total_p = sum(pred["zone_probabilities"][z] for z in candidate_zones)
    assert total_p >= 0.80


def test_15_top_n_ranking_is_correctly_sorted(atms_data):
    """15. Verify candidate list is sorted in descending order of overall_score."""
    case = {"bank": "SBI", "amount": 20000, "hour": 14}
    pred = {
        "predicted_zone": "Zone_08",
        "confidence_tier": "HIGH",
        "probability_margin": 0.85,
        "second_best_zone": "Zone_07",
        "zone_probabilities": {"Zone_08": 0.92, "Zone_07": 0.08},
    }
    res = rank_atms_for_case(case, pred, atms_df=atms_data, top_n=10)
    candidates = res["ranked_atms"]
    assert len(candidates) > 0
    scores = [c["overall_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert candidates[0]["rank"] == 1


def test_16_ranking_is_deterministic_for_identical_inputs(atms_data):
    """16. Verify ranking produces identical ordering and scores for identical inputs."""
    case = {"bank": "HDFC Bank", "amount": 50000, "hour": 23, "complaint_id": "DET_01"}
    pred = {
        "predicted_zone": "Zone_08",
        "confidence_tier": "MEDIUM",
        "probability_margin": 0.20,
        "second_best_zone": "Zone_07",
        "zone_probabilities": {"Zone_08": 0.60, "Zone_07": 0.40},
    }
    res1 = rank_atms_for_case(case, pred, atms_df=atms_data, top_n=10)
    res2 = rank_atms_for_case(case, pred, atms_df=atms_data, top_n=10)

    assert len(res1["ranked_atms"]) == len(res2["ranked_atms"])
    for c1, c2 in zip(res1["ranked_atms"], res2["ranked_atms"]):
        assert c1["atm_id"] == c2["atm_id"]
        assert c1["overall_score"] == c2["overall_score"]
        assert c1["rank"] == c2["rank"]


def test_17_no_hidden_cashout_coordinates_are_used():
    """17. Verify no hidden cash-out coordinates enter feature extraction or ranking."""
    case = {
        "complaint_id": "LEAK_CHECK",
        "synthetic_cashout_latitude": 28.55,
        "synthetic_cashout_longitude": 77.35,
        "cashout_latitude": 28.55,
        "amount": 20000,
        "city": "Jalandhar",
    }
    cleaned = validate_and_normalize_case_input(case)
    assert "synthetic_cashout_latitude" not in cleaned
    assert "synthetic_cashout_longitude" not in cleaned
    assert "cashout_latitude" not in cleaned


def test_18_no_complaint_target_leakage_enters_atm_activity(activity_data):
    """18. Verify ATM activity has no complaint ID or target column."""
    assert "complaint_id" not in activity_data.columns
    assert "withdrawal_zone" not in activity_data.columns


def test_19_analyze_case_returns_the_complete_contract():
    """19. Verify analyze_case returns all required keys conforming to contract."""
    res = analyze_case({
        "complaint_id": "CT-CONTRACT-TEST",
        "amount": 30000,
        "bank": "HDFC Bank",
        "city": "Amritsar",
        "transaction_type": "UPI",
        "fraud_type": "OTP Fraud",
    })
    assert res["status"] == "SUCCESS"
    assert "case_id" in res
    assert "prediction" in res
    assert "ranking" in res
    assert "explanation" in res
    assert "provenance" in res

    pred = res["prediction"]
    assert "predicted_zone" in pred
    assert "prediction_confidence" in pred
    assert "second_best_zone" in pred
    assert "probability_margin" in pred
    assert "confidence_tier" in pred
    assert "zone_probabilities" in pred

    rank = res["ranking"]
    assert "candidate_zones" in rank
    assert "cross_zone_search" in rank
    assert "ranked_atms" in rank


def test_20_all_10_zone_probabilities_are_preserved():
    """20. Verify that all 10 canonical withdrawal zones exist in predicted probabilities."""
    res = analyze_case({
        "complaint_id": "PROB_10_TEST",
        "amount": 10000,
        "bank": "SBI",
        "city": "Ludhiana",
    })
    probs = res["prediction"]["zone_probabilities"]
    assert len(probs) == 10
    expected_zones = [f"Zone_{i:02d}" for i in range(1, 11)]
    for z in expected_zones:
        assert z in probs
        assert 0.0 <= probs[z] <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_21_explanations_do_not_claim_confirmation():
    """21. Verify explanations strictly avoid claiming confirmed cash-out location."""
    res = analyze_case({
        "complaint_id": "DISCLAIMER_TEST",
        "amount": 40000,
        "bank": "ICICI Bank",
        "city": "Chandigarh",
    })
    expl = res["explanation"]
    full_text = " ".join([
        expl.get("zone_summary", ""),
        expl.get("atm_summary", ""),
        expl.get("evidence_disclaimer", ""),
    ]).lower()

    assert "confirmed withdrawal location" not in full_text
    assert "confirmed atm" not in full_text
    assert "actual withdrawal location" not in full_text
    assert "evidence" in expl["evidence_disclaimer"].lower()


def test_22_panipat_with_no_osm_atm_data_is_handled_gracefully():
    """22. Verify Panipat (Zone_05, 0 OSM ATMs) completes without error."""
    res = analyze_case({
        "complaint_id": "PANIPAT_TEST",
        "amount": 25000,
        "bank": "Punjab National Bank",
        "city": "Panipat",
    })
    assert res["status"] == "SUCCESS"
    assert res["prediction"]["predicted_zone"] == "Zone_05"
    # Should complete without crashing and return evaluated ATMs from candidate zones
    assert len(res["ranking"]["ranked_atms"]) > 0


def test_23_missing_bank_operator_information_is_handled_gracefully():
    """23. Verify handling of missing bank/operator in both case input and ATM records."""
    res = analyze_case({
        "complaint_id": "UNKNOWN_BANK_TEST",
        "amount": 15000,
        "bank": "unknown",
        "city": "Amritsar",
    })
    assert res["status"] == "SUCCESS"
    for atm in res["ranking"]["ranked_atms"]:
        assert atm["bank_score"] in [50.0, 20.0, 70.0, 100.0]


def test_24_unknown_operating_hours_do_not_receive_fabricated_values():
    """24. Verify unknown operating hours receive neutral score without fabrication."""
    label, score = classify_time_compatibility(None, hour=23)
    assert label == "UNKNOWN"
    assert score == 50.0
