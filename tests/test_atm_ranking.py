"""Unit tests for ATM candidate scoring and ranking engine."""

import pytest
from src.atm.ranking import calculate_candidate_score, rank_atm_candidates

def test_calculate_candidate_score_bounds():
    """Verify candidate score produces normalized values between 0.0 and 1.0."""
    score = calculate_candidate_score(
        distance_km=1.5,
        atm_bank="HDFC Bank",
        complaint_bank="HDFC",
        is_24x7=True,
        historical_fraud_count=2,
        is_night=False,
    )
    assert 0.0 <= score <= 1.0

def test_closer_atm_gets_higher_score():
    """Verify proximity improves candidate ranking score all else equal."""
    score_close = calculate_candidate_score(distance_km=0.5, atm_bank="SBI", complaint_bank="SBI")
    score_far = calculate_candidate_score(distance_km=15.0, atm_bank="SBI", complaint_bank="SBI")
    assert score_close > score_far

def test_rank_atm_candidates_ordering():
    """Verify rank_atm_candidates sorts descending by candidate_score and labels top candidate."""
    candidates = [
        {"atm_id": "ATM_FAR", "bank": "SBI", "distance_km": 10.0, "is_24x7": True},
        {"atm_id": "ATM_CLOSE", "bank": "SBI", "distance_km": 0.8, "is_24x7": True},
    ]
    ranked = rank_atm_candidates(candidates, complaint_bank="SBI")
    assert len(ranked) == 2
    assert ranked[0]["atm_id"] == "ATM_CLOSE"
    assert ranked[0]["designation"] == "Highest-ranked candidate"
    assert ranked[0]["candidate_score"] >= ranked[1]["candidate_score"]
