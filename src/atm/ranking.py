"""ATM Candidate Ranking Engine for CyberTrace.

Calculates multi-criteria candidate scores based on:
- Distance from complaint / zone centroid
- Bank compatibility matching
- Historical synthetic activity
- 24x7 operational availability
- Temporal compatibility

NOTE: All scores are analytical rankings. CyberTrace does NOT confirm that any
candidate ATM was actually used. Evidence requires banking audit records.
"""

from typing import List, Dict, Any, Optional
import math
from src.utils.logging import get_logger

logger = get_logger("ATMRanking")

def calculate_candidate_score(
    distance_km: float,
    atm_bank: str,
    complaint_bank: Optional[str] = None,
    is_24x7: bool = True,
    historical_fraud_count: int = 0,
    is_night: bool = False,
) -> float:
    """Compute normalized candidate score between 0.0 and 1.0."""
    # 1. Proximity score (exponential decay over distance in km)
    proximity_score = math.exp(-0.15 * max(0.0, distance_km))

    # 2. Bank compatibility score
    bank_score = 0.5
    if complaint_bank and atm_bank:
        if complaint_bank.lower() in atm_bank.lower() or atm_bank.lower() in complaint_bank.lower():
            bank_score = 1.0
        else:
            bank_score = 0.35

    # 3. Operational accessibility score
    access_score = 1.0
    if is_night and not is_24x7:
        access_score = 0.25

    # 4. Historical activity signal (capped influence)
    activity_score = min(1.0, historical_fraud_count / 10.0)

    # Weighted composite score
    composite = (
        0.40 * proximity_score
        + 0.30 * bank_score
        + 0.15 * access_score
        + 0.15 * activity_score
    )

    return round(float(composite), 3)

def rank_atm_candidates(
    candidates: List[Dict[str, Any]],
    complaint_bank: Optional[str] = None,
    complaint_hour: Optional[int] = None,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Rank candidate ATMs within a predicted withdrawal zone.

    Returns ranked candidates sorted descending by candidate_score.
    """
    if not candidates:
        return []

    is_night = False
    if complaint_hour is not None:
        is_night = (complaint_hour >= 22 or complaint_hour <= 5)

    ranked_list = []
    for cand in candidates:
        dist = float(cand.get("distance_km", 5.0))
        bank = str(cand.get("bank", "Unknown"))
        is_247 = bool(cand.get("is_24x7", True))
        hist_fraud = int(cand.get("historical_fraud_withdrawals", 0))

        score = calculate_candidate_score(
            distance_km=dist,
            atm_bank=bank,
            complaint_bank=complaint_bank,
            is_24x7=is_247,
            historical_fraud_count=hist_fraud,
            is_night=is_night,
        )

        record = {
            "atm_id": cand.get("atm_id", "Unknown"),
            "bank": bank,
            "operator": cand.get("operator", bank),
            "distance_km": round(dist, 2),
            "candidate_score": score,
            "address": cand.get("address", "N/A"),
            "latitude": cand.get("latitude"),
            "longitude": cand.get("longitude"),
            "source": cand.get("source", "OpenStreetMap"),
            "designation": "Candidate",
        }
        ranked_list.append(record)

    ranked_list.sort(key=lambda x: x["candidate_score"], reverse=True)

    if ranked_list:
        ranked_list[0]["designation"] = "Highest-ranked candidate"

    logger.info(f"Ranked {len(ranked_list)} candidate ATMs. Top score: {ranked_list[0]['candidate_score'] if ranked_list else 0}")
    return ranked_list[:top_n]
