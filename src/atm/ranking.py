"""ATM Candidate Ranking Engine for CyberTrace Phase 7.

Implements transparent, multi-criteria spatial candidate scoring uniting:
1. Withdrawal zone prediction likelihood from Phase 6 location classifier
2. Geographic proximity to candidate zone reference centroids
3. Victim bank vs OpenStreetMap ATM operator compatibility
4. Simulated historical ATM operational activity
5. Temporal operational compatibility (e.g. 24x7 availability)
6. Cash amount distribution compatibility

NOTE: All scores and rankings are probabilistic candidates for investigative triage.
CyberTrace NEVER confirms that any ATM was the actual cash-out location without
authoritative banking logs and surveillance evidence.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd

from src.config import (
    ATM_RANKING_WEIGHTS,
    CANDIDATE_SEARCH_CONFIG,
    REPORTS_DIR,
)
from src.atm.loader import (
    load_atm_locations,
    load_zone_centroids,
    get_atm_activity_profile,
    get_atm_zone_assignment,
)
from src.geographic.distance import haversine_distance
from src.utils.logging import get_logger

logger = get_logger("ATMRanking")


def normalize_bank_name(name: Optional[str]) -> str:
    """Normalize bank or operator string for robust, deterministic matching."""
    if not name:
        return "unknown"
    s = str(name).strip().lower()
    if s in ["", "none", "nan", "null", "unknown"]:
        return "unknown"

    # Remove standard corporate legal suffixes and punctuation
    for token in ["bank", "ltd", "limited", "corp", "corporation", "atm", "branch"]:
        s = s.replace(token, " ")
    s = s.replace(".", "").replace("-", " ").replace("_", " ").strip()
    # Normalize multiple whitespaces
    s = " ".join(s.split())

    # Standard Indian commercial banking synonym map
    synonyms = {
        "sbi": "state of india",
        "state of india": "state of india",
        "state": "state of india",
        "pnb": "punjab national",
        "punjab national": "punjab national",
        "bob": "of baroda",
        "of baroda": "of baroda",
        "baroda": "of baroda",
        "boi": "of india",
        "of india": "of india",
        "icici": "icici",
        "hdfc": "hdfc",
        "axis": "axis",
        "kotak": "kotak mahindra",
        "kotak mahindra": "kotak mahindra",
        "canara": "canara",
        "union": "union of india",
        "union of india": "union of india",
        "indusind": "indusind",
        "yes": "yes",
    }
    return synonyms.get(s, s)


def classify_bank_compatibility(
    complaint_bank: Optional[str],
    atm_bank: Optional[str],
    atm_operator: Optional[str] = None,
) -> Tuple[str, float]:
    """Classify bank relationship and return (classification_label, score_0_to_100).

    Behavior:
    - EXACT_MATCH: 100.0 (e.g. HDFC vs HDFC Bank)
    - PARTIAL_MATCH: 70.0 (e.g. State Bank vs State Bank of India)
    - UNKNOWN: 50.0 (neutral score when OSM bank/operator is unknown)
    - MISMATCH: 20.0 (negative ranking contribution when explicitly different banks)
    """
    c_norm = normalize_bank_name(complaint_bank)
    b_norm = normalize_bank_name(atm_bank)
    op_norm = normalize_bank_name(atm_operator)

    # If either complaint bank or ATM bank/operator is unknown
    if c_norm == "unknown" or (b_norm == "unknown" and op_norm == "unknown"):
        return "UNKNOWN", 50.0

    # Check exact match with either bank or operator
    if (c_norm == b_norm and b_norm != "unknown") or (c_norm == op_norm and op_norm != "unknown"):
        return "EXACT_MATCH", 100.0

    # Check partial match (substring containment of substantive root)
    if (
        (c_norm in b_norm or b_norm in c_norm) and b_norm != "unknown" and len(min(c_norm, b_norm)) >= 3
    ) or (
        (c_norm in op_norm or op_norm in c_norm) and op_norm != "unknown" and len(min(c_norm, op_norm)) >= 3
    ):
        return "PARTIAL_MATCH", 70.0

    # Explicitly different banks
    return "MISMATCH", 20.0


def classify_time_compatibility(
    is_24x7_parsed: Optional[bool],
    hour: Optional[int],
) -> Tuple[str, float]:
    """Classify temporal operational accessibility and return (label, score_0_to_100).

    Behavior:
    - If explicit is_24x7 is True: OPEN_COMPATIBLE (100.0)
    - If explicit is_24x7 is False and night (22:00 - 05:59): POSSIBLY_INCOMPATIBLE (25.0)
    - If explicit is_24x7 is False and daytime (06:00 - 21:59): OPEN_COMPATIBLE (85.0)
    - If operating information is unknown (None): UNKNOWN (50.0 neutral score)
    """
    if is_24x7_parsed is True:
        return "OPEN_COMPATIBLE", 100.0

    is_night = False
    if hour is not None:
        is_night = (hour >= 22 or hour <= 5)

    if is_24x7_parsed is False:
        if is_night:
            return "POSSIBLY_INCOMPATIBLE", 25.0
        return "OPEN_COMPATIBLE", 85.0

    # Unspecified operating hours in OSM
    return "UNKNOWN", 50.0


def calculate_amount_compatibility(
    complaint_amount: float,
    activity_profile: Dict[str, Any],
) -> Tuple[str, float]:
    """Evaluate compatibility between complaint amount and ATM simulated withdrawal history.

    Returns:
        (compatibility_tier, score_0_to_100)
    """
    avg_amt = float(activity_profile.get("average_amount", 3500.0))
    high_val_count = float(activity_profile.get("high_value_withdrawal_count", 0.0))
    vol = float(activity_profile.get("estimated_cash_volume", 15000.0))

    if complaint_amount <= 0:
        return "NEUTRAL", 50.0

    # High-value complaint (> 25k)
    if complaint_amount >= 25000:
        if high_val_count >= 0.5 or vol >= 20000:
            return "HIGH_VOLUME_COMPATIBLE", 90.0
        elif high_val_count > 0:
            return "MODERATE_VOLUME_COMPATIBLE", 75.0
        else:
            return "BELOW_TYPICAL_CAPACITY", 40.0

    # Moderate/standard withdrawal (5k - 25k)
    elif complaint_amount >= 5000:
        if avg_amt >= 2000 or vol >= 10000:
            return "STANDARD_COMPATIBLE", 85.0
        return "MODERATE_COMPATIBLE", 65.0

    # Low-value withdrawal (< 5k)
    else:
        return "LOW_VALUE_COMPATIBLE", 90.0


def calculate_spatial_score(distance_km: float) -> float:
    """Calculate proximity score based on exponential decay over distance (km) to zone center.

    Score is bounded strictly in [0.0, 100.0].
    """
    decay = math.exp(-max(0.0, distance_km) / 25.0)
    return round(float(np.clip(decay * 100.0, 0.0, 100.0)), 2)


def calculate_zone_probability_score(
    zone_probabilities: Dict[str, float],
    assigned_zone: str,
    secondary_zone: Optional[str] = None,
) -> float:
    """Calculate zone probability score (0-100) combining primary and secondary zone affinities."""
    p_primary = float(zone_probabilities.get(assigned_zone, 0.0))
    p_sec = float(zone_probabilities.get(secondary_zone, 0.0)) if secondary_zone else 0.0

    # Primary zone carries 85% weight, secondary zone carries 15% cross-boundary weight
    composite_p = (p_primary * 0.85) + (p_sec * 0.15)
    return round(float(np.clip(composite_p * 100.0, 0.0, 100.0)), 2)


def select_candidate_zones(
    prediction_result: Dict[str, Any],
    search_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], bool]:
    """Select candidate zones according to diagnostic confidence and probability mass rules.

    Rules:
    - HIGH CONFIDENCE (margin >= 0.30): predicted_zone as primary candidate zone
    - MEDIUM CONFIDENCE (0.15 <= margin < 0.30): predicted_zone + second_best_zone (cross_zone = True)
    - LOW CONFIDENCE (margin < 0.15): accumulate zones until cumulative probability reaches threshold (0.80)
    - Fallback: If selected zones contain 0 ATMs (e.g. Panipat Zone_05), include adjacent runner-up zone
    """
    cfg = search_config or CANDIDATE_SEARCH_CONFIG
    cum_threshold = cfg.get("cumulative_probability_threshold", 0.80)
    high_margin = cfg.get("high_confidence_margin", 0.30)
    med_margin = cfg.get("medium_confidence_margin", 0.15)

    pred_zone = prediction_result.get("predicted_zone", "Zone_01")
    second_zone = prediction_result.get("second_best_zone", "Zone_02")
    margin = float(prediction_result.get("probability_margin", 0.0))
    tier = str(prediction_result.get("confidence_tier", "HIGH")).upper()
    probs = prediction_result.get("zone_probabilities", {})

    sorted_zones = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    candidate_zones: List[str] = []
    cross_zone_search = False

    if margin >= high_margin and tier == "HIGH":
        candidate_zones = [pred_zone]
        cross_zone_search = False
    elif margin >= med_margin or tier == "MEDIUM":
        candidate_zones = [pred_zone, second_zone]
        cross_zone_search = True
    else:
        # Low confidence: select zones until reaching cumulative probability threshold
        accumulated_p = 0.0
        candidate_zones = []
        for z, p in sorted_zones:
            candidate_zones.append(z)
            accumulated_p += p
            if accumulated_p >= cum_threshold:
                break
        if len(candidate_zones) < 2 and second_zone not in candidate_zones:
            candidate_zones.append(second_zone)
        cross_zone_search = True

    # Deduplicate while preserving order
    deduped = []
    for z in candidate_zones:
        if z not in deduped:
            deduped.append(z)

    # Special handling for Zone_05 (Panipat, known 0 OSM ATMs)
    # If Zone_05 is the sole candidate zone, add second-best zone so candidates can be explored
    if deduped == ["Zone_05"]:
        if second_zone and second_zone != "Zone_05":
            deduped.append(second_zone)
            cross_zone_search = True

    return deduped, cross_zone_search


def rank_atms_for_case(
    case_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    atms_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Execute full Phase 7 transparent ATM candidate ranking for a complaint case.

    Parameters:
        case_data: Validated complaint metadata dictionary
        prediction_result: Phase 6 location model output dictionary conforming to contract
        atms_df: Optional preloaded ATM locations DataFrame
        top_n: Number of top candidate ATMs to return (default 10)
        weights: Optional custom ranking weights (must sum to 1.0)

    Returns:
        Structured dictionary conforming to docs/atm_ranking_contract.md
    """
    if atms_df is None:
        atms_df = load_atm_locations()

    scoring_weights = weights or ATM_RANKING_WEIGHTS
    weight_sum = sum(scoring_weights.values())
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(f"Ranking weights must sum strictly to 1.0 (received sum: {weight_sum:.4f})")

    # 1. Determine candidate zones & cross-zone status
    candidate_zones, cross_zone_active = select_candidate_zones(prediction_result)
    zone_probs = prediction_result.get("zone_probabilities", {})

    # 2. Filter ATMs belonging primarily or secondarily to candidate zones
    eligible_mask = (
        atms_df["assigned_zone"].isin(candidate_zones) |
        atms_df["secondary_zone"].isin(candidate_zones)
    )
    eligible_atms = atms_df[eligible_mask].copy()

    # If no ATMs in selected zones (e.g. edge-case Panipat without fallback), evaluate all study ATMs
    if len(eligible_atms) == 0:
        eligible_atms = atms_df.copy()

    complaint_bank = case_data.get("bank")
    complaint_amt = float(case_data.get("amount", 0.0))
    complaint_hour = case_data.get("hour")
    if complaint_hour is None and "complaint_time" in case_data:
        try:
            complaint_hour = int(str(case_data["complaint_time"]).split(":")[0])
        except Exception:
            complaint_hour = None

    centroids_df = load_zone_centroids()
    pred_zone = prediction_result.get("predicted_zone", "Zone_01")
    pred_centroid = centroids_df[centroids_df["withdrawal_zone"] == pred_zone]
    if len(pred_centroid) > 0:
        ref_lat = float(pred_centroid.iloc[0]["centroid_latitude"])
        ref_lon = float(pred_centroid.iloc[0]["centroid_longitude"])
    else:
        ref_lat, ref_lon = 28.554406, 77.349756

    ranked_records: List[Dict[str, Any]] = []

    for _, row in eligible_atms.iterrows():
        atm_id = str(row["atm_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        assigned_zone = str(row["assigned_zone"])
        sec_zone = str(row["secondary_zone"])
        atm_bank = str(row["bank"])
        operator = str(row["operator"])
        is_24x7 = row["is_24x7_parsed"]

        # A. Proximity to reference zone centroid
        dist_km = haversine_distance(lat, lon, ref_lat, ref_lon)

        # B. Component 1: Zone Probability Score (0-100)
        s_zone = calculate_zone_probability_score(zone_probs, assigned_zone, sec_zone)

        # C. Component 2: Spatial Proximity Score (0-100)
        s_spatial = calculate_spatial_score(dist_km)

        # D. Component 3: Bank Compatibility Score (0-100)
        bank_label, s_bank = classify_bank_compatibility(complaint_bank, atm_bank, operator)

        # E. Component 4: Temporal Accessibility Score (0-100)
        time_label, s_time = classify_time_compatibility(is_24x7, complaint_hour)

        # F. Component 5: Simulated Historical Activity Score (0-100)
        profile = get_atm_activity_profile(atm_id, hour=complaint_hour)
        s_activity = float(np.clip(profile.get("activity_score", 50.0), 0.0, 100.0))

        # G. Component 6: Amount Compatibility Score (0-100)
        amt_label, s_amt = calculate_amount_compatibility(complaint_amt, profile)

        # Weighted Composite Overall Score
        overall = (
            scoring_weights["zone_probability"] * s_zone
            + scoring_weights["spatial"] * s_spatial
            + scoring_weights["bank"] * s_bank
            + scoring_weights["activity"] * s_activity
            + scoring_weights["time"] * s_time
            + scoring_weights["amount"] * s_amt
        )
        overall = round(float(np.clip(overall, 0.0, 100.0)), 2)

        # Evidence / Contextual flags
        flags: List[str] = []
        if bank_label == "EXACT_MATCH":
            flags.append("EXACT_BANK_MATCH")
        elif bank_label == "PARTIAL_MATCH":
            flags.append("PARTIAL_BANK_MATCH")
        elif bank_label == "UNKNOWN":
            flags.append("UNKNOWN_BANK_DATA")
        else:
            flags.append("BANK_MISMATCH")

        if time_label == "OPEN_COMPATIBLE" and is_24x7 is True:
            flags.append("24X7_OPEN")
        elif time_label == "POSSIBLY_INCOMPATIBLE":
            flags.append("RESTRICTED_NIGHT_HOURS")
        elif time_label == "UNKNOWN":
            flags.append("UNKNOWN_HOURS")

        if s_activity >= 40.0:
            flags.append("HIGH_SIMULATED_ACTIVITY")
        elif s_activity >= 20.0:
            flags.append("MODERATE_SIMULATED_ACTIVITY")

        if assigned_zone == pred_zone:
            flags.append("PRIMARY_ZONE_CANDIDATE")
        else:
            flags.append("CROSS_ZONE_CANDIDATE")

        if amt_label in ["HIGH_VOLUME_COMPATIBLE", "STANDARD_COMPATIBLE"]:
            flags.append("AMOUNT_COMPATIBLE")

        # Plain-language explanation for this candidate
        exp_parts = []
        exp_parts.append(
            f"Assigned to {assigned_zone} ({dist_km:.1f} km from reference sector center; zone likelihood score {s_zone:.1f}/100)."
        )
        if bank_label == "EXACT_MATCH":
            exp_parts.append(f"Strong bank compatibility with victim bank '{complaint_bank}'.")
        elif bank_label == "UNKNOWN":
            exp_parts.append("Bank compatibility unknown due to missing operator metadata in OpenStreetMap.")
        elif bank_label == "MISMATCH":
            exp_parts.append(f"Bank mismatch: ATM operated by '{atm_bank}' vs complaint bank '{complaint_bank}'.")

        if is_24x7 is True:
            exp_parts.append("24x7 operational availability confirmed.")
        elif is_24x7 is False and (complaint_hour is not None and (complaint_hour >= 22 or complaint_hour <= 5)):
            exp_parts.append("Potential night-time operational restriction.")

        explanation_text = " ".join(exp_parts)

        ranked_records.append({
            "atm_id": atm_id,
            "bank": atm_bank,
            "operator": operator,
            "latitude": lat,
            "longitude": lon,
            "city": str(row["city"]),
            "district": str(row["district"]),
            "state": str(row["state"]),
            "zone": assigned_zone,
            "secondary_zone": sec_zone,
            "distance_km": round(dist_km, 2),
            "overall_score": overall,
            "zone_probability_score": s_zone,
            "spatial_score": s_spatial,
            "bank_score": s_bank,
            "time_score": s_time,
            "activity_score": s_activity,
            "amount_compatibility_score": s_amt,
            "evidence_flags": flags,
            "explanation": explanation_text,
            "source": str(row.get("source", "OpenStreetMap")),
        })

    # Sort descending by overall_score, breaking ties by distance_km ascending
    ranked_records.sort(key=lambda x: (-x["overall_score"], x["distance_km"]))

    # Assign rank index (1-indexed) and truncate to top_n
    top_candidates = []
    for idx, rec in enumerate(ranked_records[:top_n], start=1):
        rec["rank"] = idx
        rec["designation"] = "Highest-ranked candidate" if idx == 1 else f"Candidate #{idx}"
        top_candidates.append(rec)

    return {
        "case_id": str(case_data.get("complaint_id", "UNKNOWN_CASE")),
        "model_id": str(prediction_result.get("model_id", "random_forest_full_none")),
        "predicted_zone": pred_zone,
        "confidence_tier": str(prediction_result.get("confidence_tier", "HIGH")),
        "prediction_confidence": float(prediction_result.get("prediction_confidence", 0.0)),
        "candidate_zones": candidate_zones,
        "cross_zone_search": cross_zone_active,
        "total_atms_evaluated": len(ranked_records),
        "scoring_weights": scoring_weights,
        "ranked_atms": top_candidates,
    }


# ==============================================================================
# Backward Compatibility Shims for Phase 1 / Phase 2 Tests
# ==============================================================================

def calculate_candidate_score(
    distance_km: float,
    atm_bank: str,
    complaint_bank: Optional[str] = None,
    is_24x7: bool = True,
    historical_fraud_count: int = 0,
    is_night: bool = False,
) -> float:
    """Backward-compatible candidate score returning float in [0.0, 1.0]."""
    proximity_score = math.exp(-0.15 * max(0.0, distance_km))

    bank_score = 0.5
    if complaint_bank and atm_bank:
        if complaint_bank.lower() in atm_bank.lower() or atm_bank.lower() in complaint_bank.lower():
            bank_score = 1.0
        else:
            bank_score = 0.35

    access_score = 1.0
    if is_night and not is_24x7:
        access_score = 0.25

    activity_score = min(1.0, historical_fraud_count / 10.0)

    composite = (
        0.40 * proximity_score
        + 0.30 * bank_score
        + 0.15 * access_score
        + 0.15 * activity_score
    )
    return round(float(np.clip(composite, 0.0, 1.0)), 3)


def rank_atm_candidates(
    candidates: List[Dict[str, Any]],
    complaint_bank: Optional[str] = None,
    complaint_hour: Optional[int] = None,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Backward-compatible ATM ranking function for existing tests and legacy interfaces."""
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
        hist_fraud = int(cand.get("historical_fraud_withdrawals", cand.get("historical_fraud_count", 0)))

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
            "overall_score": round(score * 100.0, 1),
            "address": cand.get("address", "N/A"),
            "latitude": cand.get("latitude"),
            "longitude": cand.get("longitude"),
            "city": cand.get("city", "Unknown"),
            "district": cand.get("district", "Unknown"),
            "state": cand.get("state", "Unknown"),
            "source": cand.get("source", "OpenStreetMap"),
            "evidence_flags": ["LEGACY_COMPATIBILITY"],
        }
        ranked_list.append(record)

    ranked_list.sort(key=lambda x: x["candidate_score"], reverse=True)
    top_candidates = ranked_list[:top_n]
    for idx, c in enumerate(top_candidates, 1):
        c["rank"] = idx
        c["designation"] = "Highest-ranked candidate" if idx == 1 else f"Candidate #{idx}"
    return top_candidates
