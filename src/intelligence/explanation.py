"""Non-technical investigative explanation generator for CyberTrace intelligence.

Translates machine learning zone predictions, uncertainty diagnostics, and multi-criteria
ATM candidate rankings into clear, plain-language operational summaries for investigators.
"""

from typing import Dict, Any, List, Optional, Union


def generate_investigative_explanation(
    case_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    ranking_result: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Generate clear, non-technical explanations for law enforcement / intelligence analysts.

    Parameters:
        case_data: Normalized complaint metadata dictionary
        prediction_result: Phase 6 location classifier output contract
        ranking_result: Phase 7 ATM candidate ranking output contract or list of ranked ATMs

    Returns:
        Structured dictionary containing plain-language intelligence briefs and recommended actions.
    """
    pred_zone = prediction_result.get("predicted_zone", "Unknown Zone")
    confidence_pct = round(float(prediction_result.get("prediction_confidence", prediction_result.get("confidence", 0.0))) * 100, 1)
    tier = str(prediction_result.get("confidence_tier", "HIGH")).upper()
    margin = float(prediction_result.get("probability_margin", 0.0))
    second_zone = prediction_result.get("second_best_zone", "None")

    if isinstance(ranking_result, list):
        ranked_atms = ranking_result
        cross_zone = False
    else:
        ranked_atms = ranking_result.get("ranked_atms", [])
        cross_zone = bool(ranking_result.get("cross_zone_search", False))

    # 1. Zone Summary
    if cross_zone:
        zone_summary = (
            f"The intelligence platform predicts {pred_zone} as the highest-probability cash-out area with "
            f"{confidence_pct}% model confidence (Tier: {tier}, margin {margin:.2f}). Because of high boundary "
            f"proximity and contested probability with {second_zone}, a cross-boundary candidate search was "
            f"automatically activated across both sectors."
        )
    else:
        zone_summary = (
            f"The intelligence platform predicts {pred_zone} as the highest-probability cash-out area with "
            f"{confidence_pct}% model confidence (Tier: {tier}, margin {margin:.2f})."
        )

    # 2. ATM Summary
    if ranked_atms:
        top_atm = ranked_atms[0]
        atm_id = top_atm.get("atm_id", "Unknown")
        bank = top_atm.get("bank", "Unknown")
        city = top_atm.get("city", "Unknown")
        dist = top_atm.get("distance_km", 0.0)
        score = top_atm.get("overall_score", 0.0)
        bank_score = top_atm.get("bank_score", 50.0)

        bank_note = (
            f"operated by {bank} (compatible with victim bank '{case_data.get('bank')}')"
            if bank_score >= 70.0
            else f"operated by {bank}"
        )
        if bank.lower() == "unknown":
            bank_note = "with unspecified bank/operator data in OpenStreetMap"

        atm_summary = (
            f"Candidate ATM '{atm_id}' in {city} {bank_note} was identified as the highest-ranked candidate "
            f"(composite score: {score:.1f}/100, {dist:.1f} km from sector reference center)."
        )
    else:
        atm_summary = (
            "No verified OpenStreetMap ATM points of interest were located within the immediate perimeter of "
            f"the candidate sectors (e.g. Panipat / {pred_zone})."
        )

    # 3. Key Operational Signals
    signals = [
        f"Geographic complaint origin in {case_data.get('city', 'reported city')}, {case_data.get('state', '')}.",
        f"Victim financial institution recorded as {case_data.get('bank', 'reporting bank')}.",
        f"Disputed amount ₹{case_data.get('amount', 0):,} ({case_data.get('amount_category', 'Standard')}).",
        f"Incident timing recorded at {case_data.get('hour', 0):02d}:00 hours ({'night-time' if case_data.get('is_night') else 'daytime'}).",
    ]
    if cross_zone:
        signals.append(f"Dual-sector evaluation active: {pred_zone} and {second_zone} evaluated concurrently.")

    # 4. Actionable Next Steps
    actions = []
    if ranked_atms:
        top_ids = ", ".join([a.get("atm_id", "") for a in ranked_atms[:3]])
        actions.append(f"Prioritize CCTV footage and physical audit requests for top candidate ATMs ({top_ids}).")
    actions.extend([
        f"Request formal transaction interchange reconciliation from {case_data.get('bank', 'the issuing bank')}.",
        "Cross-reference suspect device telecommunications tower pings against candidate ATM coordinates.",
        "File intelligence brief under National Cyber Crime Reporting Portal reference.",
    ])

    return {
        "zone_summary": zone_summary,
        "atm_summary": atm_summary,
        "key_signals": signals,
        "recommended_focus": actions,
        "evidence_disclaimer": (
            "All candidate rankings represent mathematical and geographic likelihoods. "
            "Verification using official banking and surveillance evidence is strictly mandatory prior to legal enforcement."
        ),
    }
