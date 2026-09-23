"""Non-technical investigative explanation generator for CyberTrace intelligence."""

from typing import Dict, Any, List

def generate_investigative_explanation(
    case_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    ranked_atms: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate clear, non-technical explanations for law enforcement / intelligence analysts."""
    predicted_zone = prediction_result.get("predicted_zone", "Unknown Zone")
    confidence_pct = round(prediction_result.get("confidence", 0.0) * 100, 1)

    top_atm = ranked_atms[0] if ranked_atms else None
    top_atm_text = (
        f"Candidate ATM '{top_atm.get('atm_id')}' operated by {top_atm.get('bank')} was identified as the highest-ranked candidate "
        f"located {top_atm.get('distance_km')} km from the reference zone center."
        if top_atm
        else "No candidate ATMs were found in the immediate perimeter."
    )

    signals = [
        f"Geographic proximity of complaint incident in {case_data.get('city', 'the reported city')}.",
        f"Target bank compatibility ({case_data.get('bank', 'reporting bank')}) matching candidate cash points.",
        f"Transaction type '{case_data.get('transaction_type', 'digital transfer')}' with amount ₹{case_data.get('amount', 0):,}.",
        f"Temporal window recorded at {case_data.get('hour', 0):02d}:00 hours ({'night-time' if case_data.get('is_night') else 'daytime'}).",
    ]

    focus_actions = [
        f"Review CCTV footage and physical surveillance records for {top_atm.get('atm_id') if top_atm else 'top candidate ATMs'} during the incident time window.",
        f"Request formal transaction logs and interchange reconciliation from {case_data.get('bank', 'the issuing bank')}.",
        "Cross-reference suspect device location with candidate ATM cell tower records.",
        "File supplementary investigative notes under National Cyber Crime Reporting Portal reference.",
    ]

    return {
        "zone_summary": f"The intelligence platform predicts {predicted_zone} as the highest-probability cash-out area with {confidence_pct}% model confidence.",
        "atm_summary": top_atm_text,
        "key_signals": signals,
        "recommended_focus": focus_actions,
        "evidence_disclaimer": "All candidate rankings represent mathematical likelihoods. Verification using official banking and surveillance evidence is mandatory prior to legal enforcement.",
    }
