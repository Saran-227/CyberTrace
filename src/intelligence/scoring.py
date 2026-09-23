"""Composite scoring synthesis for intelligence prioritization."""

from typing import Dict, Any

def compute_composite_intelligence_score(
    prediction_confidence: float,
    top_atm_score: float,
    amount_severity: str = "Medium",
) -> float:
    """Combine ML zone confidence, candidate ATM ranking score, and fraud severity.

    Used by investigators to prioritize actionable leads.
    """
    severity_weights = {
        "Low (<5k)": 0.6,
        "Medium (5k-25k)": 0.8,
        "High (25k-50k)": 0.95,
        "Critical (>50k)": 1.0,
    }
    sev_w = severity_weights.get(amount_severity, 0.8)

    composite = (0.50 * prediction_confidence + 0.35 * top_atm_score + 0.15 * sev_w)
    return round(float(composite), 3)
