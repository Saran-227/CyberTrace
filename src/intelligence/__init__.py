"""Intelligence reporting, explanation, and synthesis module for CyberTrace."""

from .scoring import compute_composite_intelligence_score
from .explanation import generate_investigative_explanation
from .report import generate_executive_report

__all__ = [
    "compute_composite_intelligence_score",
    "generate_investigative_explanation",
    "generate_executive_report",
]
