"""Validation utility for CyberTrace data schemas and inputs."""

from typing import List, Dict, Any, Tuple
import pandas as pd
from src.config import COMPLAINT_SCHEMA, CANDIDATE_FEATURES, TARGET_COLUMN

def validate_complaint_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single complaint dictionary against expected minimal candidate keys."""
    missing = [f for f in CANDIDATE_FEATURES if f not in record]
    if missing:
        return False, [f"Missing required candidate feature: {col}" for col in missing]
    return True, []

def validate_dataframe_schema(df: pd.DataFrame, expected_columns: List[str]) -> Tuple[bool, List[str]]:
    """Check if all expected columns are present in DataFrame."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return False, [f"DataFrame missing column: {col}" for col in missing]
    return True, []

def check_target_leakage(features: List[str]) -> bool:
    """Ensure target variable withdrawal_zone is never included in training features."""
    return TARGET_COLUMN in features
