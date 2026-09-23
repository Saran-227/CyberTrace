"""Data loading module for CyberTrace datasets."""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from src.config import (
    COMPLAINTS_RAW_PATH,
    COMPLAINTS_PROCESSED_PATH,
    ATM_LOCATIONS_RAW_PATH,
    ATM_LOCATIONS_PROCESSED_PATH,
    ATM_ACTIVITY_RAW_PATH,
    ATM_ACTIVITY_PROCESSED_PATH,
)
from src.utils.logging import get_logger

logger = get_logger("DataLoader")

def load_raw_complaints(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load raw synthetic cybercrime complaints dataset."""
    target_path = path or COMPLAINTS_RAW_PATH
    if not target_path.exists():
        logger.warning(f"Raw complaints file not found at: {target_path}")
        return None
    logger.info(f"Loading raw complaints from {target_path}")
    return pd.read_csv(target_path)

def load_processed_complaints(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load processed synthetic cybercrime complaints dataset."""
    target_path = path or COMPLAINTS_PROCESSED_PATH
    if not target_path.exists():
        logger.warning(f"Processed complaints file not found at: {target_path}")
        return None
    logger.info(f"Loading processed complaints from {target_path}")
    return pd.read_csv(target_path)

def load_atm_locations(use_raw: bool = False) -> Optional[pd.DataFrame]:
    """Load ATM locations dataset (raw or processed)."""
    target_path = ATM_LOCATIONS_RAW_PATH if use_raw else ATM_LOCATIONS_PROCESSED_PATH
    if not target_path.exists():
        logger.warning(f"ATM locations file not found at: {target_path}")
        return None
    return pd.read_csv(target_path)

def load_atm_activity(use_raw: bool = False) -> Optional[pd.DataFrame]:
    """Load ATM historical activity dataset (synthetic)."""
    target_path = ATM_ACTIVITY_RAW_PATH if use_raw else ATM_ACTIVITY_PROCESSED_PATH
    if not target_path.exists():
        logger.warning(f"ATM activity file not found at: {target_path}")
        return None
    return pd.read_csv(target_path)

def get_dataset_status() -> Dict[str, Any]:
    """Return presence and record count summary for all primary datasets."""
    status = {}
    files = {
        "raw_complaints": COMPLAINTS_RAW_PATH,
        "processed_complaints": COMPLAINTS_PROCESSED_PATH,
        "raw_atm_locations": ATM_LOCATIONS_RAW_PATH,
        "processed_atm_locations": ATM_LOCATIONS_PROCESSED_PATH,
        "raw_atm_activity": ATM_ACTIVITY_RAW_PATH,
        "processed_atm_activity": ATM_ACTIVITY_PROCESSED_PATH,
    }
    for key, path in files.items():
        exists = path.exists()
        rows = 0
        if exists:
            try:
                # Fast count
                rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
            except Exception:
                rows = -1
        status[key] = {
            "exists": exists,
            "path": str(path),
            "records": max(0, rows),
        }
    return status
