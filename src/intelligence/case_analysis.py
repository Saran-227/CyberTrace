"""Dynamic Live Case Analysis Orchestration Service for CyberTrace Phase 7.

Orchestrates the entire live end-to-end pipeline:
User Input
    ↓
Validation & Normalization
    ↓
Feature Preprocessing
    ↓
Primary Location Model (with graceful Fallback)
    ↓
Predicted Withdrawal Zone & 10-Zone Probabilities
    ↓
Candidate Zone Selection (with Cross-Zone NCR Search)
    ↓
ATM Discovery & Spatial Filtering
    ↓
Multi-Criteria Candidate Ranking
    ↓
Non-Technical Plain-Language Explanations
    ↓
Frontend Result Object
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from src.config import (
    PRIMARY_MODEL_ID,
    FALLBACK_MODEL_ID,
    PRIMARY_MODEL_PATH,
    FALLBACK_MODEL_PATH,
    LOCATION_CLASSIFIER_DIR,
    ATM_RANKING_WEIGHTS,
    CANDIDATE_SEARCH_CONFIG,
)
from src.preprocessing.features import engineer_complaint_features
from src.atm.loader import load_atm_locations, load_zone_centroids
from src.atm.ranking import rank_atms_for_case
from src.intelligence.explanation import generate_investigative_explanation
from src.utils.logging import get_logger

logger = get_logger("CaseAnalysisService")

# Cached model instances in memory
_CACHED_MODELS: Dict[str, Any] = {}


def get_location_model(model_id: Optional[str] = None) -> Tuple[Any, str]:
    """Retrieve or load trained location classification model pipeline.

    Tries primary model first (random_forest_full_none), gracefully falling back
    to fallback model (logistic_full_none) if primary is unavailable or fails.
    """
    global _CACHED_MODELS

    target_id = model_id or PRIMARY_MODEL_ID
    if target_id in _CACHED_MODELS:
        return _CACHED_MODELS[target_id], target_id

    # 1. Attempt loading requested model
    model_path = LOCATION_CLASSIFIER_DIR / f"{target_id}.joblib"
    if model_path.exists():
        try:
            pipeline = joblib.load(model_path)
            _CACHED_MODELS[target_id] = pipeline
            logger.info(f"Loaded requested model pipeline: {target_id}")
            return pipeline, target_id
        except Exception as exc:
            logger.warning(f"Failed to load requested model '{target_id}': {exc}. Attempting fallback...")

    # 2. Fallback attempt
    fallback_id = FALLBACK_MODEL_ID
    if fallback_id in _CACHED_MODELS:
        return _CACHED_MODELS[fallback_id], fallback_id

    fallback_path = FALLBACK_MODEL_PATH
    if fallback_path.exists():
        try:
            pipeline = joblib.load(fallback_path)
            _CACHED_MODELS[fallback_id] = pipeline
            logger.info(f"Loaded fallback model pipeline: {fallback_id}")
            return pipeline, fallback_id
        except Exception as exc:
            logger.error(f"Failed to load fallback model '{fallback_id}': {exc}")

    # 3. Final resort: search any joblib in directory
    any_models = list(LOCATION_CLASSIFIER_DIR.glob("*.joblib"))
    if any_models:
        pipeline = joblib.load(any_models[0])
        loaded_id = any_models[0].stem
        _CACHED_MODELS[loaded_id] = pipeline
        return pipeline, loaded_id

    raise RuntimeError(f"No trained location model artifacts found in {LOCATION_CLASSIFIER_DIR}.")


def validate_and_normalize_case_input(raw_input: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Validate and normalize raw user case input, guaranteeing clean values and preventing crashes."""
    if isinstance(raw_input, pd.DataFrame):
        data = raw_input.iloc[0].to_dict()
    else:
        data = dict(raw_input)

    # 1. Reference Complaint ID
    complaint_id = str(data.get("complaint_id", f"CT-CASE-{int(time.time())}"))

    # 2. Date and Time parsing
    date_str = str(data.get("complaint_date", datetime.today().strftime("%Y-%m-%d"))).strip()
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        parsed_date = datetime.today()
        date_str = parsed_date.strftime("%Y-%m-%d")

    time_str = str(data.get("complaint_time", "12:00")).strip()
    try:
        hour = int(time_str.split(":")[0])
        hour = max(0, min(23, hour))
    except Exception:
        hour = int(data.get("hour", 12))
        hour = max(0, min(23, hour))

    dow = int(data.get("day_of_week", parsed_date.weekday()))
    dow = max(0, min(6, dow))
    day_name = parsed_date.strftime("%A")
    is_weekend = int(dow >= 5)
    is_night = int(hour >= 22 or hour <= 5)

    # 3. Monetary amount
    try:
        raw_amt = float(data.get("amount", 25000.0))
        amount = max(100.0, raw_amt)  # enforce positive non-zero amount
    except Exception:
        amount = 25000.0

    # 4. Bank
    bank = str(data.get("bank", "unknown")).strip()
    if bank.lower() in ["", "none", "nan", "null"]:
        bank = "unknown"

    # 5. Rails and Fraud Modus Operandi
    tx_type = str(data.get("transaction_type", "UPI")).strip() or "UPI"
    fraud_type = str(data.get("fraud_type", "Phishing/Smishing")).strip() or "Phishing/Smishing"

    # 6. City, State, District, and Geographic Coordinates
    city_defaults = {
        "Amritsar": (31.6340, 74.8723, "Amritsar", "Punjab"),
        "Jalandhar": (31.3260, 75.5762, "Jalandhar", "Punjab"),
        "Ludhiana": (30.9010, 75.8573, "Ludhiana", "Punjab"),
        "Patiala": (30.3398, 76.3869, "Patiala", "Punjab"),
        "Chandigarh": (30.7333, 76.7794, "Chandigarh", "Chandigarh"),
        "Ambala": (30.3782, 76.7767, "Ambala", "Haryana"),
        "Panipat": (29.3909, 76.9635, "Panipat", "Haryana"),
        "Ghaziabad": (28.6692, 77.4538, "Ghaziabad", "Uttar Pradesh"),
        "Meerut": (28.9845, 77.7064, "Meerut", "Uttar Pradesh"),
        "New Delhi": (28.6139, 77.2090, "New Delhi", "Delhi"),
        "Noida": (28.5355, 77.3910, "Gautam Buddha Nagar", "Uttar Pradesh"),
        "Gurugram": (28.4595, 77.0266, "Gurugram", "Haryana"),
        "Faridabad": (28.4089, 77.3178, "Faridabad", "Haryana"),
        "Alwar": (27.5530, 76.6346, "Alwar", "Rajasthan"),
        "Jaipur": (26.9124, 75.7873, "Jaipur", "Rajasthan"),
    }

    city = str(data.get("city", "New Delhi")).strip()
    if city not in city_defaults:
        # Match case-insensitively or default to New Delhi
        matched_city = None
        for k in city_defaults:
            if k.lower() == city.lower():
                matched_city = k
                break
        city = matched_city or "New Delhi"

    def_lat, def_lon, def_dist, def_state = city_defaults[city]

    try:
        lat = float(data.get("complaint_latitude", def_lat))
        if lat < -90.0 or lat > 90.0 or np.isnan(lat):
            lat = def_lat
    except Exception:
        lat = def_lat

    try:
        lon = float(data.get("complaint_longitude", def_lon))
        if lon < -180.0 or lon > 180.0 or np.isnan(lon):
            lon = def_lon
    except Exception:
        lon = def_lon

    district = str(data.get("district", def_dist)).strip() or def_dist
    state = str(data.get("state", def_state)).strip() or def_state

    # 7. Amount category risk tier
    if amount < 5000:
        amount_cat = "Low (<5k)"
    elif amount < 25000:
        amount_cat = "Medium (5k-25k)"
    elif amount < 50000:
        amount_cat = "High (25k-50k)"
    else:
        amount_cat = "Critical (>50k)"

    return {
        "complaint_id": complaint_id,
        "complaint_date": date_str,
        "complaint_time": time_str,
        "amount": round(amount, 2),
        "bank": bank,
        "transaction_type": tx_type,
        "fraud_type": fraud_type,
        "city": city,
        "district": district,
        "state": state,
        "complaint_latitude": round(lat, 5),
        "complaint_longitude": round(lon, 5),
        "hour": hour,
        "day_of_week": dow,
        "day_name": day_name,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "amount_category": amount_cat,
    }


def analyze_case(
    case_input: Union[Dict[str, Any], pd.DataFrame],
    model_id: Optional[str] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Execute live end-to-end case analysis uniting ML location prediction and ATM candidate ranking.

    Parameters:
        case_input: Case dictionary or DataFrame containing complaint attributes
        model_id: Optional model experiment identifier (defaults to PRIMARY_MODEL_ID)
        top_n: Number of candidate ATMs to return (default 10)

    Returns:
        Structured result dictionary conforming to CyberTrace prediction & ranking contracts.
    """
    t_start = time.time()

    # 1. Validate and normalize input
    case_data = validate_and_normalize_case_input(case_input)

    # 2. Preprocess and derive features
    df_raw = pd.DataFrame([case_data])
    df_engineered = engineer_complaint_features(df_raw)

    # 3. Load location classifier (primary with graceful fallback)
    pipeline, active_model_id = get_location_model(model_id)

    # 4. Predict zone probabilities
    raw_probs = pipeline.predict_proba(df_engineered)[0]
    classes = pipeline.named_steps["classifier"].classes_

    # Map class probabilities across all 10 canonical zones
    canonical_zones = [f"Zone_{i:02d}" for i in range(1, 11)]
    zone_probs: Dict[str, float] = {}
    for z in canonical_zones:
        if z in classes:
            idx = int(np.where(classes == z)[0][0])
            zone_probs[z] = round(float(raw_probs[idx]), 4)
        else:
            zone_probs[z] = 0.0

    # Sort descending to find top and runner-up candidate zones
    sorted_zones = sorted(zone_probs.items(), key=lambda x: x[1], reverse=True)
    predicted_zone = sorted_zones[0][0]
    prediction_confidence = float(sorted_zones[0][1])
    second_best_zone = sorted_zones[1][0]
    second_confidence = float(sorted_zones[1][1])
    probability_margin = round(prediction_confidence - second_confidence, 4)

    # Diagnostic confidence tiers established in Phase 6
    if probability_margin >= CANDIDATE_SEARCH_CONFIG["high_confidence_margin"]:
        confidence_tier = "HIGH"
    elif probability_margin >= CANDIDATE_SEARCH_CONFIG["medium_confidence_margin"]:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"

    prediction_contract = {
        "complaint_id": case_data["complaint_id"],
        "predicted_zone": predicted_zone,
        "prediction_confidence": prediction_confidence,
        "second_best_zone": second_best_zone,
        "probability_margin": probability_margin,
        "confidence_tier": confidence_tier,
        "model_id": active_model_id,
        "zone_probabilities": zone_probs,
    }

    # 5. Load verified ATM locations
    atms_df = load_atm_locations()

    # 6. Execute transparent multi-criteria ATM candidate ranking
    ranking_result = rank_atms_for_case(
        case_data=case_data,
        prediction_result=prediction_contract,
        atms_df=atms_df,
        top_n=top_n,
    )

    # 7. Generate non-technical plain-language intelligence brief
    explanation_result = generate_investigative_explanation(
        case_data=case_data,
        prediction_result=prediction_contract,
        ranking_result=ranking_result,
    )

    execution_ms = round((time.time() - t_start) * 1000, 2)

    # Assemble comprehensive result object
    return {
        "status": "SUCCESS",
        "case_id": case_data["complaint_id"],
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": execution_ms,
        "model_id": active_model_id,
        "case_data": case_data,
        "prediction": prediction_contract,
        "ranking": ranking_result,
        "explanation": explanation_result,
        "provenance": {
            "model_family": "RandomForestClassifier (Ensemble 100 trees)" if "random_forest" in active_model_id else "LogisticRegression",
            "model_artifact": str(LOCATION_CLASSIFIER_DIR / f"{active_model_id}.joblib"),
            "atm_source": "OpenStreetMap Verified Geographic Infrastructure (ODbL)",
            "activity_source": "Synthetic ATM Operational Activity (90-day baseline, zero target leakage)",
            "scoring_weights": ranking_result.get("scoring_weights", ATM_RANKING_WEIGHTS),
            "disclaimer": (
                "Academic Intelligence System: Cash-out withdrawal zones and ATM candidate rankings "
                "represent probabilistic mathematical prioritization. CyberTrace does not claim confirmed "
                "forensic evidence without authoritative bank logs and physical CCTV records."
            ),
        },
    }
