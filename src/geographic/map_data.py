"""Map Data Preparation Service for CyberTrace Phase 8.

Converts Phase 7 case analysis results into a structured, validated, and normalized
geospatial contract consumed by the interactive Leaflet + OpenStreetMap component.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.geographic.zones import ZONE_DEFINITIONS
from src.utils.logging import get_logger

logger = get_logger("MapData")

_CACHED_ZONE_BBOXES: Optional[Dict[str, Tuple[float, float, float, float]]] = None


def load_zone_bounding_boxes(filepath: Optional[Path] = None) -> Dict[str, Tuple[float, float, float, float]]:
    """Load verified geographic bounding boxes for all 10 withdrawal zones."""
    global _CACHED_ZONE_BBOXES
    if _CACHED_ZONE_BBOXES is not None:
        return dict(_CACHED_ZONE_BBOXES)

    bbox_path = filepath or (REPORTS_DIR / "zone_bounding_boxes.csv")
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    if bbox_path.exists():
        df = pd.read_csv(bbox_path)
        for _, row in df.iterrows():
            zone = str(row["withdrawal_zone"])
            bboxes[zone] = (
                float(row["min_latitude"]),
                float(row["min_longitude"]),
                float(row["max_latitude"]),
                float(row["max_longitude"]),
            )
    else:
        # Fallback to predefined bounding boxes in zones.py
        for zone_id, info in ZONE_DEFINITIONS.items():
            bboxes[zone_id] = info.get("bbox", (28.0, 76.0, 29.0, 77.0))

    _CACHED_ZONE_BBOXES = bboxes
    return dict(bboxes)


def calculate_geographic_bounds(
    points: List[Tuple[float, float]],
    padding_degrees: float = 0.04,
) -> Tuple[List[List[float]], List[float], int]:
    """Calculate encompassing bounds [[min_lat, min_lon], [max_lat, max_lon]], center, and zoom."""
    if not points:
        # Default North India study region envelope
        return [[27.0, 74.5], [32.0, 78.0]], [29.5, 76.25], 8

    lats = [p[0] for p in points if not np.isnan(p[0])]
    lons = [p[1] for p in points if not np.isnan(p[1])]

    if not lats or not lons:
        return [[27.0, 74.5], [32.0, 78.0]], [29.5, 76.25], 8

    min_lat = max(-90.0, min(lats) - padding_degrees)
    max_lat = min(90.0, max(lats) + padding_degrees)
    min_lon = max(-180.0, min(lons) - padding_degrees)
    max_lon = min(180.0, max(lons) + padding_degrees)

    bounds = [
        [round(min_lat, 5), round(min_lon, 5)],
        [round(max_lat, 5), round(max_lon, 5)],
    ]
    center = [round((min_lat + max_lat) / 2.0, 5), round((min_lon + max_lon) / 2.0, 5)]

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    max_span = max(lat_span, lon_span)

    if max_span > 2.5:
        zoom = 8
    elif max_span > 1.2:
        zoom = 9
    elif max_span > 0.6:
        zoom = 10
    elif max_span > 0.3:
        zoom = 11
    elif max_span > 0.15:
        zoom = 12
    else:
        zoom = 13

    return bounds, center, zoom


def prepare_map_data(
    case_analysis: Dict[str, Any],
    max_candidates: Optional[int] = 10,
    bank_filter: Optional[str] = None,
    zone_filter: Optional[str] = None,
    selected_atm_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert Phase 7 case analysis result into a validated, map-ready data contract.

    Parameters:
        case_analysis: Standardized result from src.intelligence.case_analysis:analyze_case()
        max_candidates: Maximum number of candidate ATMs to include (default 10)
        bank_filter: Optional bank name to filter visible candidates
        zone_filter: Optional zone ID to filter visible candidates
        selected_atm_id: Optional ATM ID to mark as actively selected/focused

    Returns:
        Structured dictionary conforming to docs/map_contract.md
    """
    if not isinstance(case_analysis, dict):
        raise TypeError(f"Expected case_analysis to be a dict, got {type(case_analysis).__name__}")

    case_data = case_analysis.get("case_data", {})
    prediction = case_analysis.get("prediction", {})
    ranking = case_analysis.get("ranking", {})

    all_bboxes = load_zone_bounding_boxes()
    bounding_points: List[Tuple[float, float]] = []

    # 1. Complaint Origin Point
    complaint_point = None
    c_lat = case_data.get("complaint_latitude")
    c_lon = case_data.get("complaint_longitude")
    if c_lat is not None and c_lon is not None and not np.isnan(float(c_lat)) and not np.isnan(float(c_lon)):
        lat_f = float(c_lat)
        lon_f = float(c_lon)
        complaint_point = {
            "latitude": round(lat_f, 5),
            "longitude": round(lon_f, 5),
            "case_id": str(case_analysis.get("case_id", "UNKNOWN")),
            "city": str(case_data.get("city", "Reported City")),
            "district": str(case_data.get("district", "")),
            "state": str(case_data.get("state", "")),
            "amount": float(case_data.get("amount", 0.0)),
            "bank": str(case_data.get("bank", "Unknown Bank")),
            "date": str(case_data.get("complaint_date", "")),
            "time": str(case_data.get("complaint_time", "")),
            "label": (
                f"Complaint Location: {case_data.get('city', '')} | "
                f"Amount: INR {float(case_data.get('amount', 0.0)):,} | "
                f"Case: {case_analysis.get('case_id', '')}"
            ),
        }
        bounding_points.append((lat_f, lon_f))

    # 2. Predicted Primary Zone
    pred_zone_id = str(prediction.get("predicted_zone", "Zone_01"))
    pred_conf = float(prediction.get("prediction_confidence", 0.0))
    conf_tier = str(prediction.get("confidence_tier", "HIGH")).upper()
    margin = float(prediction.get("probability_margin", 0.0))
    second_zone_id = str(prediction.get("second_best_zone", "None"))
    zone_probs = prediction.get("zone_probabilities", {})

    pred_bbox = all_bboxes.get(pred_zone_id, (28.0, 76.0, 29.0, 77.0))
    bounding_points.extend([
        (pred_bbox[0], pred_bbox[1]),
        (pred_bbox[2], pred_bbox[3]),
    ])

    predicted_zone_info = {
        "zone_id": pred_zone_id,
        "probability": round(pred_conf * 100.0, 1),
        "confidence_tier": conf_tier,
        "probability_margin": margin,
        "second_best_zone": second_zone_id,
        "bbox": {
            "min_lat": round(pred_bbox[0], 5),
            "min_lon": round(pred_bbox[1], 5),
            "max_lat": round(pred_bbox[2], 5),
            "max_lon": round(pred_bbox[3], 5),
        },
        "color": "#00e5ff",  # Cyan primary zone
        "fill_opacity": round(float(np.clip(0.12 + (pred_conf * 0.25), 0.12, 0.40)), 2),
    }

    # 3. Candidate Zones (including Cross-Zone sectors)
    candidate_zone_ids = ranking.get("candidate_zones", [pred_zone_id])
    candidate_zones_list: List[Dict[str, Any]] = []

    for z_id in candidate_zone_ids:
        z_bbox = all_bboxes.get(z_id)
        if not z_bbox:
            continue
        z_p = float(zone_probs.get(z_id, 0.0))
        is_primary = (z_id == pred_zone_id)
        color = "#00e5ff" if is_primary else "#f59e0b"  # Cyan primary, Amber secondary
        fill_opacity = round(float(np.clip(0.10 + (z_p * 0.25), 0.08, 0.35)), 2)

        candidate_zones_list.append({
            "zone_id": z_id,
            "probability": round(z_p * 100.0, 1),
            "is_primary": is_primary,
            "bbox": {
                "min_lat": round(z_bbox[0], 5),
                "min_lon": round(z_bbox[1], 5),
                "max_lat": round(z_bbox[2], 5),
                "max_lon": round(z_bbox[3], 5),
            },
            "color": color,
            "fill_opacity": fill_opacity,
        })
        bounding_points.extend([
            (z_bbox[0], z_bbox[1]),
            (z_bbox[2], z_bbox[3]),
        ])

    # 4. Filter and Process Ranked ATM Candidates
    raw_ranked_atms = ranking.get("ranked_atms", [])
    filtered_atms: List[Dict[str, Any]] = []

    for atm in raw_ranked_atms:
        # Bank filter
        if bank_filter and bank_filter.strip().lower() not in ["all", ""]:
            atm_b = str(atm.get("bank", "")).lower()
            atm_op = str(atm.get("operator", "")).lower()
            if bank_filter.lower() not in atm_b and bank_filter.lower() not in atm_op:
                continue

        # Zone filter
        if zone_filter and zone_filter.strip().lower() not in ["all", ""]:
            if str(atm.get("zone", "")).lower() != zone_filter.lower():
                continue

        filtered_atms.append(atm)

    if max_candidates is not None and max_candidates > 0:
        filtered_atms = filtered_atms[:max_candidates]

    map_atms: List[Dict[str, Any]] = []
    for idx, atm in enumerate(filtered_atms, start=1):
        lat = float(atm.get("latitude", 0.0))
        lon = float(atm.get("longitude", 0.0))
        atm_id = str(atm.get("atm_id", "Unknown"))
        is_top = (idx == 1 and not bank_filter and not zone_filter)
        is_selected = (selected_atm_id is not None and atm_id == selected_atm_id)

        # Semantic marker styling: Green (#10b981) for #1 candidate, Amber (#f59e0b) for alternatives, Purple (#a855f7) if selected
        if is_selected:
            marker_color = "#a855f7"
            marker_radius = 11
        elif is_top:
            marker_color = "#10b981"
            marker_radius = 10
        else:
            marker_color = "#f59e0b"
            marker_radius = 7

        map_atms.append({
            "rank": int(atm.get("rank", idx)),
            "atm_id": atm_id,
            "bank": str(atm.get("bank", "Unknown")),
            "operator": str(atm.get("operator", "Unknown")),
            "city": str(atm.get("city", "")),
            "district": str(atm.get("district", "")),
            "state": str(atm.get("state", "")),
            "zone": str(atm.get("zone", "")),
            "distance_km": float(atm.get("distance_km", 0.0)),
            "overall_score": float(atm.get("overall_score", 0.0)),
            "zone_probability_score": float(atm.get("zone_probability_score", 0.0)),
            "spatial_score": float(atm.get("spatial_score", 0.0)),
            "bank_score": float(atm.get("bank_score", 50.0)),
            "time_score": float(atm.get("time_score", 50.0)),
            "activity_score": float(atm.get("activity_score", 50.0)),
            "amount_compatibility_score": float(atm.get("amount_compatibility_score", 50.0)),
            "evidence_flags": list(atm.get("evidence_flags", [])),
            "explanation": str(atm.get("explanation", "")),
            "source": str(atm.get("source", "OpenStreetMap")),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "is_top": is_top,
            "is_selected": is_selected,
            "marker_color": marker_color,
            "marker_radius": marker_radius,
            "designation": "Highest-ranked candidate" if is_top else f"Candidate #{atm.get('rank', idx)}",
        })
        bounding_points.append((lat, lon))

    # 5. Calculate Map Encompassing Bounds & Dynamic Auto-Centering
    map_bounds, map_center, map_zoom = calculate_geographic_bounds(bounding_points)

    cross_zone_active = bool(ranking.get("cross_zone_search", False))

    return {
        "case_id": str(case_analysis.get("case_id", "UNKNOWN")),
        "model_id": str(case_analysis.get("model_id", "random_forest_full_none")),
        "complaint_point": complaint_point,
        "predicted_zone": predicted_zone_info,
        "candidate_zones": candidate_zones_list,
        "zone_probabilities": zone_probs,
        "ranked_atms": map_atms,
        "total_atms_visible": len(map_atms),
        "total_atms_evaluated": int(ranking.get("total_atms_evaluated", len(raw_ranked_atms))),
        "cross_zone_search": cross_zone_active,
        "confidence_tier": conf_tier,
        "prediction_confidence": pred_conf,
        "probability_margin": margin,
        "map_bounds": map_bounds,
        "map_center": map_center,
        "map_zoom": map_zoom,
        "osm_attribution": "&copy; <a href='https://www.openstreetmap.org/copyright' target='_blank'>OpenStreetMap</a> contributors",
    }
