"""Candidate ATM discovery module for predicted cash-out zones."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.geographic.zones import get_zone_bounding_box, get_zone_centroid
from src.geographic.distance import haversine_distance
from src.atm.osm_loader import fetch_osm_atms_in_bbox
from src.data.loader import load_atm_locations
from src.utils.logging import get_logger

logger = get_logger("ATMDiscovery")

def discover_candidate_atms(
    predicted_zone: str,
    complaint_lat: Optional[float] = None,
    complaint_lon: Optional[float] = None,
    use_synthetic_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Discover all candidate ATMs within the predicted withdrawal zone.

    Pulls from OpenStreetMap cache / API, and falls back to local ATM dataset if offline.
    """
    bbox = get_zone_bounding_box(predicted_zone)
    if not bbox:
        logger.warning(f"No bounding box defined for zone '{predicted_zone}'")
        return []

    # Attempt to fetch OSM ATMs within bounding box
    atms = fetch_osm_atms_in_bbox(bbox)

    # If empty, check local processed atm_locations.csv
    if not atms:
        local_df = load_atm_locations()
        if local_df is not None and not local_df.empty:
            min_lat, min_lon, max_lat, max_lon = bbox
            filtered = local_df[
                (local_df["latitude"] >= min_lat)
                & (local_df["latitude"] <= max_lat)
                & (local_df["longitude"] >= min_lon)
                & (local_df["longitude"] <= max_lon)
            ]
            atms = filtered.to_dict(orient="records")

    # If still empty and synthetic fallback allowed during development, generate clearly labeled synthetic records
    if not atms and use_synthetic_fallback:
        centroid = get_zone_centroid(predicted_zone)
        if centroid:
            c_lat, c_lon = centroid
            sample_banks = ["HDFC", "SBI", "ICICI", "Axis Bank", "Punjab National Bank"]
            atms = []
            for i, b in enumerate(sample_banks):
                atms.append({
                    "atm_id": f"DEV-ATM-{predicted_zone}-{i+1:02d}",
                    "bank": b,
                    "operator": b,
                    "latitude": round(c_lat + (i * 0.005 - 0.01), 6),
                    "longitude": round(c_lon + (i * 0.005 - 0.01), 6),
                    "address": f"Market Complex Road, {predicted_zone}",
                    "city": "Operational Area",
                    "district": "Local Jurisdiction",
                    "state": "Punjab",
                    "is_24x7": True,
                    "source": "synthetic_dev_stub",
                })

    # Compute distances to reference coordinate (complaint or zone centroid)
    ref_lat = complaint_lat if complaint_lat is not None else (bbox[0] + bbox[2]) / 2.0
    ref_lon = complaint_lon if complaint_lon is not None else (bbox[1] + bbox[3]) / 2.0

    for atm in atms:
        lat = atm.get("latitude")
        lon = atm.get("longitude")
        if lat is not None and lon is not None:
            atm["distance_km"] = haversine_distance(ref_lat, ref_lon, float(lat), float(lon))
        else:
            atm["distance_km"] = 999.0

    logger.info(f"Discovered {len(atms)} candidate ATMs for {predicted_zone}")
    return atms
