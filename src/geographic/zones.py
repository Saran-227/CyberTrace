"""Geographic zone definitions, centroids, and bounding boxes for CyberTrace.

Focuses on realistic geographic envelopes in Punjab / North India region.
"""

from typing import Dict, Any, List, Optional, Tuple

# Predefined geographic operational zones
ZONE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Zone_01": {
        "name": "Jalandhar Urban Core",
        "district": "Jalandhar",
        "state": "Punjab",
        "centroid": (31.3260, 75.5762),
        "radius_km": 6.5,
        "bbox": (31.2800, 75.5200, 31.3700, 75.6300),  # (min_lat, min_lon, max_lat, max_lon)
    },
    "Zone_02": {
        "name": "Jalandhar Cantt & Rama Mandi",
        "district": "Jalandhar",
        "state": "Punjab",
        "centroid": (31.2950, 75.6250),
        "radius_km": 5.0,
        "bbox": (31.2600, 75.5800, 31.3300, 75.6700),
    },
    "Zone_03": {
        "name": "Ludhiana Industrial Hub",
        "district": "Ludhiana",
        "state": "Punjab",
        "centroid": (30.9010, 75.8573),
        "radius_km": 8.0,
        "bbox": (30.8400, 75.7900, 30.9600, 75.9200),
    },
    "Zone_04": {
        "name": "Ludhiana Civil Lines & Mall Road",
        "district": "Ludhiana",
        "state": "Punjab",
        "centroid": (30.9150, 75.8350),
        "radius_km": 5.5,
        "bbox": (30.8700, 75.7900, 30.9500, 75.8800),
    },
    "Zone_05": {
        "name": "Amritsar Golden Temple Perimeter & Walled City",
        "district": "Amritsar",
        "state": "Punjab",
        "centroid": (31.6200, 74.8765),
        "radius_km": 6.0,
        "bbox": (31.5700, 74.8200, 31.6700, 74.9300),
    },
    "Zone_06": {
        "name": "Amritsar Ranjit Avenue Commercial Area",
        "district": "Amritsar",
        "state": "Punjab",
        "centroid": (31.6500, 74.8550),
        "radius_km": 5.0,
        "bbox": (31.6100, 74.8100, 31.6900, 74.9000),
    },
    "Zone_07": {
        "name": "Patiala Urban & University Sector",
        "district": "Patiala",
        "state": "Punjab",
        "centroid": (30.3398, 76.3869),
        "radius_km": 7.0,
        "bbox": (30.2900, 76.3300, 30.3900, 76.4400),
    },
    "Zone_08": {
        "name": "Chandigarh Sector 17 & 35 Commercial District",
        "district": "Chandigarh",
        "state": "Chandigarh",
        "centroid": (30.7333, 76.7794),
        "radius_km": 6.0,
        "bbox": (30.6900, 76.7300, 30.7800, 76.8300),
    },
    "Zone_09": {
        "name": "Mohali Phase 7 & 8 IT Corridor",
        "district": "SAS Nagar",
        "state": "Punjab",
        "centroid": (30.7046, 76.7179),
        "radius_km": 6.5,
        "bbox": (30.6500, 76.6700, 30.7500, 76.7600),
    },
    "Zone_10": {
        "name": "Phagwara GT Road Transit Zone",
        "district": "Kapurthala",
        "state": "Punjab",
        "centroid": (31.2215, 75.7725),
        "radius_km": 5.0,
        "bbox": (31.1800, 75.7200, 31.2600, 75.8200),
    },
}

def get_zone_info(zone_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata, centroid, and bounding box for a given zone ID."""
    return ZONE_DEFINITIONS.get(zone_id)

def get_zone_centroid(zone_id: str) -> Optional[Tuple[float, float]]:
    """Return (latitude, longitude) centroid of the zone."""
    info = get_zone_info(zone_id)
    return info["centroid"] if info else None

def get_zone_bounding_box(zone_id: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (min_lat, min_lon, max_lat, max_lon) bounding box tuple."""
    info = get_zone_info(zone_id)
    return info["bbox"] if info else None

def get_all_zones() -> List[str]:
    """Return list of all recognized zone identifiers."""
    return list(ZONE_DEFINITIONS.keys())
