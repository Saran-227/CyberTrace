"""Geographic distance computation utilities using Haversine formula."""

import math
from typing import List, Tuple
import numpy as np

EARTH_RADIUS_KM = 6371.0088

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(EARTH_RADIUS_KM * c, 3)

def compute_distance_matrix(
    origin: Tuple[float, float],
    destinations: List[Tuple[float, float]],
) -> List[float]:
    """Calculate distances from a single origin (lat, lon) to a list of destinations."""
    lat1, lon1 = origin
    return [haversine_distance(lat1, lon1, lat2, lon2) for lat2, lon2 in destinations]
