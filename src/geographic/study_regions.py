"""Study regions configuration for CyberTrace ATM geographic dataset.

Defines bounding boxes, centroids, and administrative hierarchy for target cities
across Punjab, Chandigarh, Haryana, Delhi, Uttar Pradesh, and Rajasthan.
Designed for easy extension and tuning.
"""

from typing import Dict, Any, List, Tuple

# Bounding box format: (min_lat, min_lon, max_lat, max_lon)
STUDY_REGIONS: Dict[str, Dict[str, Any]] = {
    # --- Punjab ---
    "Jalandhar": {
        "city": "Jalandhar",
        "district": "Jalandhar",
        "state": "Punjab",
        "centroid": (31.3260, 75.5762),
        "bbox": (31.2400, 75.4800, 31.4000, 75.6800),
    },
    "Ludhiana": {
        "city": "Ludhiana",
        "district": "Ludhiana",
        "state": "Punjab",
        "centroid": (30.9010, 75.8573),
        "bbox": (30.8200, 75.7600, 31.0000, 75.9600),
    },
    "Amritsar": {
        "city": "Amritsar",
        "district": "Amritsar",
        "state": "Punjab",
        "centroid": (31.6340, 74.8723),
        "bbox": (31.5600, 74.7800, 31.7200, 74.9600),
    },
    "Patiala": {
        "city": "Patiala",
        "district": "Patiala",
        "state": "Punjab",
        "centroid": (30.3398, 76.3869),
        "bbox": (30.2700, 76.3000, 30.4200, 76.4700),
    },

    # --- Chandigarh (UT) ---
    "Chandigarh": {
        "city": "Chandigarh",
        "district": "Chandigarh",
        "state": "Chandigarh",
        "centroid": (30.7333, 76.7794),
        "bbox": (30.6800, 76.7100, 30.8000, 76.8500),
    },

    # --- Haryana ---
    "Ambala": {
        "city": "Ambala",
        "district": "Ambala",
        "state": "Haryana",
        "centroid": (30.3782, 76.7767),
        "bbox": (30.3100, 76.7000, 30.4500, 76.8600),
    },
    "Panipat": {
        "city": "Panipat",
        "district": "Panipat",
        "state": "Haryana",
        "centroid": (29.3909, 76.9635),
        "bbox": (29.3300, 76.9000, 29.4600, 77.0300),
    },
    "Gurugram": {
        "city": "Gurugram",
        "district": "Gurugram",
        "state": "Haryana",
        "centroid": (28.4595, 77.0266),
        "bbox": (28.3700, 76.9300, 28.5300, 77.1200),
    },
    "Faridabad": {
        "city": "Faridabad",
        "district": "Faridabad",
        "state": "Haryana",
        "centroid": (28.4089, 77.3178),
        "bbox": (28.3200, 77.2400, 28.4800, 77.3900),
    },

    # --- Delhi (UT/NCT) ---
    "New Delhi": {
        "city": "New Delhi",
        "district": "New Delhi",
        "state": "Delhi",
        "centroid": (28.6139, 77.2090),
        "bbox": (28.5200, 77.1200, 28.7000, 77.3000),
    },

    # --- Uttar Pradesh ---
    "Noida": {
        "city": "Noida",
        "district": "Gautam Buddha Nagar",
        "state": "Uttar Pradesh",
        "centroid": (28.5355, 77.3910),
        "bbox": (28.4600, 77.3000, 28.6200, 77.4800),
    },
    "Ghaziabad": {
        "city": "Ghaziabad",
        "district": "Ghaziabad",
        "state": "Uttar Pradesh",
        "centroid": (28.6692, 77.4538),
        "bbox": (28.6000, 77.3600, 28.7500, 77.5300),
    },
    "Meerut": {
        "city": "Meerut",
        "district": "Meerut",
        "state": "Uttar Pradesh",
        "centroid": (28.9845, 77.7064),
        "bbox": (28.9000, 77.6200, 29.0600, 77.7900),
    },

    # --- Rajasthan ---
    "Jaipur": {
        "city": "Jaipur",
        "district": "Jaipur",
        "state": "Rajasthan",
        "centroid": (26.9124, 75.7873),
        "bbox": (26.8200, 75.6900, 27.0200, 75.8900),
    },
    "Alwar": {
        "city": "Alwar",
        "district": "Alwar",
        "state": "Rajasthan",
        "centroid": (27.5530, 76.6346),
        "bbox": (27.5000, 76.5600, 27.6200, 76.6900),
    },
}

def get_study_region(city_name: str) -> Dict[str, Any]:
    """Retrieve metadata and bounding box for a study region."""
    return STUDY_REGIONS.get(city_name)

def get_all_study_cities() -> List[str]:
    """List all configured study city names."""
    return list(STUDY_REGIONS.keys())
