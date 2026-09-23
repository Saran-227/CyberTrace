"""OpenStreetMap / Overpass API Ingestion and Local Caching Layer.

Fetches ATM locations via Overpass API / OpenStreetMap API, stores deterministic local
JSON caches under data/external/osm/, normalizes records with strict data fidelity
(zero inference of banks), and fails gracefully when offline without crashing.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import requests

from src.config import (
    OSM_CACHE_DIR,
    OVERPASS_API_URL,
    OVERPASS_ENDPOINTS,
    OVERPASS_TIMEOUT,
    OVERPASS_CACHE_TTL_HOURS,
)
from src.utils.logging import get_logger

logger = get_logger("OSMLoader")

OSM_HEADERS = {
    "User-Agent": "CyberTrace-Academic-Intelligence/1.0 (LocationResearch; mailto:contact@cybertrace.local)",
    "Accept": "application/json, application/xml",
}

def is_valid_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate bounding box coordinate format and spatial bounds."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    min_lat, min_lon, max_lat, max_lon = bbox
    try:
        min_lat, min_lon, max_lat, max_lon = float(min_lat), float(min_lon), float(max_lat), float(max_lon)
    except (ValueError, TypeError):
        return False
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        return False
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        return False
    if min_lat >= max_lat or min_lon >= max_lon:
        return False
    return True

def get_bbox_cache_key(bbox: Tuple[float, float, float, float], name: Optional[str] = None) -> str:
    """Generate deterministic cache key for a bounding box or named region."""
    if name:
        safe_name = name.strip().lower().replace(" ", "_").replace("/", "_")
        return f"osm_atms_{safe_name}"
    bbox_str = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    h = hashlib.sha256(bbox_str.encode("utf-8")).hexdigest()[:12]
    return f"osm_atms_bbox_{h}"

def get_cached_atms(
    bbox: Tuple[float, float, float, float],
    name: Optional[str] = None,
    allow_expired: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Retrieve locally cached Overpass/OSM ATM raw elements if available."""
    cache_key = get_bbox_cache_key(bbox, name=name)
    cache_file = OSM_CACHE_DIR / f"{cache_key}.json"

    # Also check fallback hash filename if named file doesn't exist
    if not cache_file.exists() and name:
        alt_key = get_bbox_cache_key(bbox, name=None)
        alt_file = OSM_CACHE_DIR / f"{alt_key}.json"
        if alt_file.exists():
            cache_file = alt_file

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("timestamp", 0)
            ttl_seconds = OVERPASS_CACHE_TTL_HOURS * 3600
            is_fresh = (time.time() - cached_at) < ttl_seconds

            if is_fresh or allow_expired:
                elements = data.get("elements", [])
                logger.info(
                    f"Loaded {len(elements)} ATM elements from local OSM cache "
                    f"({cache_file.name}, fresh={is_fresh})."
                )
                return elements
            else:
                logger.info(f"OSM cache in {cache_file.name} is older than TTL. Network refresh candidate.")
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_file}: {e}")
    return None

def save_osm_cache(
    bbox: Tuple[float, float, float, float],
    elements: List[Dict[str, Any]],
    name: Optional[str] = None,
    status: str = "OK",
    notes: str = "",
) -> Path:
    """Save OpenStreetMap query results to local disk cache deterministically."""
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_bbox_cache_key(bbox, name=name)
    cache_file = OSM_CACHE_DIR / f"{cache_key}.json"
    payload = {
        "region_name": name or "bbox_query",
        "bbox": bbox,
        "timestamp": time.time(),
        "status": status,
        "notes": notes,
        "count": len(elements),
        "elements": elements,
    }
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved {len(elements)} OSM ATM nodes to {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to write OSM cache: {e}")
    return cache_file

def normalize_osm_element(
    elem: Dict[str, Any],
    default_city: str = "",
    default_district: str = "",
    default_state: str = "",
) -> Dict[str, Any]:
    """Convert raw OpenStreetMap node/way into standard CyberTrace ATM schema.

    Strict Data Rules:
    - Never invent bank or operator names. If unavailable, use 'unknown'.
    - Never infer bank from branch proximity or neighborhood.
    - is_24x7 is True ONLY when explicitly stated in tags (e.g. opening_hours=24/7),
      False when non-24/7 schedule given, otherwise 'unknown'.
    - Address is constructed solely from explicit addr:* tags or 'unknown'.
    - Stable atm_id derived from OSM element type and numeric ID.
    """
    tags = elem.get("tags", {})
    osm_type = str(elem.get("type", "node")).upper()
    osm_id = elem.get("id")

    # Extract coordinates (handles nodes and way/relation centers)
    lat = elem.get("lat")
    lon = elem.get("lon")
    if (lat is None or lon is None) and "center" in elem:
        lat = elem["center"].get("lat")
        lon = elem["center"].get("lon")

    # Generate deterministic, stable ATM identifier
    if osm_id is not None:
        stable_id = f"OSM-{osm_type}-{osm_id}"
    else:
        # Fallback hash of coordinates if ID absent
        coord_hash = hashlib.sha256(f"{lat}_{lon}".encode("utf-8")).hexdigest()[:10]
        stable_id = f"OSM-GEO-{coord_hash}"

    # Extract operator / brand / bank strictly without guessing
    raw_operator = tags.get("operator", "").strip()
    raw_brand = tags.get("brand", "").strip()
    raw_name = tags.get("name", "").strip()

    # Operator normalization
    if raw_operator:
        operator = raw_operator
    elif raw_brand:
        operator = raw_brand
    else:
        operator = "unknown"

    # Bank normalization
    if raw_brand:
        bank = raw_brand
    elif raw_operator:
        bank = raw_operator
    elif raw_name and not raw_name.lower().startswith("atm"):
        bank = raw_name
    elif raw_name and "bank" in raw_name.lower():
        bank = raw_name
    else:
        bank = "unknown"

    # Normalize invalid or placeholder strings
    if bank.lower() in ["", "none", "null", "nan", "unspecified", "atm"]:
        bank = "unknown"
    if operator.lower() in ["", "none", "null", "nan", "unspecified", "atm"]:
        operator = "unknown"

    # Address construction from explicit OSM tags only
    addr_parts = []
    for k in ["addr:housenumber", "addr:street", "addr:suburb", "addr:locality"]:
        val = tags.get(k, "").strip()
        if val:
            addr_parts.append(val)
    address = ", ".join(addr_parts) if addr_parts else "unknown"

    # Administrative geography (tags first, study region defaults second)
    city = tags.get("addr:city", "").strip() or default_city or "unknown"
    if city.lower() in ["gurgaon", "gurugram"]:
        city = "Gurugram"
    district = tags.get("addr:district", "").strip() or default_district or city
    state = tags.get("addr:state", "").strip() or default_state or "unknown"

    # 24/7 verification
    hours = tags.get("opening_hours", "").strip().lower()
    if hours in ["24/7", "24x7", "24 hours", "all day", "open 24/7", "24/7 hours"]:
        is_24x7: Union[bool, str] = True
    elif hours:
        is_24x7 = False
    else:
        is_24x7 = "unknown"

    return {
        "atm_id": stable_id,
        "bank": bank,
        "operator": operator,
        "latitude": round(float(lat), 6) if lat is not None else None,
        "longitude": round(float(lon), 6) if lon is not None else None,
        "address": address,
        "city": city,
        "district": district,
        "state": state,
        "is_24x7": is_24x7,
        "source": "OpenStreetMap",
    }

def _fetch_from_osm_api_subbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Query official OpenStreetMap API 0.6 directly within sub-bounding box."""
    # OSM API 0.6 expects: left(min_lon), bottom(min_lat), right(max_lon), top(max_lat)
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
    try:
        resp = requests.get(url, headers=OSM_HEADERS, timeout=12)
        if resp.status_code == 200:
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as parse_err:
                return [], "PARSE_FAILED", f"XML ParseError: {parse_err}"
            elements = []
            for node in root.findall("node"):
                tags = {tag.get("k"): tag.get("v") for tag in node.findall("tag")}
                # Capture amenity=atm or bank with atm=yes
                if tags.get("amenity") == "atm" or (tags.get("amenity") == "bank" and tags.get("atm") == "yes"):
                    elements.append({
                        "type": "node",
                        "id": int(node.get("id")),
                        "lat": float(node.get("lat")),
                        "lon": float(node.get("lon")),
                        "tags": tags,
                    })
            if elements:
                return elements, "OK", f"Retrieved {len(elements)} elements from OSM API 0.6"
            return [], "NO_OSM_ATMS_FOUND", "OSM API 0.6 returned 0 ATMs in subbox"
        else:
            return [], "QUERY_FAILED", f"OSM API returned HTTP {resp.status_code}"
    except Exception as exc:
        return [], "QUERY_FAILED", f"OSM API subbox query failed: {exc}"

def _query_overpass_or_osm(
    bbox: Tuple[float, float, float, float],
    name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Query OpenStreetMap data using Overpass API endpoint pool with direct OSM API fallback.

    Returns:
        (elements, status, notes)
        status is one of: OK, NO_OSM_ATMS_FOUND, QUERY_FAILED, INVALID_BBOX, PARSE_FAILED
    """
    if not is_valid_bbox(bbox):
        return [], "INVALID_BBOX", f"Bounding box {bbox} is invalid or has inverted coordinates"

    min_lat, min_lon, max_lat, max_lon = bbox

    # 1. Primary: Overpass QL with multiple mirror endpoints
    overpass_query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    (
      node["amenity"="atm"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["amenity"="bank"]["atm"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["amenity"="atm"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """

    last_error = ""
    had_successful_empty = False
    for ep in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(ep, data={"data": overpass_query}, headers=OSM_HEADERS, timeout=OVERPASS_TIMEOUT)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as parse_err:
                    last_error = f"JSON parsing failed from {ep}: {parse_err}"
                    continue
                elements = data.get("elements", [])
                if elements:
                    return elements, "OK", f"Retrieved {len(elements)} elements from {ep}"
                else:
                    had_successful_empty = True
                    last_error = f"0 ATM elements returned from {ep}"
            else:
                last_error = f"HTTP {resp.status_code} from {ep}"
        except requests.Timeout:
            last_error = f"Timeout ({OVERPASS_TIMEOUT}s) from {ep}"
        except Exception as exc:
            last_error = f"{type(exc).__name__} from {ep}: {exc}"

    # 2. Fallback: Query OpenStreetMap official API 0.6 across central subbox
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0
    core_span = 0.015  # ~1.5km box to stay within node limits
    sub_elems, sub_status, sub_note = _fetch_from_osm_api_subbox(
        mid_lat - core_span, mid_lon - core_span, mid_lat + core_span, mid_lon + core_span
    )
    if sub_status == "OK" and sub_elems:
        return sub_elems, "OK", f"Retrieved {len(sub_elems)} elements from OSM API 0.6 fallback"
    if sub_status == "PARSE_FAILED":
        return [], "PARSE_FAILED", sub_note

    if had_successful_empty:
        return [], "NO_OSM_ATMS_FOUND", "Query succeeded across endpoints but 0 ATM elements found"

    return [], "QUERY_FAILED", f"All query attempts failed. Last error: {last_error}"

def fetch_osm_atms_with_status(
    bbox: Tuple[float, float, float, float],
    name: Optional[str] = None,
    force_refresh: bool = False,
    default_city: str = "",
    default_district: str = "",
    default_state: str = "",
) -> Dict[str, Any]:
    """Query OpenStreetMap/Overpass API and return structured status and records.

    Returns dict with keys:
    - status: OK, NO_OSM_ATMS_FOUND, QUERY_FAILED, INVALID_BBOX, PARSE_FAILED
    - elements: List of raw OSM elements
    - records: List of normalized ATM records
    - raw_count: int
    - notes: str
    - cache_hit: bool
    """
    if not is_valid_bbox(bbox):
        return {
            "status": "INVALID_BBOX",
            "elements": [],
            "records": [],
            "raw_count": 0,
            "notes": f"Bounding box {bbox} is invalid or has inverted coordinates",
            "cache_hit": False,
        }

    if not force_refresh:
        cached_elements = get_cached_atms(bbox, name=name, allow_expired=False)
        if cached_elements is not None:
            records = [
                normalize_osm_element(
                    e,
                    default_city=default_city,
                    default_district=default_district,
                    default_state=default_state,
                )
                for e in cached_elements
            ]
            status = "OK" if cached_elements else "NO_OSM_ATMS_FOUND"
            return {
                "status": status,
                "elements": cached_elements,
                "records": records,
                "raw_count": len(cached_elements),
                "notes": f"Loaded from local cache ({name or 'bbox'})",
                "cache_hit": True,
            }

    logger.info(f"Querying OpenStreetMap/Overpass for region '{name or 'bbox'}' {bbox}...")
    elements, status, notes = _query_overpass_or_osm(bbox, name=name)

    if status in ["OK", "NO_OSM_ATMS_FOUND"]:
        save_osm_cache(bbox, elements, name=name, status=status, notes=notes)
        records = [
            normalize_osm_element(
                e,
                default_city=default_city,
                default_district=default_district,
                default_state=default_state,
            )
            for e in elements
        ]
        return {
            "status": status,
            "elements": elements,
            "records": records,
            "raw_count": len(elements),
            "notes": notes,
            "cache_hit": False,
        }

    # Offline / Stale Cache fallback on failure
    fallback_elements = get_cached_atms(bbox, name=name, allow_expired=True)
    if fallback_elements is not None:
        logger.info(f"Operating in offline fallback mode using cached data for '{name or 'bbox'}'.")
        records = [
            normalize_osm_element(
                e,
                default_city=default_city,
                default_district=default_district,
                default_state=default_state,
            )
            for e in fallback_elements
        ]
        return {
            "status": "OK" if fallback_elements else "NO_OSM_ATMS_FOUND",
            "elements": fallback_elements,
            "records": records,
            "raw_count": len(fallback_elements),
            "notes": f"Offline fallback from expired cache. Query issue: {status} ({notes})",
            "cache_hit": True,
        }

    return {
        "status": status,
        "elements": [],
        "records": [],
        "raw_count": 0,
        "notes": notes,
        "cache_hit": False,
    }

def fetch_osm_atms_in_bbox(
    bbox: Tuple[float, float, float, float],
    name: Optional[str] = None,
    force_refresh: bool = False,
    default_city: str = "",
    default_district: str = "",
    default_state: str = "",
) -> List[Dict[str, Any]]:
    """Backwards-compatible wrapper returning normalized records."""
    res = fetch_osm_atms_with_status(
        bbox=bbox,
        name=name,
        force_refresh=force_refresh,
        default_city=default_city,
        default_district=default_district,
        default_state=default_state,
    )
    return res["records"]

