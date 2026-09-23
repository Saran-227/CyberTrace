"""Unit tests for Phase 2A ATM Geographic Dataset pipeline, normalization, validation, and caching."""

from pathlib import Path
import json
import pytest
import pandas as pd

from src.config import ATM_LOCATIONS_SCHEMA, OSM_CACHE_DIR
from src.atm.osm_loader import (
    normalize_osm_element,
    get_bbox_cache_key,
    get_cached_atms,
    save_osm_cache,
    fetch_osm_atms_in_bbox,
)
from src.atm.validator import validate_atm_dataframe
from src.atm.pipeline import build_atm_locations_dataset

def test_atm_schema_validation():
    """1. Test that validator enforces exact ATM schema."""
    valid_df = pd.DataFrame([{
        "atm_id": "OSM-NODE-101",
        "bank": "HDFC",
        "operator": "HDFC",
        "latitude": 31.3260,
        "longitude": 75.5762,
        "address": "Model Town",
        "city": "Jalandhar",
        "district": "Jalandhar",
        "state": "Punjab",
        "is_24x7": True,
        "source": "OpenStreetMap",
    }])
    val = validate_atm_dataframe(valid_df)
    assert val["is_valid"] is True
    assert val["missing_columns"] == []

    # Incomplete schema
    invalid_df = valid_df.drop(columns=["operator"])
    val_inv = validate_atm_dataframe(invalid_df)
    assert val_inv["is_valid"] is False
    assert "operator" in val_inv["missing_columns"]

def test_latitude_range_validation():
    """2. Test that out-of-range latitude is flagged as invalid."""
    bad_lat_df = pd.DataFrame([{
        "atm_id": "OSM-NODE-102",
        "bank": "SBI",
        "operator": "SBI",
        "latitude": 95.0,  # Invalid: > 90
        "longitude": 75.0,
        "address": "unknown",
        "city": "Amritsar",
        "district": "Amritsar",
        "state": "Punjab",
        "is_24x7": "unknown",
        "source": "OpenStreetMap",
    }])
    val = validate_atm_dataframe(bad_lat_df)
    assert val["is_valid"] is False
    assert val["invalid_coordinates_count"] == 1

def test_longitude_range_validation():
    """3. Test that out-of-range longitude is flagged as invalid."""
    bad_lon_df = pd.DataFrame([{
        "atm_id": "OSM-NODE-103",
        "bank": "SBI",
        "operator": "SBI",
        "latitude": 31.0,
        "longitude": -195.0,  # Invalid: < -180
        "address": "unknown",
        "city": "Amritsar",
        "district": "Amritsar",
        "state": "Punjab",
        "is_24x7": "unknown",
        "source": "OpenStreetMap",
    }])
    val = validate_atm_dataframe(bad_lon_df)
    assert val["is_valid"] is False
    assert val["invalid_coordinates_count"] == 1

def test_stable_atm_id_generation():
    """4. Test that stable ATM ID is deterministically generated from OSM type and ID."""
    elem_node = {"type": "node", "id": 99887766, "lat": 31.0, "lon": 75.0, "tags": {}}
    elem_way = {"type": "way", "id": 55443322, "center": {"lat": 31.1, "lon": 75.1}, "tags": {}}

    res_node = normalize_osm_element(elem_node)
    res_way = normalize_osm_element(elem_way)

    assert res_node["atm_id"] == "OSM-NODE-99887766"
    assert res_way["atm_id"] == "OSM-WAY-55443322"

    # Regenerating same element yields identical ID
    res_node_repeat = normalize_osm_element(elem_node)
    assert res_node["atm_id"] == res_node_repeat["atm_id"]

def test_unknown_value_normalization():
    """5. Test that missing bank, operator, address, and hours normalize strictly to 'unknown' without guessing."""
    elem = {
        "type": "node",
        "id": 112233,
        "lat": 30.9,
        "lon": 75.8,
        "tags": {},  # No operator, brand, name, or opening_hours
    }
    normalized = normalize_osm_element(elem, default_city="Ludhiana", default_district="Ludhiana", default_state="Punjab")
    assert normalized["bank"] == "unknown"
    assert normalized["operator"] == "unknown"
    assert normalized["address"] == "unknown"
    assert normalized["is_24x7"] == "unknown"
    assert normalized["source"] == "OpenStreetMap"

    # With explicit tags
    elem_explicit = {
        "type": "node",
        "id": 112234,
        "lat": 30.9,
        "lon": 75.8,
        "tags": {
            "operator": "Punjab National Bank",
            "opening_hours": "24/7",
            "addr:street": "Mall Road",
        },
    }
    norm_explicit = normalize_osm_element(elem_explicit)
    assert norm_explicit["bank"] == "Punjab National Bank"
    assert norm_explicit["operator"] == "Punjab National Bank"
    assert norm_explicit["address"] == "Mall Road"
    assert norm_explicit["is_24x7"] is True

def test_duplicate_detection():
    """6. Test that duplicate ATM records are detected by validator and deduplicated in pipeline."""
    df_dupes = pd.DataFrame([
        {
            "atm_id": "OSM-NODE-200",
            "bank": "Axis Bank",
            "operator": "Axis Bank",
            "latitude": 30.7333,
            "longitude": 76.7794,
            "address": "Sector 17",
            "city": "Chandigarh",
            "district": "Chandigarh",
            "state": "Chandigarh",
            "is_24x7": True,
            "source": "OpenStreetMap",
        },
        {
            "atm_id": "OSM-NODE-200",  # Duplicate ID
            "bank": "Axis Bank",
            "operator": "Axis Bank",
            "latitude": 30.7333,
            "longitude": 76.7794,
            "address": "Sector 17",
            "city": "Chandigarh",
            "district": "Chandigarh",
            "state": "Chandigarh",
            "is_24x7": True,
            "source": "OpenStreetMap",
        },
    ])
    val = validate_atm_dataframe(df_dupes)
    assert val["duplicate_atm_ids"] == 1
    assert val["duplicate_coordinates"] == 1

def test_cache_key_generation():
    """7. Test deterministic cache key generation."""
    bbox = (31.2400, 75.4800, 31.4000, 75.6800)
    key1 = get_bbox_cache_key(bbox, name="Jalandhar")
    key2 = get_bbox_cache_key(bbox, name="Jalandhar")
    key_bbox = get_bbox_cache_key(bbox, name=None)

    assert key1 == key2
    assert key1 == "osm_atms_jalandhar"
    assert key_bbox.startswith("osm_atms_bbox_")

def test_cached_offline_loading(tmp_path, monkeypatch):
    """8. Test that cached data is retrieved locally without requiring network requests."""
    bbox = (28.45, 77.02, 28.50, 77.08)
    name = "test_offline_city"
    dummy_elements = [
        {"type": "node", "id": 7771, "lat": 28.46, "lon": 77.03, "tags": {"operator": "ICICI Bank"}},
        {"type": "node", "id": 7772, "lat": 28.47, "lon": 77.04, "tags": {}},
    ]

    # Save to cache
    save_osm_cache(bbox, dummy_elements, name=name)

    # Force network failure by mocking requests.post
    def mock_post_fail(*args, **kwargs):
        raise ConnectionError("Simulated offline network failure")

    import requests
    monkeypatch.setattr(requests, "post", mock_post_fail)

    # Fetch should successfully load from cache
    loaded = fetch_osm_atms_in_bbox(bbox, name=name, force_refresh=False)
    assert len(loaded) == 2
    assert loaded[0]["atm_id"] == "OSM-NODE-7771"
    assert loaded[0]["operator"] == "ICICI Bank"
    assert loaded[1]["atm_id"] == "OSM-NODE-7772"
    assert loaded[1]["bank"] == "unknown"

def test_all_study_cities_in_coverage_report():
    """9. Test that every configured study city appears in the coverage report."""
    from src.config import ATM_COVERAGE_REPORT_PATH
    from src.geographic.study_regions import get_all_study_cities

    assert ATM_COVERAGE_REPORT_PATH.exists(), "Coverage report file must exist"
    cov_df = pd.read_csv(ATM_COVERAGE_REPORT_PATH)
    all_cities = get_all_study_cities()

    assert len(cov_df) == len(all_cities)
    for city in all_cities:
        assert city in cov_df["city"].values, f"City '{city}' missing from coverage report"

def test_query_failure_distinguishable_from_zero_records(monkeypatch):
    """10. Test that network/query failure is strictly distinct from zero ATM records found."""
    import requests
    from src.atm.osm_loader import fetch_osm_atms_with_status

    test_bbox = (25.10, 75.10, 25.20, 75.20)

    # A: Simulated query failure (all endpoints fail / timeout)
    def mock_post_raise(*args, **kwargs):
        raise requests.ConnectionError("Simulated server outage")

    monkeypatch.setattr(requests, "post", mock_post_raise)
    monkeypatch.setattr(requests, "get", mock_post_raise)

    fail_res = fetch_osm_atms_with_status(test_bbox, name="test_failing_query", force_refresh=True)
    assert fail_res["status"] == "QUERY_FAILED"
    assert fail_res["status"] != "NO_OSM_ATMS_FOUND"

    # B: Simulated successful response with 0 ATMs
    class MockSuccessEmptyResponse:
        status_code = 200
        def json(self):
            return {"elements": []}

    def mock_post_empty(*args, **kwargs):
        return MockSuccessEmptyResponse()

    monkeypatch.setattr(requests, "post", mock_post_empty)
    empty_res = fetch_osm_atms_with_status(test_bbox, name="test_empty_query", force_refresh=True)
    assert empty_res["status"] == "NO_OSM_ATMS_FOUND"
    assert empty_res["status"] != "QUERY_FAILED"

def test_bounding_boxes_are_valid():
    """11. Test that all configured bounding boxes in study_regions are geographically valid."""
    from src.geographic.study_regions import STUDY_REGIONS
    from src.atm.osm_loader import is_valid_bbox

    for city_name, reg in STUDY_REGIONS.items():
        bbox = reg["bbox"]
        assert is_valid_bbox(bbox), f"Bounding box for '{city_name}' is invalid: {bbox}"
        min_lat, min_lon, max_lat, max_lon = bbox
        assert min_lat < max_lat
        assert min_lon < max_lon
        assert -90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0
        assert -180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0

        # Centroid must lie within bbox
        c_lat, c_lon = reg["centroid"]
        assert min_lat <= c_lat <= max_lat, f"Centroid lat {c_lat} outside bbox for '{city_name}'"
        assert min_lon <= c_lon <= max_lon, f"Centroid lon {c_lon} outside bbox for '{city_name}'"

def test_no_atm_outside_study_region():
    """12. Test that every ATM in atm_locations.csv is located within its city bounding box."""
    from src.config import ATM_LOCATIONS_PROCESSED_PATH
    from src.geographic.study_regions import STUDY_REGIONS

    assert ATM_LOCATIONS_PROCESSED_PATH.exists()
    df = pd.read_csv(ATM_LOCATIONS_PROCESSED_PATH)
    assert not df.empty

    for _, row in df.iterrows():
        city = row["city"]
        reg = STUDY_REGIONS.get(city)
        assert reg is not None, f"City '{city}' not in configured study regions"
        min_lat, min_lon, max_lat, max_lon = reg["bbox"]
        lat = row["latitude"]
        lon = row["longitude"]
        assert min_lat <= lat <= max_lat, f"ATM {row['atm_id']} latitude {lat} outside {city} bbox"
        assert min_lon <= lon <= max_lon, f"ATM {row['atm_id']} longitude {lon} outside {city} bbox"

def test_coverage_report_schema_valid():
    """13. Test that reports/atm_coverage_report.csv conforms strictly to required schema and valid statuses."""
    from src.config import ATM_COVERAGE_REPORT_PATH, ATM_COVERAGE_SCHEMA

    assert ATM_COVERAGE_REPORT_PATH.exists()
    cov_df = pd.read_csv(ATM_COVERAGE_REPORT_PATH)

    # 1. Exact columns
    assert list(cov_df.columns) == ATM_COVERAGE_SCHEMA

    # 2. Valid statuses
    valid_statuses = {"OK", "NO_OSM_ATMS_FOUND", "QUERY_FAILED", "INVALID_BBOX", "PARSE_FAILED"}
    for status in cov_df["status"]:
        assert status in valid_statuses, f"Invalid coverage report status: {status}"

    # 3. Numeric non-negative counts
    assert (cov_df["raw_osm_objects"] >= 0).all()
    assert (cov_df["atm_records"] >= 0).all()
    assert (cov_df["known_bank_operator"] >= 0).all()
    assert (cov_df["unknown_bank_operator"] >= 0).all()

