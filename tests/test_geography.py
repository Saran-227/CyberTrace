"""Unit tests for geographic calculations, distance, and zone bounding boxes."""

import pytest
from src.geographic.distance import haversine_distance, compute_distance_matrix
from src.geographic.zones import get_zone_centroid, get_zone_bounding_box, get_all_zones

def test_haversine_distance_accuracy():
    """Verify distance computation against known geographic points."""
    # Jalandhar (31.3260, 75.5762) to Ludhiana (30.9010, 75.8573) is ~54-60 km
    dist = haversine_distance(31.3260, 75.5762, 30.9010, 75.8573)
    assert 50.0 < dist < 65.0
    # Zero distance for identical points
    assert haversine_distance(31.0, 75.0, 31.0, 75.0) == 0.0

def test_distance_matrix():
    """Verify distance matrix from single origin to multiple destinations."""
    origin = (31.0, 75.0)
    dests = [(31.0, 75.0), (31.5, 75.5)]
    matrix = compute_distance_matrix(origin, dests)
    assert len(matrix) == 2
    assert matrix[0] == 0.0
    assert matrix[1] > 0.0

def test_zone_definitions_integrity():
    """Verify all defined zones have valid centroids and bounding boxes."""
    zones = get_all_zones()
    assert len(zones) >= 10
    for z in zones:
        centroid = get_zone_centroid(z)
        bbox = get_zone_bounding_box(z)
        assert centroid is not None and len(centroid) == 2
        assert bbox is not None and len(bbox) == 4
        min_lat, min_lon, max_lat, max_lon = bbox
        assert min_lat < max_lat
        assert min_lon < max_lon
