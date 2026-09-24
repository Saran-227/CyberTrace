"""Unit and integration tests for CyberTrace Phase 3: Geographic Analysis & Target Validation.

Covers all 17 Part 21 requirements:
1. Exactly 20,000 complaint records.
2. Exactly 10 withdrawal zones.
3. Zone counts sum to 20,000.
4. Zone percentages sum approximately to 100%.
5. All zone coordinates are valid.
6. Zone centroids are valid.
7. Zone centroid lies within its descriptive bounding box.
8. City-zone crosstab totals equal 20,000.
9. Bank-zone crosstab totals equal 20,000.
10. Transaction-zone crosstab totals equal 20,000.
11. Fraud-zone crosstab totals equal 20,000.
12. No hidden cash-out coordinates are used.
13. ATM source dataset is not modified.
14. ATM coordinates remain valid.
15. Coverage report includes all configured zones/cities.
16. Generated report files exist.
17. No NaN/inf values occur in required geographic outputs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    ATM_LOCATIONS_PROCESSED_PATH,
    ATM_ACTIVITY_PROCESSED_PATH,
    REPORTS_DIR,
)

FIGURES_DIR = REPORTS_DIR / "figures"

@pytest.fixture(scope="module")
def complaints_df():
    """Load processed complaints dataset."""
    assert COMPLAINTS_PROCESSED_PATH.exists()
    return pd.read_csv(COMPLAINTS_PROCESSED_PATH)

@pytest.fixture(scope="module")
def atms_df():
    """Load processed ATM locations dataset."""
    assert ATM_LOCATIONS_PROCESSED_PATH.exists()
    return pd.read_csv(ATM_LOCATIONS_PROCESSED_PATH)

@pytest.fixture(scope="module")
def zone_stats_df():
    """Load generated zone statistics."""
    path = REPORTS_DIR / "zone_statistics.csv"
    assert path.exists()
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def zone_centroids_df():
    """Load generated zone centroids."""
    path = REPORTS_DIR / "zone_centroids.csv"
    assert path.exists()
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def zone_bboxes_df():
    """Load generated zone bounding boxes."""
    path = REPORTS_DIR / "zone_bounding_boxes.csv"
    assert path.exists()
    return pd.read_csv(path)


def test_01_exactly_20000_complaint_records(complaints_df):
    """1. Test that the complaint dataset contains exactly 20,000 records."""
    assert len(complaints_df) == 20000


def test_02_exactly_10_withdrawal_zones(complaints_df):
    """2. Test that exactly 10 withdrawal zones exist (Zone_01 through Zone_10)."""
    zones = sorted(complaints_df["withdrawal_zone"].unique())
    expected = [f"Zone_{i:02d}" for i in range(1, 11)]
    assert zones == expected


def test_03_zone_counts_sum_to_20000(zone_stats_df):
    """3. Test that zone counts sum to 20,000."""
    total_count = int(zone_stats_df["complaint_count"].sum())
    assert total_count == 20000


def test_04_zone_percentages_sum_to_100(zone_stats_df):
    """4. Test that zone percentages sum approximately to 100%."""
    total_pct = float(zone_stats_df["percentage"].sum())
    assert np.isclose(total_pct, 100.0, atol=0.01)


def test_05_all_zone_coordinates_valid(complaints_df):
    """5. Test that all latitude and longitude coordinates are physically and geographically valid."""
    lats = complaints_df["complaint_latitude"]
    lons = complaints_df["complaint_longitude"]

    assert (lats >= -90.0).all() and (lats <= 90.0).all()
    assert (lons >= -180.0).all() and (lons <= 180.0).all()

    # Bounded in North India study envelope (approx 25N-33N, 73E-79E)
    assert (lats >= 25.0).all() and (lats <= 33.0).all()
    assert (lons >= 73.0).all() and (lons <= 79.0).all()


def test_06_zone_centroids_valid(zone_centroids_df):
    """6. Test that every zone centroid is valid and within legitimate bounds."""
    assert len(zone_centroids_df) == 10
    c_lats = zone_centroids_df["centroid_latitude"]
    c_lons = zone_centroids_df["centroid_longitude"]

    assert (c_lats >= 26.0).all() and (c_lats <= 32.5).all()
    assert (c_lons >= 74.0).all() and (c_lons <= 78.5).all()


def test_07_zone_centroid_within_bounding_box(zone_stats_df):
    """7. Test that each zone centroid lies strictly within its descriptive bounding box."""
    for _, row in zone_stats_df.iterrows():
        z = row["withdrawal_zone"]
        c_lat, c_lon = row["centroid_latitude"], row["centroid_longitude"]
        min_lat, max_lat = row["min_latitude"], row["max_latitude"]
        min_lon, max_lon = row["min_longitude"], row["max_longitude"]

        assert min_lat <= c_lat <= max_lat, f"{z} centroid lat {c_lat} not in [{min_lat}, {max_lat}]"
        assert min_lon <= c_lon <= max_lon, f"{z} centroid lon {c_lon} not in [{min_lon}, {max_lon}]"


def test_08_city_zone_crosstab_total_20000():
    """8. Test that city-zone crosstab totals exactly 20,000."""
    path = REPORTS_DIR / "city_zone_crosstab.csv"
    assert path.exists()
    ct = pd.read_csv(path, index_col=0)
    assert "Total" in ct.index and "Total" in ct.columns
    assert int(ct.loc["Total", "Total"]) == 20000


def test_09_bank_zone_crosstab_total_20000():
    """9. Test that bank-zone crosstab totals exactly 20,000."""
    path = REPORTS_DIR / "bank_zone_crosstab.csv"
    assert path.exists()
    ct = pd.read_csv(path, index_col=0)
    assert "Total" in ct.index and "Total" in ct.columns
    assert int(ct.loc["Total", "Total"]) == 20000


def test_10_transaction_zone_crosstab_total_20000():
    """10. Test that transaction-zone crosstab totals exactly 20,000."""
    path = REPORTS_DIR / "transaction_type_zone_crosstab.csv"
    assert path.exists()
    ct = pd.read_csv(path, index_col=0)
    assert "Total" in ct.index and "Total" in ct.columns
    assert int(ct.loc["Total", "Total"]) == 20000


def test_11_fraud_zone_crosstab_total_20000():
    """11. Test that fraud-zone crosstab totals exactly 20,000."""
    path = REPORTS_DIR / "fraud_type_zone_crosstab.csv"
    assert path.exists()
    ct = pd.read_csv(path, index_col=0)
    assert "Total" in ct.index and "Total" in ct.columns
    assert int(ct.loc["Total", "Total"]) == 20000


def test_12_no_hidden_cashout_coordinates_used(complaints_df):
    """12. Test that hidden synthetic cash-out coordinates are strictly excluded from dataset and analysis."""
    forbidden = ["synthetic_cashout_latitude", "synthetic_cashout_longitude"]
    for col in forbidden:
        assert col not in complaints_df.columns, f"Target leakage! {col} found in processed dataset."

    # Also check generated reports
    assoc_df = pd.read_csv(REPORTS_DIR / "feature_target_association.csv")
    assert not assoc_df["feature"].isin(forbidden).any()


def test_13_atm_source_dataset_not_modified(atms_df):
    """13. Test that ATM source dataset has exactly 333 verified records and 14/15 cities."""
    assert len(atms_df) == 333
    assert atms_df["atm_id"].nunique() == 333
    assert atms_df["city"].nunique() == 14
    assert "Panipat" not in atms_df["city"].values


def test_14_atm_coordinates_remain_valid(atms_df):
    """14. Test that all ATM coordinates remain within legitimate WGS-84 ranges."""
    assert (atms_df["latitude"] >= -90.0).all() and (atms_df["latitude"] <= 90.0).all()
    assert (atms_df["longitude"] >= -180.0).all() and (atms_df["longitude"] <= 180.0).all()


def test_15_coverage_report_includes_all_configured_cities():
    """15. Test that city ATM coverage report includes all 15 study cities."""
    path = REPORTS_DIR / "city_atm_coverage.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 15
    assert "Panipat" in df["city"].values
    panipat_row = df[df["city"] == "Panipat"].iloc[0]
    assert panipat_row["atm_count"] == 0
    assert panipat_row["coverage_status"] == "NO_ATM_DATA"


def test_16_generated_report_files_exist():
    """16. Test that all required Phase 3 report CSV/TXT/MD files and figures exist."""
    required_csvs = [
        "phase3_dataset_audit.csv",
        "phase3_dataset_audit.txt",
        "zone_statistics.csv",
        "zone_centroids.csv",
        "zone_bounding_boxes.csv",
        "zone_centroid_distance_matrix.csv",
        "zone_separation_summary.csv",
        "zone_overlap_matrix.csv",
        "city_zone_crosstab.csv",
        "city_zone_percentage.csv",
        "bank_zone_crosstab.csv",
        "bank_zone_percentage.csv",
        "transaction_type_zone_crosstab.csv",
        "transaction_type_zone_percentage.csv",
        "fraud_type_zone_crosstab.csv",
        "fraud_type_zone_percentage.csv",
        "hour_zone_distribution.csv",
        "day_zone_distribution.csv",
        "amount_zone_statistics.csv",
        "feature_target_association.csv",
        "zone_atm_coverage.csv",
        "atm_bank_coverage.csv",
        "city_atm_coverage.csv",
        "class_imbalance_summary.csv",
        "phase3_target_assessment.md",
    ]
    for filename in required_csvs:
        p = REPORTS_DIR / filename
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty report file: {filename}"

    for i in range(1, 11):
        fig_name = f"{i:02d}_"
        matching = list(FIGURES_DIR.glob(f"{fig_name}*.png"))
        assert len(matching) == 1, f"Missing figure {i:02d} in reports/figures/"


def test_17_no_nan_or_inf_in_required_geographic_outputs():
    """17. Test that no NaN or infinite values occur in required geographic outputs."""
    for filename in ["zone_statistics.csv", "zone_centroids.csv", "zone_bounding_boxes.csv", "zone_centroid_distance_matrix.csv", "zone_separation_summary.csv"]:
        df = pd.read_csv(REPORTS_DIR / filename, index_col=0 if "matrix" in filename else None)
        assert not df.isna().any().any(), f"NaN values detected in {filename}"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            assert not np.isinf(df[c]).any(), f"Inf values detected in {filename} column {c}"
