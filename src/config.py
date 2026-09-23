"""CyberTrace Centralized Configuration & System Constants.

Contains directory paths, feature schemas, model candidate configurations,
and external endpoints.
"""

from pathlib import Path
import os

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OSM_CACHE_DIR = EXTERNAL_DATA_DIR / "osm"

MODELS_DIR = BASE_DIR / "models"
LOCATION_CLASSIFIER_DIR = MODELS_DIR / "location_classifier"
ATM_RANKER_DIR = MODELS_DIR / "atm_ranker"

REPORTS_DIR = BASE_DIR / "reports"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"
REPORT_TEMPLATES_DIR = REPORTS_DIR / "templates"

APP_DIR = BASE_DIR / "app"

# Ensure essential runtime directories exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OSM_CACHE_DIR,
    LOCATION_CLASSIFIER_DIR,
    ATM_RANKER_DIR,
    GENERATED_REPORTS_DIR,
    REPORT_TEMPLATES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset file paths
COMPLAINTS_RAW_PATH = RAW_DATA_DIR / "cybercrime_complaints_raw.csv"
COMPLAINTS_PROCESSED_PATH = PROCESSED_DATA_DIR / "cybercrime_complaints.csv"

ATM_LOCATIONS_RAW_PATH = RAW_DATA_DIR / "atm_locations_raw.csv"
ATM_LOCATIONS_PROCESSED_PATH = PROCESSED_DATA_DIR / "atm_locations.csv"

ATM_ACTIVITY_RAW_PATH = RAW_DATA_DIR / "atm_activity_raw.csv"
ATM_ACTIVITY_PROCESSED_PATH = PROCESSED_DATA_DIR / "atm_activity.csv"

# Dataset Schemas
COMPLAINT_SCHEMA = [
    "complaint_id",
    "complaint_date",
    "complaint_time",
    "amount",
    "bank",
    "transaction_type",
    "fraud_type",
    "city",
    "state",
    "district",
    "complaint_latitude",
    "complaint_longitude",
    "hour",
    "day_of_week",
    "day_name",
    "is_weekend",
    "is_night",
    "amount_category",
    "withdrawal_zone",
]

TARGET_COLUMN = "withdrawal_zone"

# Valid Candidate Features for ML (Excluding target and direct leakage)
CANDIDATE_FEATURES = [
    "amount",
    "bank",
    "transaction_type",
    "fraud_type",
    "state",
    "district",
    "city",
    "complaint_latitude",
    "complaint_longitude",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "amount_category",
]

NUMERICAL_FEATURES = [
    "amount",
    "complaint_latitude",
    "complaint_longitude",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
]

CATEGORICAL_FEATURES = [
    "bank",
    "transaction_type",
    "fraud_type",
    "state",
    "district",
    "city",
    "amount_category",
]

ATM_LOCATIONS_SCHEMA = [
    "atm_id",
    "bank",
    "operator",
    "latitude",
    "longitude",
    "address",
    "city",
    "district",
    "state",
    "is_24x7",
    "source",
]

ATM_ACTIVITY_SCHEMA = [
    "atm_id",
    "date",
    "hour",
    "transaction_count",
    "fraud_withdrawal_count",
    "average_amount",
    "high_value_withdrawal_count",
]

# Supported Location Classifier Model Families
SUPPORTED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
]

# External Service Endpoints & Helplines
OVERPASS_API_URL = os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")
OVERPASS_ENDPOINTS = [
    os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter"),
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_TIMEOUT = int(os.getenv("OVERPASS_TIMEOUT_SECONDS", "25"))
OVERPASS_CACHE_TTL_HOURS = int(os.getenv("OVERPASS_CACHE_TTL_HOURS", "168"))

ATM_COVERAGE_REPORT_PATH = REPORTS_DIR / "atm_coverage_report.csv"
ATM_COVERAGE_SCHEMA = [
    "city",
    "state",
    "bbox",
    "raw_osm_objects",
    "atm_records",
    "known_bank_operator",
    "unknown_bank_operator",
    "status",
    "notes",
]

CYBERCRIME_PORTAL_URL = "https://www.cybercrime.gov.in/"
FINANCIAL_FRAUD_HELPLINE = "1930"

