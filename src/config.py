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
CONFUSION_MATRICES_DIR = REPORTS_DIR / "figures" / "confusion_matrices"
PER_ZONE_METRICS_DIR = REPORTS_DIR / "per_zone_metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
PROBABILITY_DIAGNOSTICS_DIR = REPORTS_DIR / "probability_diagnostics"
FEATURE_IMPORTANCE_DIR = REPORTS_DIR / "feature_importance"

APP_DIR = BASE_DIR / "app"

# Canonical complaint dataset SHA-256 fingerprint
CANONICAL_COMPLAINTS_SHA256 = "2be4188500ff2be70522220753ba8bddd0af5466c777bd8e6ef6528d916cb4ff"

# Ensure essential runtime directories exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OSM_CACHE_DIR,
    LOCATION_CLASSIFIER_DIR,
    ATM_RANKER_DIR,
    GENERATED_REPORTS_DIR,
    REPORT_TEMPLATES_DIR,
    CONFUSION_MATRICES_DIR,
    PER_ZONE_METRICS_DIR,
    PREDICTIONS_DIR,
    PROBABILITY_DIAGNOSTICS_DIR,
    FEATURE_IMPORTANCE_DIR,
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
    "amount_log",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

NUMERICAL_FEATURES = [
    "amount",
    "amount_log",
    "complaint_latitude",
    "complaint_longitude",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "day_of_week_sin",
    "day_of_week_cos",
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

# Phase 4 Feature Configurations
FEATURE_SET_FULL = {
    "name": "full",
    "description": "Full complaint metadata including legitimate complaint geography and engineered temporal/monetary features",
    "numerical": [
        "amount",
        "amount_log",
        "complaint_latitude",
        "complaint_longitude",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "is_night",
    ],
    "categorical": [
        "bank",
        "transaction_type",
        "fraud_type",
        "state",
        "district",
        "city",
        "amount_category",
    ],
}

FEATURE_SET_GEOGRAPHIC_BLIND = {
    "name": "geographic_blind",
    "description": "Geographic-blind baseline excluding direct geographic identifiers (city, state, district, lat, lon)",
    "numerical": [
        "amount",
        "amount_log",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "is_night",
    ],
    "categorical": [
        "bank",
        "transaction_type",
        "fraud_type",
        "amount_category",
    ],
}

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
    "day_of_week",
    "is_weekend",
    "is_night",
    "transaction_count",
    "cash_withdrawal_count",
    "estimated_cash_volume",
    "activity_score",
    "high_value_withdrawal_count",
    "average_amount",
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

# Phase 5 Location Classifier Experiment Configurations (14 primary experiments)
EXPERIMENT_CONFIGS = [
    {
        "experiment_id": "logistic_full_none",
        "model_name": "Logistic Regression",
        "algorithm": "LogisticRegression",
        "feature_set": "full",
        "class_weight": None,
        "hyperparameters": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42},
    },
    {
        "experiment_id": "logistic_full_balanced",
        "model_name": "Logistic Regression",
        "algorithm": "LogisticRegression",
        "feature_set": "full",
        "class_weight": "balanced",
        "hyperparameters": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42, "class_weight": "balanced"},
    },
    {
        "experiment_id": "logistic_blind_none",
        "model_name": "Logistic Regression",
        "algorithm": "LogisticRegression",
        "feature_set": "geographic_blind",
        "class_weight": None,
        "hyperparameters": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42},
    },
    {
        "experiment_id": "logistic_blind_balanced",
        "model_name": "Logistic Regression",
        "algorithm": "LogisticRegression",
        "feature_set": "geographic_blind",
        "class_weight": "balanced",
        "hyperparameters": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42, "class_weight": "balanced"},
    },
    {
        "experiment_id": "knn_full",
        "model_name": "K-Nearest Neighbors",
        "algorithm": "KNeighborsClassifier",
        "feature_set": "full",
        "class_weight": "NOT_SUPPORTED",
        "hyperparameters": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
    },
    {
        "experiment_id": "knn_blind",
        "model_name": "K-Nearest Neighbors",
        "algorithm": "KNeighborsClassifier",
        "feature_set": "geographic_blind",
        "class_weight": "NOT_SUPPORTED",
        "hyperparameters": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
    },
    {
        "experiment_id": "decision_tree_full_none",
        "model_name": "Decision Tree",
        "algorithm": "DecisionTreeClassifier",
        "feature_set": "full",
        "class_weight": None,
        "hyperparameters": {"criterion": "gini", "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "random_state": 42},
    },
    {
        "experiment_id": "decision_tree_full_balanced",
        "model_name": "Decision Tree",
        "algorithm": "DecisionTreeClassifier",
        "feature_set": "full",
        "class_weight": "balanced",
        "hyperparameters": {"criterion": "gini", "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "random_state": 42, "class_weight": "balanced"},
    },
    {
        "experiment_id": "decision_tree_blind_none",
        "model_name": "Decision Tree",
        "algorithm": "DecisionTreeClassifier",
        "feature_set": "geographic_blind",
        "class_weight": None,
        "hyperparameters": {"criterion": "gini", "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "random_state": 42},
    },
    {
        "experiment_id": "decision_tree_blind_balanced",
        "model_name": "Decision Tree",
        "algorithm": "DecisionTreeClassifier",
        "feature_set": "geographic_blind",
        "class_weight": "balanced",
        "hyperparameters": {"criterion": "gini", "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "random_state": 42, "class_weight": "balanced"},
    },
    {
        "experiment_id": "random_forest_full_none",
        "model_name": "Random Forest",
        "algorithm": "RandomForestClassifier",
        "feature_set": "full",
        "class_weight": None,
        "hyperparameters": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1},
    },
    {
        "experiment_id": "random_forest_full_balanced",
        "model_name": "Random Forest",
        "algorithm": "RandomForestClassifier",
        "feature_set": "full",
        "class_weight": "balanced",
        "hyperparameters": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
    },
    {
        "experiment_id": "random_forest_blind_none",
        "model_name": "Random Forest",
        "algorithm": "RandomForestClassifier",
        "feature_set": "geographic_blind",
        "class_weight": None,
        "hyperparameters": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1},
    },
    {
        "experiment_id": "random_forest_blind_balanced",
        "model_name": "Random Forest",
        "algorithm": "RandomForestClassifier",
        "feature_set": "geographic_blind",
        "class_weight": "balanced",
        "hyperparameters": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
    },
]

