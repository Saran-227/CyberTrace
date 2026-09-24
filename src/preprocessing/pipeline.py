"""Scikit-Learn Preprocessing Pipeline for CyberTrace Complaint Data.

Provides dual feature configurations (Full Metadata vs. Geographic-Blind Baseline),
leakage-safe ColumnTransformer pipelines, stratified splitting, and feature name extraction.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    CANDIDATE_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_SET_FULL,
    FEATURE_SET_GEOGRAPHIC_BLIND,
    TARGET_COLUMN,
)
from src.utils.validation import check_target_leakage
from src.utils.logging import get_logger
from src.preprocessing.cleaning import clean_complaints_data
from src.preprocessing.features import engineer_complaint_features

logger = get_logger("PreprocessingPipeline")

FORBIDDEN_COLUMNS = [
    TARGET_COLUMN,
    "synthetic_cashout_latitude",
    "synthetic_cashout_longitude",
    "cashout_latitude",
    "cashout_longitude",
]


def build_feature_sets() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of configured feature sets."""
    return {
        "full": FEATURE_SET_FULL,
        "geographic_blind": FEATURE_SET_GEOGRAPHIC_BLIND,
    }


def build_preprocessor(feature_set: str = "full") -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer for the specified feature configuration.

    Parameters:
        feature_set: 'full' (legitimate metadata + geography) or 'geographic_blind' (no spatial columns).

    Returns:
        Configured ColumnTransformer with imputation and encoding.
    """
    feature_sets = build_feature_sets()
    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Supported: {list(feature_sets.keys())}")

    config = feature_sets[feature_set]
    num_cols = config["numerical"]
    cat_cols = config["categorical"]

    # Verify zero target leakage
    for col in num_cols + cat_cols:
        if col in FORBIDDEN_COLUMNS:
            raise ValueError(f"CRITICAL TARGET LEAKAGE DETECTED: Forbidden column '{col}' cannot be an input feature.")

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        f"Built ColumnTransformer for feature_set='{feature_set}': "
        f"{len(num_cols)} numerical, {len(cat_cols)} categorical features."
    )
    return preprocessor


def build_preprocessing_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Backward-compatible preprocessor builder."""
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    all_features = numerical_cols + categorical_cols
    for f in all_features:
        if f in FORBIDDEN_COLUMNS:
            raise ValueError(
                f"CRITICAL TARGET LEAKAGE DETECTED: '{TARGET_COLUMN}' cannot be an input feature."
            )

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_dataset(
    df: Optional[pd.DataFrame] = None,
    feature_set: str = "full",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load, clean, engineer features, and extract X and y for the specified feature configuration.

    Guarantees:
    - Target column is never in X.
    - Forbidden hidden cash-out coordinates cannot enter X.
    - Missing values and cyclical features are properly processed.
    """
    if df is None:
        if not COMPLAINTS_PROCESSED_PATH.exists():
            raise FileNotFoundError(f"Complaints dataset missing at {COMPLAINTS_PROCESSED_PATH}")
        df = pd.read_csv(COMPLAINTS_PROCESSED_PATH)

    # 1. Clean data (deduplication & bounds check)
    df_cleaned = clean_complaints_data(df)

    # 2. Engineer features (amount_log, hour_sin/cos, day_of_week_sin/cos)
    df_engineered = engineer_complaint_features(df_cleaned)

    # 3. Extract target
    if TARGET_COLUMN not in df_engineered.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")
    y = df_engineered[TARGET_COLUMN].copy()

    # 4. Select features according to configuration
    feature_sets = build_feature_sets()
    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Supported: {list(feature_sets.keys())}")

    config = feature_sets[feature_set]
    configured_cols = config["numerical"] + config["categorical"]

    # Verify no forbidden columns
    for f in df_engineered.columns:
        if f in FORBIDDEN_COLUMNS and f != TARGET_COLUMN:
            raise ValueError(f"Target leakage violation: '{f}' is forbidden in X.")

    # Select all configured columns that are present in the dataframe
    selected_cols = [c for c in configured_cols if c in df_engineered.columns]
    if not selected_cols:
        selected_cols = [c for c in df_engineered.columns if c not in FORBIDDEN_COLUMNS and c != TARGET_COLUMN]

    X = df_engineered[selected_cols].copy()
    logger.info(f"Prepared dataset '{feature_set}': X shape {X.shape}, y length {len(y)}.")
    return X, y


def prepare_xy(
    df: pd.DataFrame,
    feature_set: str = "full",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Backward-compatible helper to extract candidate features X and target y."""
    return prepare_dataset(df=df, feature_set=feature_set)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform deterministic, stratified train/test split.

    The test partition is isolated and must not be used for fitting preprocessors or models.
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    logger.info(
        f"Split data with test_size={test_size}, random_state={random_state}, stratified={stratify}. "
        f"Train: {len(X_train):,}, Test: {len(X_test):,}."
    )
    return X_train, X_test, y_train, y_test


def get_stratified_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Return configured StratifiedKFold cross-validator for Phase 5 benchmarking."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compute_class_weights(y: pd.Series) -> Dict[str, float]:
    """Compute balanced class weights to address target imbalance in Phase 5 classifiers."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {cls: round(float(w), 4) for cls, w in zip(classes, weights)}


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for manual inspection if not fitted
        names = []
        for name, transformer, cols in preprocessor.transformers:
            if name != "remainder":
                for col in cols:
                    names.append(f"{name}__{col}")
        return names
