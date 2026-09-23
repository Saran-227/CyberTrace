"""Scikit-Learn Preprocessing Pipeline for CyberTrace Complaint Data.

Strictly enforces target leakage prevention by validating that 'withdrawal_zone'
is never included in the input feature set.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    CANDIDATE_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.utils.validation import check_target_leakage
from src.utils.logging import get_logger

logger = get_logger("PreprocessingPipeline")

def build_preprocessing_pipeline(
    numerical_cols: List[str] = NUMERICAL_FEATURES,
    categorical_cols: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.

    Validates against target leakage.
    """
    all_features = numerical_cols + categorical_cols
    if check_target_leakage(all_features):
        raise ValueError(
            f"CRITICAL TARGET LEAKAGE DETECTED: '{TARGET_COLUMN}' cannot be an input feature."
        )

    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info(
        f"Built ColumnTransformer with {len(numerical_cols)} numerical "
        f"and {len(categorical_cols)} categorical candidate features."
    )
    return preprocessor

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract candidate features X and target y, verifying zero target leakage."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    # Select only valid candidate features that exist in the dataframe
    valid_features = [col for col in CANDIDATE_FEATURES if col in df.columns]

    # Leakage assertion
    if TARGET_COLUMN in valid_features:
        valid_features.remove(TARGET_COLUMN)

    X = df[valid_features].copy()
    y = df[TARGET_COLUMN].copy()

    logger.info(f"Prepared X with shape {X.shape} and target y with {y.nunique()} unique zones.")
    return X, y
