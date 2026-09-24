"""Data preprocessing and feature engineering package for CyberTrace."""

from .cleaning import (
    clean_complaints_data,
    remove_duplicates,
    impute_missing_values,
    normalize_categorical_values,
    validate_cleaning_bounds,
)
from .features import (
    engineer_complaint_features,
    categorize_amount,
    add_engineered_features,
)
from .pipeline import (
    build_preprocessing_pipeline,
    build_preprocessor,
    build_feature_sets,
    prepare_dataset,
    prepare_xy,
    split_data,
    get_feature_names,
    get_stratified_cv,
    compute_class_weights,
)

__all__ = [
    "clean_complaints_data",
    "remove_duplicates",
    "impute_missing_values",
    "normalize_categorical_values",
    "validate_cleaning_bounds",
    "engineer_complaint_features",
    "categorize_amount",
    "add_engineered_features",
    "build_preprocessing_pipeline",
    "build_preprocessor",
    "build_feature_sets",
    "prepare_dataset",
    "prepare_xy",
    "split_data",
    "get_feature_names",
    "get_stratified_cv",
    "compute_class_weights",
]
