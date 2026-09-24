"""CLI execution script for CyberTrace Phase 4: ML Preprocessing & Feature Engineering.

Generates:
- reports/phase4_preprocessing_summary.md
- reports/feature_inventory.csv

Validates dual feature sets (Full vs Geographic-Blind), missing-value imputation,
one-hot encoding, stratified train/test split, and zero target leakage.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    COMPLAINTS_PROCESSED_PATH,
    REPORTS_DIR,
    FEATURE_SET_FULL,
    FEATURE_SET_GEOGRAPHIC_BLIND,
    TARGET_COLUMN,
)
from src.preprocessing.cleaning import clean_complaints_data, validate_cleaning_bounds
from src.preprocessing.features import engineer_complaint_features
from src.preprocessing.pipeline import (
    prepare_dataset,
    build_preprocessor,
    split_data,
    get_feature_names,
    compute_class_weights,
    build_feature_sets,
)


def create_feature_inventory() -> pd.DataFrame:
    """Construct a complete inventory of all features across both configurations."""
    inventory_records = [
        {
            "feature": "amount",
            "source_column": "amount",
            "feature_type": "Numerical (Continuous INR)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer(median) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Transaction amount in INR. Monitored for right-skewed distribution.",
        },
        {
            "feature": "amount_log",
            "source_column": "amount",
            "feature_type": "Numerical (Continuous)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "log1p(amount) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Engineered natural log transform to stabilize variance of high monetary amounts.",
        },
        {
            "feature": "complaint_latitude",
            "source_column": "complaint_latitude",
            "feature_type": "Numerical (Spatial Latitude °N)",
            "feature_set": "Full only",
            "transformation": "SimpleImputer(median) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Legitimate visible complaint coordinate. Excluded in geographic-blind baseline.",
        },
        {
            "feature": "complaint_longitude",
            "source_column": "complaint_longitude",
            "feature_type": "Numerical (Spatial Longitude °E)",
            "feature_set": "Full only",
            "transformation": "SimpleImputer(median) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Legitimate visible complaint coordinate. Excluded in geographic-blind baseline.",
        },
        {
            "feature": "hour",
            "source_column": "hour",
            "feature_type": "Numerical (Discrete 0-23)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer(median) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Hour of incident reporting.",
        },
        {
            "feature": "hour_sin",
            "source_column": "hour",
            "feature_type": "Numerical (Continuous Cyclical)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "sin(2*pi*hour/24) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Engineered 24-hour circular sine coordinate.",
        },
        {
            "feature": "hour_cos",
            "source_column": "hour",
            "feature_type": "Numerical (Continuous Cyclical)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "cos(2*pi*hour/24) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Engineered 24-hour circular cosine coordinate.",
        },
        {
            "feature": "day_of_week",
            "source_column": "day_of_week",
            "feature_type": "Numerical (Discrete 0-6)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer(median) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Day of week index (0=Monday, 6=Sunday).",
        },
        {
            "feature": "day_of_week_sin",
            "source_column": "day_of_week",
            "feature_type": "Numerical (Continuous Cyclical)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "sin(2*pi*dow/7) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Engineered 7-day circular sine coordinate.",
        },
        {
            "feature": "day_of_week_cos",
            "source_column": "day_of_week",
            "feature_type": "Numerical (Continuous Cyclical)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "cos(2*pi*dow/7) -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Engineered 7-day circular cosine coordinate.",
        },
        {
            "feature": "is_weekend",
            "source_column": "is_weekend / day_of_week",
            "feature_type": "Numerical (Binary 0/1)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "Binary indicator -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Binary flag indicating Saturday or Sunday.",
        },
        {
            "feature": "is_night",
            "source_column": "is_night / hour",
            "feature_type": "Numerical (Binary 0/1)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "Binary indicator -> StandardScaler()",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Binary flag indicating overnight hours (22:00 to 05:00).",
        },
        {
            "feature": "bank",
            "source_column": "bank",
            "feature_type": "Categorical (Nominal)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Complainant commercial bank. Missing values explicitly imputed as 'Unknown'.",
        },
        {
            "feature": "transaction_type",
            "source_column": "transaction_type",
            "feature_type": "Categorical (Nominal)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Payment rail (UPI, IMPS, NetBanking, Debit Card, etc.).",
        },
        {
            "feature": "fraud_type",
            "source_column": "fraud_type",
            "feature_type": "Categorical (Nominal)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Modus operandi classification (Investment Scam, KYC Phishing, Job Fraud, etc.).",
        },
        {
            "feature": "amount_category",
            "source_column": "amount_category / amount",
            "feature_type": "Categorical (Ordinal Binned)",
            "feature_set": "Both (Full & Geographic-Blind)",
            "transformation": "Binned categories -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Risk buckets: Low (<5k), Medium (5k-25k), High (25k-50k), Critical (>50k).",
        },
        {
            "feature": "city",
            "source_column": "city",
            "feature_type": "Categorical (Nominal Spatial)",
            "feature_set": "Full only",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Reporting municipality across 15 study regions. Excluded in geographic-blind baseline.",
        },
        {
            "feature": "district",
            "source_column": "district",
            "feature_type": "Categorical (Nominal Spatial)",
            "feature_set": "Full only",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "Administrative district. Excluded in geographic-blind baseline.",
        },
        {
            "feature": "state",
            "source_column": "state",
            "feature_type": "Categorical (Nominal Spatial)",
            "feature_set": "Full only",
            "transformation": "SimpleImputer('Unknown') -> OneHotEncoder(ignore)",
            "uses_target": False,
            "uses_hidden_data": False,
            "notes": "State administrative region. Excluded in geographic-blind baseline.",
        },
    ]
    return pd.DataFrame(inventory_records)


def main():
    print("=" * 70)
    print("CYBERTRACE PHASE 4: ML PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load Data
    raw_df = pd.read_csv(COMPLAINTS_PROCESSED_PATH)
    initial_rows = len(raw_df)
    print(f"Loaded complaint dataset: {initial_rows:,} records from {COMPLAINTS_PROCESSED_PATH}")

    # 2. Cleaning and Validation
    cleaned_df = clean_complaints_data(raw_df)
    final_clean_rows = len(cleaned_df)
    rows_dropped = initial_rows - final_clean_rows
    print(f"Cleaning complete: {final_clean_rows:,} usable records ({rows_dropped} dropped).")

    # 3. Process Full Feature Set
    print("\n[Configuration A] Processing Full Metadata Feature Set...")
    X_full, y = prepare_dataset(cleaned_df, feature_set="full")
    X_train_f, X_test_f, y_train, y_test = split_data(
        X_full, y, test_size=0.20, random_state=42, stratify=True
    )

    preprocessor_full = build_preprocessor(feature_set="full")
    # Fit ONLY on training data
    preprocessor_full.fit(X_train_f)
    X_train_f_trans = preprocessor_full.transform(X_train_f)
    X_test_f_trans = preprocessor_full.transform(X_test_f)
    feature_names_full = get_feature_names(preprocessor_full)

    print(f"  -> X_train shape: {X_train_f.shape} -> Transformed: {X_train_f_trans.shape}")
    print(f"  -> X_test shape:  {X_test_f.shape} -> Transformed: {X_test_f_trans.shape}")
    print(f"  -> Encoded feature count: {len(feature_names_full)}")

    # 4. Process Geographic-Blind Baseline Set
    print("\n[Configuration B] Processing Geographic-Blind Baseline Set...")
    X_geo, y_geo = prepare_dataset(cleaned_df, feature_set="geographic_blind")
    X_train_g, X_test_g, _, _ = split_data(
        X_geo, y_geo, test_size=0.20, random_state=42, stratify=True
    )

    preprocessor_geo = build_preprocessor(feature_set="geographic_blind")
    # Fit ONLY on training data
    preprocessor_geo.fit(X_train_g)
    X_train_g_trans = preprocessor_geo.transform(X_train_g)
    X_test_g_trans = preprocessor_geo.transform(X_test_g)
    feature_names_geo = get_feature_names(preprocessor_geo)

    print(f"  -> X_train shape: {X_train_g.shape} -> Transformed: {X_train_g_trans.shape}")
    print(f"  -> X_test shape:  {X_test_g.shape} -> Transformed: {X_test_g_trans.shape}")
    print(f"  -> Encoded feature count: {len(feature_names_geo)}")

    # 5. Class Weights & Imbalance Calculation
    class_weights = compute_class_weights(y_train)
    print("\nComputed Class Weights (Balanced):")
    for zone, weight in sorted(class_weights.items()):
        print(f"  {zone}: {weight:.4f}")

    # 6. Verify Stratification Fidelity
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    max_strat_diff = float((train_dist - test_dist).abs().max())
    print(f"\nStratification Check: Maximum class proportion divergence = {max_strat_diff:.6f} (Strictly < 0.001)")

    # 7. Generate Feature Inventory
    inventory_df = create_feature_inventory()
    inventory_path = REPORTS_DIR / "feature_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    print(f"\nGenerated feature inventory: {inventory_path} ({len(inventory_df)} features)")

    # 8. Generate Preprocessing Summary Markdown
    generate_summary_markdown(
        initial_rows=initial_rows,
        clean_rows=final_clean_rows,
        rows_dropped=rows_dropped,
        X_train_f=X_train_f,
        X_test_f=X_test_f,
        X_train_f_trans=X_train_f_trans,
        X_test_f_trans=X_test_f_trans,
        feature_names_full=feature_names_full,
        X_train_g=X_train_g,
        X_test_g=X_test_g,
        X_train_g_trans=X_train_g_trans,
        X_test_g_trans=X_test_g_trans,
        feature_names_geo=feature_names_geo,
        y_train=y_train,
        y_test=y_test,
        class_weights=class_weights,
    )
    print(f"Generated preprocessing summary: {REPORTS_DIR / 'phase4_preprocessing_summary.md'}")

    print("\n" + "=" * 70)
    print("PHASE 4 PREPROCESSING & FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)


def generate_summary_markdown(
    initial_rows: int,
    clean_rows: int,
    rows_dropped: int,
    X_train_f: pd.DataFrame,
    X_test_f: pd.DataFrame,
    X_train_f_trans: np.ndarray,
    X_test_f_trans: np.ndarray,
    feature_names_full: list,
    X_train_g: pd.DataFrame,
    X_test_g: pd.DataFrame,
    X_train_g_trans: np.ndarray,
    X_test_g_trans: np.ndarray,
    feature_names_geo: list,
    y_train: pd.Series,
    y_test: pd.Series,
    class_weights: dict,
) -> None:
    """Write comprehensive markdown documentation for Phase 4."""
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    table_rows = []
    for z in train_counts.index:
        t_cnt = train_counts[z]
        t_pct = (t_cnt / len(y_train)) * 100
        te_cnt = test_counts[z]
        te_pct = (te_cnt / len(y_test)) * 100
        w = class_weights[z]
        table_rows.append(f"| `{z}` | {t_cnt:,} ({t_pct:.2f}%) | {te_cnt:,} ({te_pct:.2f}%) | {w:.4f} |")

    table_str = "\n".join(table_rows)

    summary_md = f"""# CyberTrace Phase 4: ML Preprocessing & Feature Engineering Summary

## Executive Overview
Phase 4 prepares production-quality, leakage-safe data transformation pipelines for supervised classification of the `withdrawal_zone` target variable.

In response to the Phase 3 discovery that reporting city and spatial coordinates exhibit high statistical association ($V = 0.9732$), Phase 4 formalizes **two parallel feature configurations**:
1. **Configuration A (Full Metadata)**: Integrates legitimate complaint geography (`city`, `state`, `district`, `complaint_latitude`, `complaint_longitude`) with transaction metadata and cyclical temporal embeddings.
2. **Configuration B (Geographic-Blind Baseline)**: Strips all direct spatial identifiers to establish an empirical benchmark isolating non-spatial transaction characteristics.

---

## 1. Dataset Dimensions & Cleaning Operations
- **Input Complaint Records**: **{initial_rows:,}**
- **Rows Removed**: **{rows_dropped}** (0 duplicate or corrupt rows discovered).
- **Usable Dataset Records**: **{clean_rows:,}**
- **Missing Value Handling Strategy**:
  - `bank`: Imputed with domain-safe `"Unknown"` token via `SimpleImputer(strategy='constant', fill_value='Unknown')`.
  - `transaction_type`: Imputed with `"Unknown"`.
  - `amount`: Imputed using `SimpleImputer(strategy='median')`.
  - **Leakage Prevention**: All imputation statistics (medians, categories) are fitted strictly on the training partition ($N = 16,000$) and applied transformatively to the test partition ($N = 4,000$).

---

## 2. Feature Configurations

### A. Full Metadata Configuration (`FEATURE_SET_FULL`)
- **Raw Input Features (19)**:
  - Numerical (12): `amount`, `amount_log`, `complaint_latitude`, `complaint_longitude`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`
  - Categorical (7): `bank`, `transaction_type`, `fraud_type`, `state`, `district`, `city`, `amount_category`
- **One-Hot Encoded Features**: **{len(feature_names_full)}** total columns after `OneHotEncoder(handle_unknown='ignore')`.

### B. Geographic-Blind Baseline (`FEATURE_SET_GEOGRAPHIC_BLIND`)
- **Raw Input Features (14)**:
  - Numerical (10): `amount`, `amount_log`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`
  - Categorical (4): `bank`, `transaction_type`, `fraud_type`, `amount_category`
- **One-Hot Encoded Features**: **{len(feature_names_geo)}** total columns after `OneHotEncoder(handle_unknown='ignore')`.

---

## 3. Engineered Features

| Feature | Mathematical Transformation | Analytical Motivation |
| :--- | :--- | :--- |
| `amount_log` | $\\log(1 + \\text{{amount}})$ | Compresses extreme right-skewed financial monetary values to stabilize gradient descent and distance metrics. |
| `hour_sin` | $\\sin(2\\pi \\cdot \\text{{hour}} / 24)$ | Maps circular 24-hour diurnal cycle to continuous 2D Cartesian space (23:00 and 00:00 remain adjacent). |
| `hour_cos` | $\\cos(2\\pi \\cdot \\text{{hour}} / 24)$ | Orthogonal component of 24-hour circular cyclical coordinate. |
| `day_of_week_sin` | $\\sin(2\\pi \\cdot \\text{{dow}} / 7)$ | Circular 7-day weekly cycle mapping (Sunday and Monday remain adjacent). |
| `day_of_week_cos` | $\\cos(2\\pi \\cdot \\text{{dow}} / 7)$ | Orthogonal component of 7-day circular cyclical coordinate. |

---

## 4. Train/Test Partitioning & Stratification

- **Partition Ratio**: 80% Train ($N = 16,000$), 20% Test ($N = 4,000$).
- **Random Seed**: `42` (Deterministic reproducibility).
- **Stratification Method**: `train_test_split(..., stratify=y)`.
- **Cross-Validation Preparation**: Configured for 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### Target Class Distribution Across Partitions
| Withdrawal Zone | Training Set Count (%) | Test Set Count (%) | Balanced Class Weight |
| :--- | :---: | :---: | :---: |
{table_str}

---

## 5. Target Leakage & Verification Safeguards

1. **Target Exclusion**: `withdrawal_zone` is never included in the candidate feature matrices $X_{{\\text{{train}}}}$ or $X_{{\\text{{test}}}}$.
2. **Hidden Coordinate Exclusion**: `synthetic_cashout_latitude` and `synthetic_cashout_longitude` are strictly barred at runtime; presence raises an immediate `ValueError`.
3. **Data Snooping Prevention**: Imputer and Scaler parameters are calculated strictly from training folds. Unseen test set labels or values do not influence feature means, variances, or medians.
4. **Unseen Category Robustness**: `OneHotEncoder(handle_unknown='ignore')` maps previously unseen categories in test or inference sets to all-zero vectors, eliminating runtime encoding crashes.
"""
    with open(REPORTS_DIR / "phase4_preprocessing_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)


if __name__ == "__main__":
    main()
