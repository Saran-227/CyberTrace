# CyberTrace Phase 4: ML Preprocessing & Feature Engineering Summary

## Executive Overview
Phase 4 prepares production-quality, leakage-safe data transformation pipelines for supervised classification of the `withdrawal_zone` target variable.

In response to the Phase 3 discovery that reporting city and spatial coordinates exhibit high statistical association ($V = 0.9732$), Phase 4 formalizes **two parallel feature configurations**:
1. **Configuration A (Full Metadata)**: Integrates legitimate complaint geography (`city`, `state`, `district`, `complaint_latitude`, `complaint_longitude`) with transaction metadata and cyclical temporal embeddings.
2. **Configuration B (Geographic-Blind Baseline)**: Strips all direct spatial identifiers to establish an empirical benchmark isolating non-spatial transaction characteristics.

---

## 1. Dataset Dimensions & Cleaning Operations
- **Input Complaint Records**: **20,000**
- **Rows Removed**: **0** (0 duplicate or corrupt rows discovered).
- **Usable Dataset Records**: **20,000**
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
- **One-Hot Encoded Features**: **80** total columns after `OneHotEncoder(handle_unknown='ignore')`.

### B. Geographic-Blind Baseline (`FEATURE_SET_GEOGRAPHIC_BLIND`)
- **Raw Input Features (14)**:
  - Numerical (10): `amount`, `amount_log`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`
  - Categorical (4): `bank`, `transaction_type`, `fraud_type`, `amount_category`
- **One-Hot Encoded Features**: **41** total columns after `OneHotEncoder(handle_unknown='ignore')`.

---

## 3. Engineered Features

| Feature | Mathematical Transformation | Analytical Motivation |
| :--- | :--- | :--- |
| `amount_log` | $\log(1 + \text{amount})$ | Compresses extreme right-skewed financial monetary values to stabilize gradient descent and distance metrics. |
| `hour_sin` | $\sin(2\pi \cdot \text{hour} / 24)$ | Maps circular 24-hour diurnal cycle to continuous 2D Cartesian space (23:00 and 00:00 remain adjacent). |
| `hour_cos` | $\cos(2\pi \cdot \text{hour} / 24)$ | Orthogonal component of 24-hour circular cyclical coordinate. |
| `day_of_week_sin` | $\sin(2\pi \cdot \text{dow} / 7)$ | Circular 7-day weekly cycle mapping (Sunday and Monday remain adjacent). |
| `day_of_week_cos` | $\cos(2\pi \cdot \text{dow} / 7)$ | Orthogonal component of 7-day circular cyclical coordinate. |

---

## 4. Train/Test Partitioning & Stratification

- **Partition Ratio**: 80% Train ($N = 16,000$), 20% Test ($N = 4,000$).
- **Random Seed**: `42` (Deterministic reproducibility).
- **Stratification Method**: `train_test_split(..., stratify=y)`.
- **Cross-Validation Preparation**: Configured for 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### Target Class Distribution Across Partitions
| Withdrawal Zone | Training Set Count (%) | Test Set Count (%) | Balanced Class Weight |
| :--- | :---: | :---: | :---: |
| `Zone_01` | 993 (6.21%) | 248 (6.20%) | 1.6113 |
| `Zone_02` | 1,262 (7.89%) | 315 (7.88%) | 1.2678 |
| `Zone_03` | 1,454 (9.09%) | 364 (9.10%) | 1.1004 |
| `Zone_04` | 2,073 (12.96%) | 519 (12.97%) | 0.7718 |
| `Zone_05` | 611 (3.82%) | 153 (3.82%) | 2.6187 |
| `Zone_06` | 673 (4.21%) | 168 (4.20%) | 2.3774 |
| `Zone_07` | 4,349 (27.18%) | 1,087 (27.18%) | 0.3679 |
| `Zone_08` | 3,471 (21.69%) | 868 (21.70%) | 0.4610 |
| `Zone_09` | 186 (1.16%) | 46 (1.15%) | 8.6022 |
| `Zone_10` | 928 (5.80%) | 232 (5.80%) | 1.7241 |

---

## 5. Target Leakage & Verification Safeguards

1. **Target Exclusion**: `withdrawal_zone` is never included in the candidate feature matrices $X_{\text{train}}$ or $X_{\text{test}}$.
2. **Hidden Coordinate Exclusion**: `synthetic_cashout_latitude` and `synthetic_cashout_longitude` are strictly barred at runtime; presence raises an immediate `ValueError`.
3. **Data Snooping Prevention**: Imputer and Scaler parameters are calculated strictly from training folds. Unseen test set labels or values do not influence feature means, variances, or medians.
4. **Unseen Category Robustness**: `OneHotEncoder(handle_unknown='ignore')` maps previously unseen categories in test or inference sets to all-zero vectors, eliminating runtime encoding crashes.
