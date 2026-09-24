# Data Preprocessing, Feature Engineering & Leakage Prevention Architecture

## 1. Executive Summary

Phase 4 establishes the production-grade preprocessing and feature engineering foundation for supervised classification of the `withdrawal_zone` target variable.

In accordance with findings from Phase 3 (where reporting city exhibited strong spatial association, Cramér's $V = 0.9732$), the architecture introduces **two parallel feature configurations** to allow Phase 5 models to quantify spatial dependence vs. non-spatial transaction signal:
1. **Configuration A (`FEATURE_SET_FULL`)**: Comprehensive metadata incorporating legitimate complaint geography alongside transaction attributes and cyclical temporal embeddings.
2. **Configuration B (`FEATURE_SET_GEOGRAPHIC_BLIND`)**: Rigorous baseline deliberately excluding direct spatial identifiers (`city`, `state`, `district`, `complaint_latitude`, `complaint_longitude`).

---

## 2. Preprocessing Architecture & Components

The preprocessing layer is encapsulated in `src/preprocessing/`:

```
src/preprocessing/
├── cleaning.py      # Deduplication, bounds validation, missingness defaults, normalization
├── features.py      # Trigonometric cyclical time embeddings, log amounts, risk categorizations
├── pipeline.py      # ColumnTransformer construction, dataset preparation, stratified splitting
└── __init__.py      # High-level API exports
```

### High-Level API Interface
- `prepare_dataset(df=None, feature_set='full')`: Cleans, engineers features, verifies zero leakage, and returns $(X, y)$.
- `build_preprocessor(feature_set='full')`: Constructs a Scikit-Learn `ColumnTransformer` tailored to the requested feature set.
- `split_data(X, y, test_size=0.20, random_state=42, stratify=True)`: Produces deterministic, stratified train/test partitions ($N_{\text{train}} = 16,000$, $N_{\text{test}} = 4,000$).
- `compute_class_weights(y)`: Computes balanced class weights for cost-sensitive learning in Phase 5.
- `get_feature_names(preprocessor)`: Extracts human-readable encoded column names from a fitted transformer.

---

## 3. Feature Configurations & Inventory

### A. Configuration A: Full Metadata (`FEATURE_SET_FULL`)
- **Raw Feature Count**: 19 columns
  - **Numerical (12)**: `amount`, `amount_log`, `complaint_latitude`, `complaint_longitude`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`
  - **Categorical (7)**: `bank`, `transaction_type`, `fraud_type`, `city`, `state`, `district`, `amount_category`
- **Encoded Dimension**: **80** columns after one-hot encoding.

### B. Configuration B: Geographic-Blind Baseline (`FEATURE_SET_GEOGRAPHIC_BLIND`)
- **Raw Feature Count**: 14 columns
  - **Numerical (10)**: `amount`, `amount_log`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`
  - **Categorical (4)**: `bank`, `transaction_type`, `fraud_type`, `amount_category`
- **Encoded Dimension**: **41** columns after one-hot encoding.

---

## 4. Missing-Value Handling Strategy

To prevent data snooping and maintain legal/academic research fidelity:
1. **Categorical Features** (`bank`, `transaction_type`, `district`, `city`, etc.):
   - Imputed using `SimpleImputer(strategy='constant', fill_value='Unknown')`.
   - Preserves missingness as an explicit informative category rather than silently discarding records.
2. **Numerical Features** (`amount`, `complaint_latitude`, `complaint_longitude`, etc.):
   - Imputed using `SimpleImputer(strategy='median')`.
   - **Leakage Prevention**: Imputation medians are learned strictly on the training partition ($X_{\text{train}}$) and applied without modification to the test partition ($X_{\text{test}}$).

---

## 5. Categorical Encoding & Numerical Scaling

- **Categorical Encoding**:
  - Encoded with Scikit-Learn's `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Guarantees that any novel or unseen bank/city in subsequent inference will be encoded cleanly as an all-zero vector, eliminating runtime crashes.
- **Numerical Scaling**:
  - Scaled with `StandardScaler()` (zero mean, unit variance).
  - Normalizes distance metrics for distance-sensitive algorithms (KNN, Logistic Regression) while preserving compatibility with tree-based models (Decision Trees, Random Forests).

---

## 6. Feature Engineering

| Feature | Transformation Formula | Domain Rationale |
| :--- | :--- | :--- |
| **`amount_log`** | $\log(1 + \text{amount})$ | Dampens the extreme right-skewness of financial fraud losses (up to ₹100,000), stabilizing linear loss surfaces. |
| **`hour_sin`**, **`hour_cos`** | $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$, $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ | Preserves the cyclical continuity of diurnal hours (ensuring 23:00 and 00:00 remain adjacent in Euclidean space). |
| **`day_of_week_sin`**, **`day_of_week_cos`** | $\sin\left(\frac{2\pi \cdot \text{dow}}{7}\right)$, $\cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$ | Preserves circular 7-day weekly recurrence (Sunday and Monday remain adjacent). |
| **`amount_category`** | Binned risk tiers | Categorizes transaction magnitude into intuitive risk bands: Low (<5k), Medium (5k-25k), High (25k-50k), Critical (>50k). |

---

## 7. Strict Target Leakage Prevention

1. **Target Exclusion**: `withdrawal_zone` is strictly separated from $X$ and returned exclusively as target $y$.
2. **Forbidden Coordinates Enforcement**: Hidden coordinates (`synthetic_cashout_latitude`, `synthetic_cashout_longitude`) are blacklisted; their presence in any feature configuration raises an immediate `ValueError`.
3. **Training Isolation**: Preprocessors are never fitted on full dataset before splitting. `fit()` is executed strictly on the training split ($N = 16,000$).
4. **Target-Derived Features Prohibited**: No spatial distance to withdrawal zone centroids or synthetic cash-out points is computed or provided to the models.

---

## 8. Class Imbalance Mitigation Strategy

Phase 3 established that `withdrawal_zone` exhibits a **23.43:1** imbalance ratio (`Zone_07` at 27.18% vs. `Zone_09` at 1.16%).

### Implementation Rules:
1. **Preserve Natural Distribution**: No synthetic oversampling (SMOTE) or random undersampling is applied during preprocessing to preserve authentic empirical density.
2. **Stratified Splitting**: `split_data` employs `stratify=y`, ensuring training ($N=16,000$) and test ($N=4,000$) partitions have identical class proportions (max divergence $< 0.0002$).
3. **Balanced Class Weights**: `compute_class_weights(y_train)` computes exact weights:
   - `Zone_07` (Majority): `0.3679`
   - `Zone_09` (Minority): `8.6022`
   - Phase 5 model training will supply these weights to `class_weight='balanced'`.
