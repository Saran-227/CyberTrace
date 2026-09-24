# Machine Learning Pipeline Architecture

## 1. Problem Formulation
Predicting the likely geographic cash-out sector (`withdrawal_zone`) given complaint metadata:
$$\hat{y} = f(X)$$
Where $y \in \{\text{Zone\_01}, \dots, \text{Zone\_10}\}$ and $X$ is a feature vector containing non-leaking transaction attributes.

## 2. Leakage Prevention Rule
Target leakage occurs if features directly convey information about the withdrawal zone that would be unavailable at inference time:
- Hidden synthetic cash-out coordinates: **PROHIBITED**
- Zone centroid approximations: **PROHIBITED**
- `withdrawal_zone` in feature matrix: **STRICTLY PROHIBITED**

## 3. Dual Feature Configurations (Phase 4)
CyberTrace maintains two formalized feature configurations to evaluate the exact contribution of spatial vs. non-spatial predictors:

### Configuration A: Full Metadata (`FEATURE_SET_FULL`)
- **19 Raw Features** $\to$ **80 Encoded Features** via `ColumnTransformer`.
- **Numerical (12)**: `amount`, `amount_log`, `complaint_latitude`, `complaint_longitude`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`.
- **Categorical (7)**: `bank`, `transaction_type`, `fraud_type`, `city`, `state`, `district`, `amount_category`.
- **Transformations**: `SimpleImputer(median) -> StandardScaler()`, `SimpleImputer('Unknown') -> OneHotEncoder(ignore)`.

### Configuration B: Geographic-Blind Baseline (`FEATURE_SET_GEOGRAPHIC_BLIND`)
- **14 Raw Features** $\to$ **41 Encoded Features** via `ColumnTransformer`.
- Excludes all direct spatial identifiers (`city`, `state`, `district`, `complaint_latitude`, `complaint_longitude`).
- **Numerical (10)**: `amount`, `amount_log`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_of_week_sin`, `day_of_week_cos`, `is_weekend`, `is_night`.
- **Categorical (4)**: `bank`, `transaction_type`, `fraud_type`, `amount_category`.
- *Purpose*: Isolates transaction modus operandi to benchmark how much spatial predictors boost classification accuracy.

## 4. Evaluated Algorithms
The platform supports multi-model benchmarking without hardcoding a predetermined winner:
1. **Logistic Regression**: Linear multiclass baseline (Softmax).
2. **K-Nearest Neighbors (KNN)**: Spatial feature proximity baseline.
3. **Decision Tree Classifier**: Non-linear rule-based partitions.
4. **Random Forest Classifier**: Bagged ensemble of decorrelated decision trees.

## 5. Metrics & Validation
- **Stratified Holdout Split**: 80% train, 20% test (preserving exact class proportions).
- **Evaluation Criteria**:
  - Accuracy
  - Macro & Weighted Precision
  - Macro & Weighted Recall
  - Macro & Weighted F1-Score
  - Multiclass One-vs-Rest ROC-AUC
  - Confusion Matrix

---

## 6. Target Intelligence & Phase 4 Preprocessing Guidelines

Phase 3 established the empirical and spatial properties of the `withdrawal_zone` target:

> [!WARNING]
> **Academic Target Disclaimer**:
> `withdrawal_zone` is a synthetic target created for academic supervised-learning experimentation. It does not represent confirmed NCRP withdrawal locations.

1. **Stratification is Non-Negotiable**:
   - Due to the **23.43:1 class imbalance** (`Zone_07` has 5,436 instances while `Zone_09` has only 232), all training and evaluation routines in Phase 4 and Phase 5 must use stratified splits (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
2. **Class Weighting**:
   - Classifiers with cost-sensitive loss options (Logistic Regression, Decision Trees, Random Forests) should enable `class_weight='balanced'` to prevent minority classes from being suppressed.
3. **Primary Evaluation Metric**:
   - **Macro-F1** must serve as the primary model selection criterion in Phase 6 because standard micro-accuracy would be dominated by the 48.9% combined share of `Zone_07` and `Zone_08`.
4. **Feature Utility Established in Phase 3**:
   - `city`, `complaint_latitude`, and `complaint_longitude` have massive predictive power (Cramér's $V = 0.9732$).
   - Non-spatial attributes (`bank`, `amount`, `transaction_type`, `fraud_type`, `hour`) show near-zero leakage, functioning as secondary regularizers.

---

## 7. Phase 5 Model Training & Benchmark Results

Phase 5 evaluated all 14 primary model configurations using 5-fold cross-validation on the training set ($N = 16,000$) and evaluation on the untouched test partition ($N = 4,000$):

- **Full Metadata Performance**:
  - Accuracy: **90.8% - 92.5%**
  - Balanced Accuracy: **94.3% - 96.5%**
  - Macro F1: **0.951 - 0.966**
  - Multiclass ROC-AUC (OVR Macro): **0.986 - 0.996**
  - Generalization Gap ($|\text{CV Macro F1} - \text{Test Macro F1}|$): $< 0.006$ across all models (no overfitting detected).

- **Geographic-Blind Baseline Performance**:
  - Accuracy: **7.1% - 27.8%**
  - Balanced Accuracy: **8.8% - 10.8%**
  - Macro F1: **0.067 - 0.090**
  - Multiclass ROC-AUC: **~0.500** (random guessing).
  - *Analytical Conclusion*: Complaint geography carries the primary signal for predicting withdrawal jurisdiction. Non-spatial complaint characteristics carry negligible spatial information, proving zero synthetic leakage.

- **Class Weighting Impact**:
  - For Full Metadata models, `Zone_09` (minority, 232 cases) achieves $1.0$ F1-score even with `class_weight=None` due to clear spatial separation.
  - In Geographic-Blind models, balanced class weighting elevates minority recall at the expense of overall accuracy.

Detailed benchmarks, diagnostics, and per-zone metrics are documented in [docs/model_training.md](file:///docs/model_training.md).

