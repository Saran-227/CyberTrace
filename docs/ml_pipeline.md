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

## 3. Candidate Features
- Numerical: `amount`, `complaint_latitude`, `complaint_longitude`, `hour`, `day_of_week`, `is_weekend`, `is_night`
- Categorical: `bank`, `transaction_type`, `fraud_type`, `state`, `district`, `city`, `amount_category`

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

