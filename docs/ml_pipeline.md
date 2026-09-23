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
- **Stratified Holdout Split**: 80% train, 20% test.
- **Evaluation Criteria**:
  - Accuracy
  - Macro & Weighted Precision
  - Macro & Weighted Recall
  - Macro & Weighted F1-Score
  - Multiclass One-vs-Rest ROC-AUC
  - Confusion Matrix
