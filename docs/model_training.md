# CyberTrace Phase 5: Location Classification Model Training & Experimentation

## Executive Summary

Phase 5 establishes a rigorous, reproducible, leakage-free benchmark of supervised classification models for predicting the cybercrime cash-out zone (`withdrawal_zone`). Across **14 primary experiments**, four distinct model families were evaluated under both the **Full Metadata** and **Geographic-Blind Baseline** configurations, testing both unweighted (`class_weight=None`) and balanced (`class_weight="balanced"`) loss functions.

All models were evaluated using 5-fold `StratifiedKFold` cross-validation on the training partition ($N = 16,000$) followed by evaluation on an untouched, stratified test partition ($N = 4,000$).

---

## 1. Experimental Design & Matrix

The 14 primary experiments are codified in the experiment registry:

| Experiment ID | Algorithm | Feature Set | Encoded Features | Class Weight | Primary Hyperparameters |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `logistic_full_none` | `LogisticRegression` | Full | 80 | None | `max_iter=1000`, `solver='lbfgs'`, `random_state=42` |
| `logistic_full_balanced` | `LogisticRegression` | Full | 80 | Balanced | `max_iter=1000`, `solver='lbfgs'`, `class_weight='balanced'` |
| `logistic_blind_none` | `LogisticRegression` | Blind | 41 | None | `max_iter=1000`, `solver='lbfgs'`, `random_state=42` |
| `logistic_blind_balanced` | `LogisticRegression` | Blind | 41 | Balanced | `max_iter=1000`, `solver='lbfgs'`, `class_weight='balanced'` |
| `knn_full` | `KNeighborsClassifier` | Full | 80 | NOT_SUPPORTED | `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'` |
| `knn_blind` | `KNeighborsClassifier` | Blind | 41 | NOT_SUPPORTED | `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'` |
| `decision_tree_full_none` | `DecisionTreeClassifier` | Full | 80 | None | `criterion='gini'`, `max_depth=12`, `min_samples_split=10` |
| `decision_tree_full_balanced` | `DecisionTreeClassifier` | Full | 80 | Balanced | `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'` |
| `decision_tree_blind_none` | `DecisionTreeClassifier` | Blind | 41 | None | `criterion='gini'`, `max_depth=12`, `min_samples_split=10` |
| `decision_tree_blind_balanced` | `DecisionTreeClassifier` | Blind | 41 | Balanced | `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'` |
| `random_forest_full_none` | `RandomForestClassifier` | Full | 80 | None | `n_estimators=100`, `max_depth=15`, `min_samples_split=5` |
| `random_forest_full_balanced` | `RandomForestClassifier` | Full | 80 | Balanced | `n_estimators=100`, `max_depth=15`, `class_weight='balanced'` |
| `random_forest_blind_none` | `RandomForestClassifier` | Blind | 41 | None | `n_estimators=100`, `max_depth=15`, `min_samples_split=5` |
| `random_forest_blind_balanced` | `RandomForestClassifier` | Blind | 41 | Balanced | `n_estimators=100`, `max_depth=15`, `class_weight='balanced'` |

---

## 2. Benchmark Performance Comparison

Results on the untouched test partition ($N = 4,000$):

| Experiment ID | Model | Feature Set | Class Weight | CV Macro F1 | Test Accuracy | Test Bal. Acc | Test Macro F1 | Test Weighted F1 | Test ROC-AUC (OVR Macro) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_full_none` | Logistic Reg. | Full | None | **0.9683** | **0.9255** | 0.9643 | **0.9664** | **0.9258** | **0.9961** |
| `logistic_full_balanced` | Logistic Reg. | Full | Balanced | 0.9650 | 0.9235 | **0.9647** | 0.9642 | 0.9236 | 0.9960 |
| `random_forest_full_none` | Random Forest | Full | None | 0.9678 | 0.9235 | 0.9643 | 0.9657 | 0.9237 | 0.9956 |
| `random_forest_full_balanced` | Random Forest | Full | Balanced | 0.9675 | 0.9223 | 0.9641 | 0.9652 | 0.9224 | 0.9958 |
| `decision_tree_full_balanced` | Decision Tree | Full | Balanced | 0.9656 | 0.9195 | 0.9635 | 0.9628 | 0.9195 | 0.9942 |
| `decision_tree_full_none` | Decision Tree | Full | None | 0.9643 | 0.9143 | 0.9594 | 0.9618 | 0.9146 | 0.9917 |
| `knn_full` | KNN | Full | N/A | 0.9565 | 0.9085 | 0.9429 | 0.9508 | 0.9089 | 0.9863 |
| `logistic_blind_none` | Logistic Reg. | Blind | None | 0.0681 | 0.2782 | 0.1081 | 0.0688 | 0.1723 | 0.4975 |
| `random_forest_blind_none` | Random Forest | Blind | None | 0.0679 | 0.2682 | 0.1040 | 0.0666 | 0.1655 | 0.5043 |
| `decision_tree_blind_none` | Decision Tree | Blind | None | 0.0822 | 0.2470 | 0.1027 | 0.0801 | 0.1736 | 0.4933 |
| `knn_blind` | KNN | Blind | N/A | 0.0963 | 0.1740 | 0.0942 | 0.0895 | 0.1630 | 0.5025 |
| `random_forest_blind_balanced` | Random Forest | Blind | Balanced | 0.1000 | 0.1472 | 0.0897 | 0.0902 | 0.1538 | 0.4881 |
| `logistic_blind_balanced` | Logistic Reg. | Blind | Balanced | 0.0776 | 0.0845 | 0.0878 | 0.0750 | 0.1039 | 0.4971 |
| `decision_tree_blind_balanced` | Decision Tree | Blind | Balanced | 0.0668 | 0.0707 | 0.0991 | 0.0678 | 0.0660 | 0.4839 |

---

## 3. Key Analytical Findings

### A. The Geographic Signal Finding (Full vs Geographic-Blind)
- In the **Full Metadata** models, performance reaches **~92.5% Accuracy** and **~0.966 Macro F1**, with ROC-AUC exceeding **0.995**.
- In the **Geographic-Blind Baseline**, performance collapses to **17.4%–27.8% Accuracy**, **0.067–0.090 Macro F1**, and **~0.500 ROC-AUC** (equivalent to random guessing).
- *Interpretation*: This directly confirms the Phase 3 discovery ($V = 0.9732$) that cash-out location in cybercrime complaints is intrinsically tied to geographic jurisdiction. Non-spatial attributes (`amount`, `bank`, `rail`, `hour`) do not carry geographic signal, which also validates the absence of artificial target leakage in the synthetic generator.

### B. Class Imbalance & Class Weighting Impact
- Across the Full Metadata models, the natural 23.43:1 imbalance (`Zone_07` = 5,436 vs `Zone_09` = 232) did **not** prevent minority class detection. In fact, `Zone_09` (Alwar) achieved **1.0 Precision, 1.0 Recall, and 1.0 F1 score** across all full models under both `class_weight=None` and `class_weight="balanced"`.
- This occurs because `Zone_09` complaints originate in a geographically distinct, non-overlapping cluster.
- In the Geographic-Blind baseline, balancing class weights forced models to predict minority classes, increasing `Zone_09` recall from 0.0 to 0.13 at the cost of overall accuracy (collapsing from 27.8% to 8.5%).

### C. Zone-Specific Classification Bottlenecks
- Zones with $100\%$ precision/recall in Full Metadata models: `Zone_01`, `Zone_02`, `Zone_03`, `Zone_04`, `Zone_05`, `Zone_09`, `Zone_10`.
- The only zones with misclassifications are:
  - `Zone_07` (East NCR: Noida, Ghaziabad, East Delhi): Precision 91.9%, Recall 78.8%, F1 0.848.
  - `Zone_08` (West NCR: Gurugram, Faridabad, West Delhi): Precision 77.7%, Recall 92.7%, F1 0.846.
  - `Zone_06` (Meerut): Precision 100%, Recall 92.9%, F1 0.963.
- *Geographic Context*: As demonstrated in Phase 3 spatial analysis, `Zone_07` and `Zone_08` are contiguous metropolitan sectors separated by only 21.1 km. Complaints from border zones (e.g., Delhi center) naturally share geographic boundary proximity.

### D. Overfitting Assessment
- Comparing CV Macro F1 to Test Macro F1 across all 14 experiments reveals generalization gaps of less than **0.0098** ($< 1\%$).
- No model exhibited overfitting; training and validation generalizations are aligned.

---

## 4. Probability Output Diagnostics for ATM Candidate Ranking

Downstream ATM candidate ranking (Phase 7) depends directly on predicted zone probability distributions:
- **Calibrated Behavior**: Average predicted confidence on correct predictions is **0.941–0.955**, compared to **0.675–0.723** on incorrect predictions.
- **NCR Boundary Splits**: When a complaint occurs near the border of East and West NCR, the models appropriately split probability mass (e.g., `prob_zone_07 = 0.32`, `prob_zone_08 = 0.68`), providing nuanced soft likelihoods rather than overconfident false certainties.
- **Format Contract**: All test prediction outputs (`reports/predictions/*_predictions.csv`) contain 10 zone probability columns summing to $1.0$.

---

## 5. Model Artifacts & Registry

All 14 complete pipelines are serialized via Joblib under `models/location_classifier/`:
- Complete preprocessing + classification steps are self-contained.
- `models/location_classifier/model_registry.json` indexes training metadata, canonical SHA-256 fingerprint (`2be41885...`), hyperparameters, and performance metrics.
