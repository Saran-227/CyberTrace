# CyberTrace Model Selection Document

## Decision Summary
- **PRIMARY MODEL**: `random_forest_full_none`
  - Architecture: `RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5)`
  - Preprocessing: `FEATURE_SET_FULL` (80 encoded features via ColumnTransformer)
  - Loss Weighting: `class_weight=None`
  - Performance: **Test Macro F1 = 0.9657**, **Test Accuracy = 0.9235**, **Balanced Accuracy = 0.9643**, **ROC-AUC (OVR Macro) = 0.9956**
  - Generalization Gap: **0.0021** (CV Macro F1 0.9678 vs Test Macro F1 0.9657)

- **FALLBACK MODEL**: `logistic_full_none`
  - Architecture: `LogisticRegression(max_iter=1000, solver='lbfgs')`
  - Preprocessing: `FEATURE_SET_FULL` (80 encoded features via ColumnTransformer)
  - Loss Weighting: `class_weight=None`
  - Performance: **Test Macro F1 = 0.9664**, **Test Accuracy = 0.9255**, **Balanced Accuracy = 0.9643**, **ROC-AUC (OVR Macro) = 0.9961**
  - Generalization Gap: **0.0019** (CV Macro F1 0.9683 vs Test Macro F1 0.9664)

---

## Detailed Selection Rationale

### Why Random Forest is Selected as Primary
1. **Spatial Boundary Geometry**: Decision tree ensembles model non-linear orthogonal bounding boxes between latitude and longitude without requiring linear separability assumptions.
2. **Probability Dispersion for Ranking**: Random Forest class probabilities (averaged across 100 decorrelated trees) provide smooth, granular probability distributions across adjacent NCR zones, which serves as a superior weighting input for Phase 7 ATM ranking.
3. **Robustness to Extreme Outliers**: Invariant to extreme monetary outliers observed in Phase 4.1 (amounts up to ₹2.34M).

### Why Logistic Regression is the Selected Fallback
1. **Inference Latency**: Linear evaluation via dot-product is near-instantaneous (< 5 ms).
2. **Direct Probability Calibration**: Softmax log-odds outputs provide a mathematically continuous baseline if ensemble voting is unavailable.

---

## Operational Limitations
- Both models display spatial ambiguity between `Zone_07` (East NCR) and `Zone_08` (West NCR) due to natural metropolitan proximity (21.1 km).
- Models must not be used without geographic complaint inputs; geographic-blind versions exhibit near-chance performance (~10% balanced accuracy).
