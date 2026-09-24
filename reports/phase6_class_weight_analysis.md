# CyberTrace Phase 6: Class Weight Analysis & Findings

## Objective
Evaluate whether cost-sensitive `class_weight="balanced"` improves minority-zone classification in light of the **23.43:1 natural class imbalance** (`Zone_07` = 5,436 vs `Zone_09` = 232).

---

## 1. Summary of Empirical Findings

| Model | Feature Set | Unweighted Macro F1 | Balanced Macro F1 | F1 Delta | Unweighted Zone_09 Recall | Balanced Zone_09 Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Full | 0.9618 | 0.9628 | +0.0010 | 1.0000 | 1.0000 |
| **Decision Tree** | Blind | 0.0801 | 0.0678 | -0.0123 | 0.0000 | 0.0652 |
| **Logistic Regression** | Full | 0.9664 | 0.9642 | -0.0022 | 1.0000 | 1.0000 |
| **Logistic Regression** | Blind | 0.0688 | 0.0750 | +0.0062 | 0.0000 | 0.1304 |
| **Random Forest** | Full | 0.9657 | 0.9652 | -0.0005 | 1.0000 | 1.0000 |
| **Random Forest** | Blind | 0.0666 | 0.0902 | +0.0236 | 0.0000 | 0.0000 |

---

## 2. Key Insights
1. **Full Metadata Invariance**: In the Full Metadata configuration, class weighting produces negligible variation ($|\Delta 	ext{Macro F1}| \le 0.0022$). The minority class `Zone_09` (Alwar) achieves **100% recall and 100% precision** even under `class_weight=None`.
   - *Explanation*: Geographic coordinates and administrative boundaries uniquely partition `Zone_09` from Punjab and NCR complaints, rendering spatial separation impervious to frequency imbalance.
2. **Geographic-Blind Distortion**: In the Geographic-Blind baseline, enabling balanced weighting increases minority recall at the expense of collapsing overall accuracy (from ~27.8% down to 8.5%).
3. **Operational Recommendation**: For the production location model, `class_weight=None` is preferred for Random Forest and Logistic Regression as it preserves well-calibrated posterior probabilities without artificially distorting likelihood ratios.
