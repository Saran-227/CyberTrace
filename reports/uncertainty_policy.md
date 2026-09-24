# CyberTrace Uncertainty & Prediction Confidence Policy

## Operational Philosophy
CyberTrace operates as an **investigative decision-support system**, not an autonomous judicial verdict. All model outputs must be presented with explicit statistical uncertainty to prevent premature operational conclusions.

---

## Empirical Confidence Tiers

Based on empirical probability margin diagnostics across 4,000 test cases:

| Confidence Tier | Probability Margin Criterion | Empirical Accuracy | Recommended Operational Action |
| :--- | :---: | :---: | :--- |
| **HIGH CONFIDENCE** | $	ext{Margin} \ge 0.30$ | **~98.5%** | Primary operational focus on Top Predicted Zone. Initiate candidate ATM discovery immediately. |
| **MEDIUM CONFIDENCE** | $0.15 \le 	ext{Margin} < 0.30$ | **~82.0%** | Investigate Top Predicted Zone, but maintain secondary candidate ATM query in Second Best Zone. |
| **LOW CONFIDENCE (AMBIGUOUS)** | $	ext{Margin} < 0.15$ | **~64.0%** | Significant spatial ambiguity (typically East vs West NCR boundary). Dual-zone candidate ranking is mandatory. |

---

## UI Presentation Guidelines
1. **Terminology Mandate**:
   - Use: *"Predicted Cash-Out Sector"*, *"Statistical Likelihood"*, *"Candidate ATM Ranking"*.
   - Prohibited: *"Confirmed Cash-Out Zone"*, *"Actual ATM Used"*, *"Definitive Fraud Location"*.
2. **Dual-Zone Display**: When $	ext{Margin} < 0.20$, the interface must prominently display both the primary and runner-up zones with their respective probabilities.
