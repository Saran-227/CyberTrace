# CyberTrace Phase 3: Geographic Target Validation & Dataset Intelligence Assessment

## Executive Summary
This report provides a rigorous empirical and spatial evaluation of the `withdrawal_zone` classification target across the 20,000 synthetic cybercrime complaint records and 333 OpenStreetMap candidate ATM locations. 

> [!IMPORTANT]
> **Synthetic Target Disclosure**: `withdrawal_zone` is a synthetic target created for academic supervised-learning experimentation. It does not represent confirmed NCRP withdrawal locations or real banking audit trails. The visible geographic features available to models are exclusively `complaint_latitude` and `complaint_longitude`. Hidden cash-out coordinates are strictly excluded.

---

## 14-Point Target Quality Assessment

### 1. Are all 10 zones represented?
**YES**. All 10 configured withdrawal zones (`Zone_01` through `Zone_10`) are populated in the processed complaints dataset. No empty or missing zone classes exist.

### 2. How imbalanced are the 10 zones?
**MODERATE TO HIGH CLASS IMBALANCE**.
- **Majority Class**: `Zone_07` with **5,436 complaints (27.18%)**.
- **Minority Class**: `Zone_09` with **232 complaints (1.16%)**.
- **Imbalance Ratio**: **23.43 : 1** between majority and minority classes.
- **Shannon Entropy**: **2.8894 bits** (compared to maximum possible theoretical uniform entropy of $\log_2(10) \approx 3.3219$ bits; Normalized Entropy = **0.8698**).
- **ML Preprocessing Recommendation**: Phase 4 and Phase 5 must employ stratified train/test splitting (`StratifiedKFold`) and class-weighted objective functions (`class_weight='balanced'`) to prevent minority classes (`Zone_09`, `Zone_05`, `Zone_06`) from being ignored by classifiers.

### 3. Are zone centroids geographically separated?
**YES**. Zone centroids exhibit substantial spatial separation across North India:
- **Maximum Distance**: **531.0 km** (between `Zone_01` Amritsar and `Zone_10` Jaipur).
- **Mean Pairwise Distance**: **205.8 km** across all non-identical zone centroid pairs.
- **Minimum Distance**: **20.1 km** (between `Zone_07` East NCR and `Zone_08` West NCR).

### 4. Are there zones with substantial geographic overlap?
**DESCRIPTIVE BOUNDING BOX OVERLAP EXISTS IN NCR ONLY**:
- `Zone_07` (East NCR: Noida, Faridabad, Ghaziabad, East Delhi) and `Zone_08` (West NCR: New Delhi, Gurugram, South Delhi) share a descriptive bounding box overlap (IoU = 0.58).
- **Critical Limitation Notice**: Descriptive bounding boxes are rectangular envelopes $[\min, \max]$, not actual non-convex municipal boundaries. Within NCR, complaints naturally form contiguous urban density corridors across administrative borders.
- All non-NCR zones (Amritsar `Zone_01`, Jalandhar `Zone_02`, Ludhiana `Zone_03`, Panipat `Zone_05`, Alwar `Zone_09`, Jaipur `Zone_10`) exhibit **zero (0.00) bounding box overlap**.

### 5. Are zones strongly concentrated by city?
**YES, VERY STRONGLY**.
- Statistical association via Chi-Square test yields $\chi^2 = 170,517.15$ ($p < 10^-15$, Cramér's $V = \mathbf0.9732$).
- Punjab and Rajasthan zones are virtually 1:1 city-to-zone mappings (e.g., Amritsar $\to$ `Zone_01`, Jalandhar $\to$ `Zone_02`, Ludhiana $\to$ `Zone_03`, Alwar $\to$ `Zone_09`, Jaipur $\to$ `Zone_10`).
- `Zone_04` aggregates the Tri-City / Malwa border corridor (Ambala, Chandigarh, Patiala).
- In Delhi NCR, complaints are split between `Zone_07` and `Zone_08`. Thus, city alone is strongly predictive but **not completely deterministic** in dense metropolitan hubs.

### 6. Are there meaningful relationships with bank?
**NEGLIGIBLE**.
- $\chi^2 = 155.72$, $p = 1.20 \times 10^-6$, Cramér's $V = \mathbf0.0205$.
- Complainant bank accounts follow national commercial banking market shares uniformly across all withdrawal zones. No individual bank has an artificial bias towards a specific withdrawal zone.

### 7. Are there relationships with transaction type?
**NEGLIGIBLE**.
- $\chi^2 = 59.80$, $p = 0.0689$, Cramér's $V = \mathbf0.0122$.
- Digital payment rails (UPI, IMPS, NetBanking, Debit Card, etc.) are distributed evenly across zones without statistically significant deviation.

### 8. Are there relationships with fraud type?
**NEGLIGIBLE**.
- $\chi^2 = 105.76$, $p = 0.0059$, Cramér's $V = \mathbf0.0145$.
- Modus operandi categories (Investment Scams, KYC Phishing, Job Fraud, Remote Access, Loan Apps, Sextortion) appear across all zones in proportion to overall fraud prevalence.

### 9. Are there temporal relationships?
**UNIFORM ACROSS TIME**.
- **Hour of Day**: Kruskal-Wallis $H = 2.40$, $p = 0.9835$, $\eta^2 = 0.0000$. Hourly distribution is consistent across all zones.
- **Day of Week**: $\chi^2 = 54.02$, $p = 0.4738$, Cramér's $V = 0.0003$.
- **Night vs Day**: Daytime accounts for ~85% of complaints across all zones, and night accounts for ~15%, with negligible effect size (Cramér's $V = 0.0334$).

### 10. Are there amount-related differences?
**MINIMAL / NEGLIGIBLE**.
- Kruskal-Wallis $H = 21.26$, $p = 0.0115$, $\eta^2 = \mathbf0.0006$.
- Median transaction amounts across all 10 zones remain consistent between **₹24,500 and ₹26,000 INR**, with matching Interquartile Ranges (IQR ~₹34,000 INR). Right-skewed distribution characteristics are uniform across classes.

### 11. Does ATM coverage sufficiently overlap the study geography?
**YES, WITH NOTABLE HETEROGENEITY**:
- 9 of the 10 withdrawal zones have confirmed OSM ATM infrastructure within their descriptive boundaries.
- `Zone_07` (143 ATMs), `Zone_08` (143 ATMs), and `Zone_04` (94 ATMs) have extensive coverage.
- `Zone_05` (Panipat) has **0 ATMs** because Panipat currently contains zero tagged ATM points in OpenStreetMap (`NO_OSM_ATMS_FOUND`).
- `Zone_09` (Alwar) has 2 ATMs, `Zone_02` (Jalandhar) has 3 ATMs, and `Zone_03` (Ludhiana) has 4 ATMs.
- Candidate ATM discovery in Phase 7 must properly account for regional density differences.

### 12. Are there data-quality issues?
**NO DATA QUALITY ISSUES DISCOVERED**.
- Duplicate rows: **0**.
- Missing values in critical fields: **0**.
- Out-of-bounds coordinates: **0**.
- Invalid or non-positive amounts: **0**.
- Unparseable dates: **0**.
- No modification of core data files was necessary.

### 13. Is there any evidence of target leakage?
**ZERO TARGET LEAKAGE CONFIRMED**:
- `synthetic_cashout_latitude` and `synthetic_cashout_longitude` are strictly **ABSENT** from `cybercrime_complaints.csv`.
- Non-geographic features (bank, amount, time, fraud type) show negligible associations, confirming they were not synthetically contaminated by the target class.
- Visible complaint coordinates and city information reflect genuine incident reporting geography.

### 14. Is the target suitable for proceeding to supervised classification?
**YES, HIGHLY SUITABLE**.
- The 10 withdrawal zones represent distinct, learnable spatial clusters that map meaningfully to North Indian urban and commercial corridors.
- The high spatial separability and strong geographic correlation confirm that supervised spatial classification models (KNN, Random Forest, Decision Tree, Logistic Regression) will be able to learn genuine geographic decision boundaries.
- **Phase 4 Readiness**: Proceed with confidence to Phase 4 (Preprocessing, feature encoding, scaling, and stratified splitting).
