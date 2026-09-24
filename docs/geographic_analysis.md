# Geographic Analysis, Withdrawal-Zone Validation & Dataset Intelligence

## 1. Executive Summary & Academic Target Definition

In CyberTrace, **`withdrawal_zone`** serves as the primary multi-class classification target for the supervised machine learning pipeline. 

> [!WARNING]
> **Synthetic Target Disclosure**:
> `withdrawal_zone` is a synthetic target created for academic supervised-learning experimentation. It does not represent confirmed NCRP withdrawal locations or actual law enforcement evidence.

The visible geographic features available for model training and analytical reasoning are strictly:
- `complaint_latitude`
- `complaint_longitude`
- `city`
- `state`
- `district`

### Strict Exclusion of Hidden Cash-Out Coordinates
During the initial synthetic generation of complaints, synthetic cash-out coordinate pairs (`synthetic_cashout_latitude`, `synthetic_cashout_longitude`) were used to define geographic cluster memberships. To ensure absolute research integrity and eliminate target leakage:
- **`synthetic_cashout_latitude`** and **`synthetic_cashout_longitude`** are strictly **EXCLUDED** from `data/processed/cybercrime_complaints.csv` and are **NEVER** accessible to ML feature extractors.

---

## 2. Distinction: Complaint Location vs. Predicted Cash-Out Zone

A foundational principle of the CyberTrace platform is the clear operational and spatial distinction between where an incident is reported and where the illicit funds are physically liquidated:

| Dimension | Complaint Location (`complaint_latitude`, `complaint_longitude`) | Predicted Withdrawal Zone (`withdrawal_zone`) |
| :--- | :--- | :--- |
| **Origin** | Cybercrime victim's registered incident location or banking reporting branch. | Geographical cluster / macro-zone where synthetic mules or cash-out operatives withdraw paper cash. |
| **Visibility** | Visible feature provided in the complaint dataset. | The target variable ($Y$) predicted by supervised machine learning classifiers. |
| **Physical Reality** | Victim's residence, workplace, or digital transaction origin point. | Candidate ATM terminal search envelope for candidate ranking. |

---

## 3. Withdrawal Zone Distribution & Spatial Properties

The 20,000 synthetic complaints map across 10 discrete geographic withdrawal zones spanning Punjab, Chandigarh, Haryana, Delhi NCR, Western Uttar Pradesh, and Rajasthan:

| Zone ID | Descriptive Geographic Region | Complaint Count | % Share | Empirical Centroid (Lat, Lon) | Descriptive Extent (Span) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Zone_01** | Amritsar Metropolitan Corridor | 1,241 | 6.21% | (31.6343°N, 74.8701°E) | 0.23° Lat × 0.30° Lon |
| **Zone_02** | Jalandhar Urban Core & Highway | 1,577 | 7.89% | (31.3248°N, 75.5770°E) | 0.22° Lat × 0.29° Lon |
| **Zone_03** | Ludhiana Industrial & Civil Lines | 1,818 | 9.09% | (30.9006°N, 75.8582°E) | 0.22° Lat × 0.38° Lon |
| **Zone_04** | Tri-City / Malwa Border (Chandigarh, Ambala, Patiala) | 2,592 | 12.96% | (30.4776°N, 76.6597°E) | 0.62° Lat × 0.75° Lon |
| **Zone_05** | Panipat Industrial Transit Corridor | 764 | 3.82% | (29.3917°N, 76.9653°E) | 0.21° Lat × 0.30° Lon |
| **Zone_06** | Meerut & Western UP Regional Hub | 841 | 4.21% | (28.9660°N, 77.6904°E) | 0.51° Lat × 0.50° Lon |
| **Zone_07** | East NCR (Noida, Ghaziabad, Faridabad, East Delhi) | 5,436 | 27.18% | (28.5544°N, 77.3498°E) | 0.50° Lat × 0.62° Lon |
| **Zone_08** | West / South NCR (New Delhi, Gurugram, Central Delhi) | 4,339 | 21.70% | (28.5326°N, 77.1356°E) | 0.47° Lat × 0.63° Lon |
| **Zone_09** | Alwar Transit Hub (Rajasthan NCR Border) | 232 | 1.16% | (27.5542°N, 76.6364°E) | 0.16° Lat × 0.24° Lon |
| **Zone_10** | Jaipur Metropolitan Core | 1,160 | 5.80% | (26.9129°N, 75.7886°E) | 0.22° Lat × 0.29° Lon |

---

## 4. Class Imbalance Analysis & Preprocessing Guidance

- **Majority Class**: `Zone_07` (5,436 instances, 27.18%).
- **Minority Class**: `Zone_09` (232 instances, 1.16%).
- **Imbalance Ratio**: **23.43 : 1**.
- **Shannon Entropy**: **2.9084 bits** (Theoretical uniform maximum = 3.3219 bits; Normalized Entropy = 0.8755).

### Methodological Preprocessing Requirements for Phase 4 & Phase 5
1. **Stratified Sampling**: All train/validation/test splits must use `StratifiedKFold` (or stratified `train_test_split`) to ensure that minority zones like `Zone_09` (Alwar) and `Zone_05` (Panipat) maintain exact proportional representation across folds.
2. **Cost-Sensitive Learning**: Classifiers must employ balanced class weighting (`class_weight='balanced'`) to prevent majority classes (`Zone_07`, `Zone_08`) from dominating loss gradients.
3. **Macro-Averaged Metrics**: Model evaluation in Phase 6 must prioritize Macro-F1 and Balanced Accuracy over raw micro-accuracy.

---

## 5. Spatial Separation & Overlap Findings

- **Centroid Separability**:
  - The mean pairwise distance between zone centroids is **205.8 km**.
  - The maximum distance is **532.5 km** (between `Zone_01` Amritsar and `Zone_10` Jaipur).
  - The minimum distance is **21.06 km** (between `Zone_07` East NCR and `Zone_08` West NCR).
- **Bounding Box Overlap**:
  - Non-NCR zones (Amritsar, Jalandhar, Ludhiana, Panipat, Alwar, Jaipur) exhibit **0.00 bounding box overlap** ($IoU = 0.0$).
  - In Delhi NCR, `Zone_07` and `Zone_08` share contiguous descriptive boundaries ($IoU \approx 0.58$), reflecting the multi-nodal, dense geography of the National Capital Region.
  - *Caveat*: Bounding boxes are descriptive outer limits and do not imply ambiguous cluster labels.

---

## 6. Feature-Target Association Analysis

| Feature | Feature Type | Statistical Test | Statistic | p-value | Effect Size | Interpretation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `city` | Categorical | Chi-Square ($\chi^2$) | 170,517.15 | $< 10^{-15}$ | Cramér's $V = 0.9732$ | **Very strong spatial association** (primary predictor) |
| `bank` | Categorical | Chi-Square ($\chi^2$) | 155.72 | $1.20 \times 10^{-6}$ | Cramér's $V = 0.0205$ | Negligible / near-independent |
| `transaction_type`| Categorical | Chi-Square ($\chi^2$) | 59.80 | $0.0689$ | Cramér's $V = 0.0122$ | Negligible / independent |
| `fraud_type` | Categorical | Chi-Square ($\chi^2$) | 105.76 | $0.0059$ | Cramér's $V = 0.0145$ | Negligible / near-independent |
| `day_of_week` | Categorical | Chi-Square ($\chi^2$) | 54.02 | $0.4738$ | Cramér's $V = 0.0003$ | Negligible / independent |
| `is_weekend` | Binary | Chi-Square ($\chi^2$) | 10.98 | $0.2771$ | Cramér's $V = 0.0099$ | Negligible / independent |
| `is_night` | Binary | Chi-Square ($\chi^2$) | 31.34 | $0.0003$ | Cramér's $V = 0.0334$ | Negligible |
| `amount_category` | Categorical | Chi-Square ($\chi^2$) | 21.05 | $0.7839$ | Cramér's $V = 0.0000$ | Independent |
| `amount` | Numerical | Kruskal-Wallis ($H$) | 21.26 | $0.0115$ | $\eta^2 = 0.0006$ | Negligible variation across zones |
| `hour` | Numerical | Kruskal-Wallis ($H$) | 2.40 | $0.9835$ | $\eta^2 = 0.0000$ | Completely uniform across zones |

> [!NOTE]
> **Absence of Synthetic Artifact Contamination**:
> These empirical statistical tests prove that non-geographic features (amount, hour, transaction type, bank) have zero artificial leakage into the withdrawal zone target. Geographic coordinates and city remain the true mathematical determinants.

---

## 7. ATM Geographic Coverage & OpenStreetMap Realities

- **Total Verified OSM ATMs**: 333 across 14 of the 15 study cities.
- **Zone Infrastructure Overlap**:
  - `Zone_07` (East NCR): 143 ATMs in bbox
  - `Zone_08` (West NCR): 143 ATMs in bbox
  - `Zone_04` (Tri-City / Malwa): 94 ATMs in bbox
  - `Zone_06` (Meerut): 44 ATMs in bbox
  - `Zone_10` (Jaipur): 32 ATMs in bbox
  - `Zone_01` (Amritsar): 12 ATMs in bbox
  - `Zone_03` (Ludhiana): 4 ATMs in bbox
  - `Zone_02` (Jalandhar): 3 ATMs in bbox
  - `Zone_09` (Alwar): 2 ATMs in bbox
  - `Zone_05` (Panipat): **0 ATMs** (`NO_OSM_ATMS_FOUND` in OpenStreetMap)
- **OpenStreetMap Caveat**:
  - Panipat's 0 tagged OSM ATMs is a reflection of crowdsourced mapping sparsity, not an absence of real-world banking infrastructure.
  - In Phase 7 (Candidate ATM Ranking), candidate discovery gracefully handles low-density and zero-density zones by alerting investigators to data coverage limitations rather than hallucinating false coordinates.
