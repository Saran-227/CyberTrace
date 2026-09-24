# CyberTrace Phase 7: ATM Candidate Ranking Engine & Live Dynamic Pipeline

## 1. Executive Summary & Architecture

The CyberTrace ATM Candidate Ranking Engine bridges supervised machine learning zone predictions with open-source physical infrastructure. When an investigator submits complaint metadata and clicks **"ANALYZE CASE"**, the platform executes a dynamic, live pipeline:

```
User Complaint Input
        ↓
Data Validation & Normalization (src/intelligence/case_analysis.py)
        ↓
Feature Preprocessing (src/preprocessing/features.py)
        ↓
Primary Location Classifier (random_forest_full_none / logistic_full_none)
        ↓
Predicted Withdrawal Zone & 10-Zone Probabilities (docs/location_prediction_contract.md)
        ↓
Candidate Zone Selection & Cross-Zone Activation (src/atm/ranking.py)
        ↓
ATM Discovery & Spatial Filtering (333 verified OSM locations, src/atm/loader.py)
        ↓
Transparent Multi-Criteria Candidate Scoring (6 components, weights summing to 1.0)
        ↓
Non-Technical Explainability & Operational Brief (src/intelligence/explanation.py)
        ↓
Structured Intelligence Result Object (docs/atm_ranking_contract.md)
```

> [!IMPORTANT]
> **Claim Boundaries & Academic Disclaimer**:
> CyberTrace produces **probabilistic candidate rankings** to optimize physical investigative triage (CCTV preservation, banking interchange inquiries). The platform **NEVER** confirms that any ATM was the actual cash-out location without authoritative banking logs and physical surveillance footage.

---

## 2. Multi-Criteria Scoring Formula & Exact Weights

Candidate ATMs are evaluated using six transparent scoring components, each scaled to $[0.0, 100.0]$. The overall candidate score is computed as:

$$\text{Overall Score} = \sum_{k=1}^6 w_k \cdot S_k$$

Where the weights $w_k$ are centralized in `src/config.py` and sum strictly to $1.00$:

| Component ($k$) | Weight ($w_k$) | Component Name | Description & Formulation |
| :--- | :---: | :--- | :--- |
| 1 | **0.35** | `zone_probability_score` | Statistical likelihood from Phase 6 location model: $100 \times [0.85 \cdot P(\text{assigned\_zone}) + 0.15 \cdot P(\text{secondary\_zone})]$ |
| 2 | **0.25** | `spatial_score` | Spatial proximity decay over Haversine distance to reference sector centroid: $100 \times \exp(-d / 25.0)$ |
| 3 | **0.15** | `bank_score` | Victim bank vs OSM ATM operator compatibility tier (Exact: 100, Partial: 70, Unknown: 50, Mismatch: 20) |
| 4 | **0.10** | `activity_score` | Simulated historical operational activity score derived from temporal window around incident hour ($0 - 100$) |
| 5 | **0.10** | `time_score` | Operational hours accessibility (24x7: 100, Day/Unknown: 85/50, Night non-24x7: 25) |
| 6 | **0.05** | `amount_compatibility_score`| Compatibility between complaint amount and ATM simulated withdrawal profile ($40 - 90$) |

---

## 3. Candidate Zone Selection & Cross-Zone Search

Phase 6 error analysis demonstrated that **96.08% of all model errors** occur across the National Capital Region (NCR) border between East NCR (`Zone_07`) and West NCR (`Zone_08`). Rather than blindly restricting candidate discovery to the top-probability class, Phase 7 implements dynamic candidate-zone selection:

1. **High Confidence ($\Delta \ge 0.30$)**:
   - Primary candidate zone = `predicted_zone`.
   - `cross_zone_search = false`.
   - Applied when the classifier is decisive (e.g. non-NCR zones like Amritsar, Jalandhar, Jaipur).
2. **Medium Confidence ($0.15 \le \Delta < 0.30$)**:
   - Includes at minimum `[predicted_zone, second_best_zone]`.
   - `cross_zone_search = true`.
   - The UI explicitly displays: *"Cross-boundary candidate search active"*. Both East and West NCR ATMs are evaluated concurrently.
3. **Low Confidence ($\Delta < 0.15$)**:
   - Accumulates candidate zones in descending order of probability until cumulative probability mass reaches $\ge 0.80$ (`CANDIDATE_SEARCH_CONFIG["cumulative_probability_threshold"]`).
   - `cross_zone_search = true`.
4. **Panipat Graceful Fallback (`Zone_05`)**:
   - When `Zone_05` is predicted, the system recognizes that Panipat has 0 OSM ATMs and automatically includes adjacent candidate zones to provide valid candidate cash points without crashing.

---

## 4. Component Scoring Logics

### 4.1 Bank Compatibility Matching
- **Input**: Complaint victim bank, ATM `bank`, ATM `operator` from OpenStreetMap.
- **Normalization**: Standardizes Indian commercial banking synonyms (`sbi` $\leftrightarrow$ `State Bank of India`, `pnb` $\leftrightarrow$ `Punjab National Bank`, `bob` $\leftrightarrow$ `Bank of Baroda`).
- **Tiers**:
  - `EXACT_MATCH` (100.0): ATM is operated by or branded with the victim's issuing institution.
  - `PARTIAL_MATCH` (70.0): Substantive name overlap or banking subsidiary.
  - `UNKNOWN` (50.0): OSM record contains `unknown` bank/operator data; neutral scoring applied without fabricating information.
  - `MISMATCH` (20.0): Explicitly different bank brand (e.g. victim is SBI, ATM is ICICI Bank).

### 4.2 Time Compatibility Matching
- **Input**: Incident hour (0–23) and ATM `is_24x7` metadata.
- **Tiers**:
  - `OPEN_COMPATIBLE` (100.0): Explicitly marked 24x7 in OpenStreetMap.
  - `OPEN_COMPATIBLE` (85.0): Not marked 24x7, but incident occurred during regular daytime hours (06:00 to 21:59).
  - `UNKNOWN` (50.0): No operating hour data in OpenStreetMap; neutral scoring applied.
  - `POSSIBLY_INCOMPATIBLE` (25.0): Explicitly non-24x7 ATM during night-time incident (22:00 to 05:59).

### 4.3 Historical Operational Activity
- **Source**: 719,280 hourly synthetic records (`data/processed/atm_activity.csv`).
- **Temporal Window**: Evaluates simulated activity within a configurable $\pm 2$ hour window around the incident time.
- **Metrics**: Hourly transaction volume, cash withdrawal counts, high-value withdrawal occurrences, and mean amount.
- **Terminology**: Strictly framed as *"simulated historical operational activity"*, never *"fraud activity"*.

### 4.4 Amount Compatibility
- Evaluates whether the disputed complaint amount aligns with the simulated operational profile of the candidate cash point:
  - High-value complaints ($\ge ₹25,000$) match ATMs with historical high-volume or high-value withdrawal capability ($90.0$).
  - Standard complaints ($₹5,000 - ₹25,000$) match typical branch ATM throughput ($85.0$).
  - Evaluated as operational capacity compatibility, never as proof of physical transaction.

---

## 5. Performance & Data Provenance

1. **In-Memory Caching (`src/atm/loader.py`)**:
   - Verified ATM locations (333 records) and aggregated hourly activity profiles (333 ATMs $\times$ 24 hours) are cached in memory.
   - Live case analysis executes in **$< 15\text{ ms}$** on average.
2. **OpenStreetMap Provenance**:
   - All 333 ATM points of interest are sourced from verified OpenStreetMap nodes (`OSM-NODE-*`) under the Open Database License (ODbL).
3. **Synthetic Baseline Disclosures**:
   - Historical activity is explicitly synthetic and reproducible (Seed: 42).
