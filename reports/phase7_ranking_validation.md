# CyberTrace Phase 7: ATM Candidate Ranking Validation & Verification Report

## 1. Executive Summary

Phase 7 successfully builds, validates, and deploys the **ATM Candidate Ranking Engine** and the **Live Dynamic Case Analysis Service Layer** for CyberTrace. 

The ranking engine unites:
1. Supervised location prediction probabilities from Phase 6 (`random_forest_full_none` / `logistic_full_none`)
2. 333 verified OpenStreetMap ATM locations across 14 study cities
3. 719,280 hourly synthetic ATM operational activity records
4. Multi-criteria transparent candidate scoring across 6 documented dimensions
5. Dynamic cross-zone candidate search for National Capital Region (NCR) boundary cases
6. Non-technical investigative brief and evidence disclaimers

---

## 2. Infrastructure & Ingestion Statistics

| Metric | Measured Value | Validation Status |
| :--- | :--- | :--- |
| **Total Verified ATM Points** | 333 OSM nodes | Verified non-duplicate, valid coordinates |
| **Cities with ATM Coverage** | 14 of 15 study cities | Verified (Panipat: 0 OSM ATMs, handled via fallback) |
| **Historical Activity Records** | 719,280 hourly records | Validated: 90 days, zero target leakage, zero fraud labels |
| **Unique ATMs in Activity** | 333 ATMs (100% coverage) | Fully aligned with physical locations |
| **Canonical Withdrawal Zones** | 10 operational sectors | Centroids verified from Phase 3 empirical coordinates |
| **Average End-to-End Latency** | **12.4 ms** | Sub-second real-time responsiveness |
| **Test Suite Passing** | **142 / 142 tests passing** | 24 new Phase 7 tests + 118 existing regression tests |

---

## 3. Transparent Multi-Criteria Scoring Framework

The candidate ranking score is computed as:

$$\text{Overall Score} = \sum_{k=1}^6 w_k \cdot S_k \quad \in [0.0, 100.0]$$

| Component ($S_k$) | Weight ($w_k$) | Description | Range |
| :--- | :---: | :--- | :---: |
| `zone_probability_score` | 0.35 | Statistical likelihood from ML location model | 0.0 – 100.0 |
| `spatial_score` | 0.25 | Exponential distance decay to sector reference center | 0.0 – 100.0 |
| `bank_score` | 0.15 | Compatibility tier between victim bank and ATM operator | 20.0 – 100.0 |
| `activity_score` | 0.10 | Simulated historical operational activity in $\pm 2$ hr window | 0.0 – 100.0 |
| `time_score` | 0.10 | Operational hours compatibility (24x7 accessibility) | 25.0 – 100.0 |
| `amount_compatibility_score` | 0.05 | Historic withdrawal volume and high-value compatibility | 40.0 – 90.0 |

---

## 4. Deterministic Ranking Validation Scenarios

### Scenario A: High-Confidence Single-Zone Prediction (Jalandhar, Punjab)
- **Input**: Complaint ID `CT-JAL-01`, City: Jalandhar, Amount: ₹45,000, Bank: ICICI Bank, Time: 23:15 (Night).
- **ML Output**: Predicted Zone: `Zone_02` (Confidence: 100.0%, Margin: 1.00, Tier: HIGH).
- **Candidate Zones**: `['Zone_02']` (Cross-zone: False).
- **Ranking Results**:
  1. `OSM-NODE-5094962456` (ICICI Bank): Overall Score: **77.52/100** (Distance: 1.48 km, Bank Score: 100.0, Flags: `['EXACT_BANK_MATCH', 'UNKNOWN_HOURS', 'PRIMARY_ZONE_CANDIDATE', 'AMOUNT_COMPATIBLE']`).
  2. `OSM-NODE-5095007519` (Syndicate Bank): Overall Score: **66.81/100** (Distance: 0.25 km, Bank Score: 20.0).
  3. `OSM-NODE-1845296821` (Punjab National Bank): Overall Score: **65.08/100** (Distance: 2.64 km, Bank Score: 20.0).
- **Finding**: Proximity alone did not crown Syndicate Bank; the exact bank match of ICICI Bank elevated it to #1 candidate despite slightly greater distance, demonstrating multi-criteria balance.

### Scenario B: Medium-Confidence NCR Cross-Boundary Search (Noida / Delhi Border)
- **Input**: Complaint ID `CT-NCR-02`, City: Noida, Coordinates: (28.5355, 77.3910), Amount: ₹30,000, Bank: State Bank of India.
- **ML Output**: Predicted Zone: `Zone_07` (East NCR: 58.4%), Runner-up: `Zone_08` (West NCR: 40.2%), Margin: 0.18 (Tier: MEDIUM).
- **Candidate Zones**: `['Zone_07', 'Zone_08']` (Cross-zone: True).
- **Ranking Results**:
  - Automatically evaluated ATMs from both East NCR (Noida/Ghaziabad) and West NCR (New Delhi/Gurugram/Faridabad).
  - The UI presented: *"Cross-boundary candidate search active"*.
  - Evaluated candidate cash points across both sectors without truncation.

### Scenario C: Edge Case — Panipat (`Zone_05`, Zero OSM ATMs)
- **Input**: Complaint ID `CT-PAN-03`, City: Panipat, Amount: ₹25,000, Bank: Punjab National Bank.
- **ML Output**: Predicted Zone: `Zone_05` (Panipat).
- **Candidate Zones**: Automatically included runner-up `Zone_08` as adjacent search sector.
- **Status**: Completed gracefully without crashing; returned 10 evaluated candidate ATMs from adjacent regional sectors with clear investigative notes.

---

## 5. Test Suite Verification

Comprehensive unit tests in `tests/test_phase7_ranking.py` verified:
- ATM dataset completeness and uniqueness (333 records)
- Zero target leakage in ATM activity logs
- Scoring weights strict summation to 1.0
- Determinism and sorting order of ranked output
- Bank matching tiers and absence of text fabrication
- Cross-zone expansion on low/medium margin
- Output schema compliance with Phase 7 contract

```bash
$ python -m pytest tests/ -v
============================ 142 passed in 13.20s =============================
```

---

## 6. Known Limitations & Claim Boundaries

1. **Academic Intelligence Prototype**: Candidate ATM rankings reflect statistical likelihoods and open spatial data; they do not establish confirmed physical withdrawal.
2. **OpenStreetMap Data Sparsity**: Certain cities (e.g. Panipat) possess 0 tagged ATMs in OpenStreetMap. Fallback search logic ensures platform continuity, but geographic coverage depends on public OSM contributions.
3. **Synthetic Activity Baseline**: ATM activity scores are simulated operational baselines, not actual banking core ledger records.
