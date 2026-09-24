# CyberTrace Executive Intelligence Report Contract (Phase 9)

## 1. Overview
The Executive Intelligence Report Generator translates the live machine learning and geospatial candidate analysis output into a formal, human-readable executive intelligence brief available in **HTML** and **PDF** formats.

The report operates strictly as a downstream presentation consumer of `src/intelligence/case_analysis.py`. It **does not run ML inference** and **does not independently rank ATM candidates**.

---

## 2. Input Data Contract

The reporting pipeline consumes the authoritative `case_analysis` object produced by Phase 7 and stored in `st.session_state["case_analysis"]`.

```json
{
  "case_id": "CT-2026-8812",
  "status": "SUCCESS",
  "timestamp": "2026-09-24T14:30:00",
  "execution_time_ms": 42.5,
  "model_id": "random_forest_full_none",

  "case_data": {
    "complaint_id": "CT-2026-8812",
    "complaint_date": "2026-09-24",
    "complaint_time": "14:30:00",
    "amount": 35000.0,
    "amount_category": "Tier_3_Medium_High",
    "bank": "HDFC",
    "transaction_type": "UPI",
    "fraud_type": "Phishing/Smishing",
    "city": "Jalandhar",
    "state": "Punjab",
    "district": "Jalandhar",
    "complaint_latitude": 31.3260,
    "complaint_longitude": 75.5762,
    "hour": 14,
    "day_of_week": 3
  },

  "prediction": {
    "predicted_zone": "Zone_02",
    "prediction_confidence": 0.884,
    "second_best_zone": "Zone_01",
    "probability_margin": 0.762,
    "confidence_tier": "HIGH",
    "model_id": "random_forest_full_none",
    "zone_probabilities": {
      "Zone_01": 0.082,
      "Zone_02": 0.884,
      "Zone_03": 0.012,
      "Zone_04": 0.006,
      "Zone_05": 0.003,
      "Zone_06": 0.003,
      "Zone_07": 0.004,
      "Zone_08": 0.003,
      "Zone_09": 0.001,
      "Zone_10": 0.002
    }
  },

  "ranking": {
    "predicted_zone": "Zone_02",
    "candidate_zones": ["Zone_02"],
    "cross_zone_search": false,
    "total_atms_evaluated": 28,
    "ranked_atms": [
      {
        "rank": 1,
        "atm_id": "node/431289412",
        "bank": "HDFC Bank",
        "operator": "HDFC Bank Ltd",
        "city": "Jalandhar",
        "district": "Jalandhar",
        "zone": "Zone_02",
        "distance_km": 1.48,
        "overall_score": 84.62,
        "zone_score": 88.4,
        "spatial_score": 85.2,
        "bank_score": 100.0,
        "time_score": 80.0,
        "activity_score": 74.5,
        "amount_score": 80.0,
        "latitude": 31.3312,
        "longitude": 75.5810,
        "designation": "Highest-ranked candidate",
        "evidence_flags": {
          "bank_match": "Exact Match",
          "operating_hours": "24x7",
          "activity_level": "High"
        }
      }
    ],
    "scoring_weights": {
      "zone_probability": 0.30,
      "spatial_proximity": 0.25,
      "bank_compatibility": 0.20,
      "operating_hours": 0.10,
      "activity_compatibility": 0.10,
      "amount_compatibility": 0.05
    }
  },

  "map_data": {
    "complaint_point": {
      "latitude": 31.3260,
      "longitude": 75.5762,
      "city": "Jalandhar"
    },
    "bounds": {
      "south_west": [31.25, 75.50],
      "north_east": [31.40, 75.65]
    }
  },

  "explanation": {
    "zone_summary": "The intelligence platform predicts Zone_02 as the highest-probability cash-out area with 88.4% model confidence (Tier: HIGH, margin 0.76).",
    "atm_summary": "Candidate ATM 'node/431289412' in Jalandhar operated by HDFC Bank was identified as the highest-ranked candidate (composite score: 84.6/100, 1.5 km from sector reference center).",
    "key_signals": [
      "Geographic complaint origin in Jalandhar, Punjab.",
      "Victim financial institution recorded as HDFC.",
      "Disputed amount INR 35,000 (Tier_3_Medium_High).",
      "Incident timing recorded at 14:00 hours (daytime)."
    ],
    "recommended_focus": [
      "Prioritize CCTV footage and physical audit requests for top candidate ATMs.",
      "Request formal transaction interchange reconciliation from HDFC.",
      "Cross-reference suspect device telecommunications tower pings against candidate ATM coordinates.",
      "File intelligence brief under National Cyber Crime Reporting Portal reference."
    ],
    "evidence_disclaimer": "All candidate rankings represent mathematical and geographic likelihoods. Verification using official banking and surveillance evidence is strictly mandatory prior to legal enforcement."
  },

  "provenance": {
    "model_family": "RandomForestClassifier",
    "model_artifact": "models/location_classifier/random_forest_full_none.joblib",
    "atm_source": "OpenStreetMap Verified Geographic Infrastructure (ODbL)",
    "activity_source": "Synthetic ATM Operational Activity (90-day baseline, zero target leakage)",
    "ranking_engine_version": "v1.0",
    "report_version": "v1.0"
  }
}
```

---

## 3. Mandatory Report Sections

Every generated report (HTML and PDF) contains 14 structured sections:

1. **Executive Summary**: High-level incident and prediction briefing card for non-technical leadership.
2. **Case Information**: Structured breakdown of victim report, financial value, payment rail, and jurisdiction.
3. **Incident Profile**: Objective analytical narrative framing the statistical classification context.
4. **Location Prediction**: Statistical distribution across all 10 zones with model ID, margin, and sorted visual probability bar chart.
5. **Geographic Assessment**: Geographic coordinates, sector boundaries, and cross-zone search parameters.
6. **Candidate ATM Assessment**: Top 10 ranked OpenStreetMap candidate ATMs with distance and composite scores.
7. **Highest-Ranked ATM Candidate Spotlight**: Detailed card for Candidate #1 with individual score component breakdown.
8. **Analytical Reasoning**: Contributing signals and explanation breakdown from the explanation engine.
9. **Confidence and Uncertainty**: Formal threshold diagnostics (HIGH $\Delta \ge 0.30$, MEDIUM $0.15 \le \Delta < 0.30$, LOW $\Delta < 0.15$).
10. **Data Sources and Provenance**: Clear attribution distinguishing synthetic research data from verified external OSM data.
11. **Methodological Limitations**: 10 explicit constraints acknowledging lack of real banking logs and reliance on crowdsourced infrastructure.
12. **Verification Recommendations**: Actionable, legally compliant investigative steps for authorized law enforcement.
13. **Evidence Hierarchy**: Strict separation between Level 1 (Model Output), Level 2 (Analytical Candidate), and Level 3 (External Bank/CCTV Evidence).
14. **Official Cybercrime Reporting**: Direct links to `cybercrime.gov.in` and National Helpline `1930`.

---

## 4. Claim Boundaries & Legal Language

The report enforces strict evidentiary boundary compliance:
- **PROHIBITED TERMS**: *"Confirmed ATM"*, *"Fraud ATM"*, *"Criminal cash-out location"*, *"Actual withdrawal site"*, *"Perpetrator location"*.
- **MANDATORY TERMS**: *"Predicted withdrawal zone"*, *"Highest-ranked ATM candidate"*, *"Model-supported candidate"*, *"Analytical proximity"*.
- **SYNTHETIC DISCLOSURE**: Explicitly discloses that complaint records and ATM activity are synthetic baselines created for academic research.
- **OPENSTREETMAP ATTRIBUTION**: Retains `Map data © OpenStreetMap contributors` under the Open Database License (ODbL).
