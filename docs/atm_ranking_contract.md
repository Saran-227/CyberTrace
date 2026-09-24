# CyberTrace Phase 7: ATM Candidate Ranking Contract

## 1. Overview & Downstream Interface

The ATM Candidate Ranking Engine receives the location prediction contract produced in Phase 6, filters verified OpenStreetMap ATM locations, scores each candidate across six transparent multi-criteria dimensions, and generates a structured ranking object for the interactive Leaflet map (Phase 8), executive intelligence report (Phase 9), and Streamlit workbench (Phase 10).

---

## 2. Schema Specification (Phase 7 Contract)

```json
{
  "case_id": "CT-2026-8812",
  "model_id": "random_forest_full_none",
  "predicted_zone": "Zone_02",
  "confidence_tier": "HIGH",
  "prediction_confidence": 0.9998,
  "candidate_zones": ["Zone_02"],
  "cross_zone_search": false,
  "total_atms_evaluated": 10,
  "scoring_weights": {
    "zone_probability": 0.35,
    "spatial": 0.25,
    "bank": 0.15,
    "activity": 0.10,
    "time": 0.10,
    "amount": 0.05
  },
  "ranked_atms": [
    {
      "rank": 1,
      "designation": "Highest-ranked candidate",
      "atm_id": "OSM-NODE-5094962456",
      "bank": "ICICI Bank",
      "operator": "ICICI",
      "latitude": 31.316589,
      "longitude": 75.589165,
      "city": "Jalandhar",
      "district": "Jalandhar",
      "state": "Punjab",
      "zone": "Zone_02",
      "secondary_zone": "Zone_03",
      "distance_km": 1.48,
      "overall_score": 77.52,
      "zone_probability_score": 85.0,
      "spatial_score": 94.25,
      "bank_score": 100.0,
      "time_score": 50.0,
      "activity_score": 27.57,
      "amount_compatibility_score": 90.0,
      "evidence_flags": [
        "EXACT_BANK_MATCH",
        "UNKNOWN_HOURS",
        "PRIMARY_ZONE_CANDIDATE",
        "AMOUNT_COMPATIBLE"
      ],
      "explanation": "Assigned to Zone_02 (1.5 km from reference sector center; zone likelihood score 85.0/100). Strong bank compatibility with victim bank 'ICICI Bank'.",
      "source": "OpenStreetMap"
    }
  ]
}
```

---

## 3. Field Definitions & Bounds

| Field | Type | Bounds / Format | Description |
| :--- | :--- | :--- | :--- |
| `case_id` | String | UTF-8 identifier | Reference complaint identifier from intake |
| `model_id` | String | `random_forest_full_none` / `logistic_full_none` | Identifier of active location classifier |
| `predicted_zone` | String | `Zone_01` to `Zone_10` | Statistical primary withdrawal zone |
| `confidence_tier` | String | `HIGH`, `MEDIUM`, `LOW` | Diagnostic uncertainty tier based on margin $\Delta$ |
| `prediction_confidence`| Float | $[0.0, 1.0]$ | Model probability of the top zone |
| `candidate_zones` | List[String]| Subsets of `Zone_01`..`Zone_10` | Zones evaluated for ATM candidates |
| `cross_zone_search` | Boolean | `true` / `false` | True when secondary zones were searched |
| `total_atms_evaluated` | Integer | $\ge 0$ | Total ATMs scored in candidate zones |
| `scoring_weights` | Object | Sums strictly to $1.0$ | Exact weights applied during candidate scoring |
| `ranked_atms` | List[Object] | Ordered by `overall_score` descending | Truncated top candidates (default Top 10) |

### Candidate ATM Attributes:
- `rank`: Integer 1 to $N$, where 1 is the highest scoring candidate.
- `overall_score`: Float in $[0.0, 100.0]$, computed as the weighted sum of component scores.
- `zone_probability_score`: Float in $[0.0, 100.0]$, reflecting primary and secondary zone model likelihood.
- `spatial_score`: Float in $[0.0, 100.0]$, proximity decay over distance (km) to zone reference centroid.
- `bank_score`: Float in $\{20.0, 50.0, 70.0, 100.0\}$, reflecting operator compatibility.
- `time_score`: Float in $\{25.0, 50.0, 85.0, 100.0\}$, reflecting operational availability.
- `activity_score`: Float in $[0.0, 100.0]$, simulated historical operational activity level.
- `amount_compatibility_score`: Float in $[0.0, 100.0]$, historical withdrawal profile compatibility.
- `evidence_flags`: Descriptive classification tokens for UI filtering and investigative auditing.
- `source`: Attribution string (`OpenStreetMap`).
