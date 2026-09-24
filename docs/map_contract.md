# CyberTrace Phase 8: Interactive Map Contract

## 1. Overview & Architectural Boundaries

The CyberTrace Interactive Map is a **visualization layer** embedded via Leaflet.js and OpenStreetMap cartographic tiles. The map strictly consumes the validated output of the Phase 7 case analysis pipeline (`analyze_case()`) via `src/geographic/map_data.py:prepare_map_data()`.

> [!IMPORTANT]
> **Strict Non-Computation Principle**:
> The map component **NEVER** independently runs machine learning inference, alters prediction probabilities, or re-ranks candidate ATMs. It visualizes the current case state deterministically.

---

## 2. Schema Specification (Map Input Contract)

```json
{
  "case_id": "CT-2026-8812",
  "model_id": "random_forest_full_none",
  "complaint_point": {
    "latitude": 31.3260,
    "longitude": 75.5762,
    "case_id": "CT-2026-8812",
    "city": "Jalandhar",
    "district": "Jalandhar",
    "state": "Punjab",
    "amount": 35000.0,
    "bank": "HDFC Bank",
    "date": "2026-08-14",
    "time": "14:30",
    "label": "Complaint Location: Jalandhar | Amount: INR 35,000.0 | Case: CT-2026-8812"
  },
  "predicted_zone": {
    "zone_id": "Zone_02",
    "probability": 99.5,
    "confidence_tier": "HIGH",
    "probability_margin": 0.99,
    "second_best_zone": "Zone_01",
    "bbox": {
      "min_lat": 31.21585,
      "min_lon": 75.44940,
      "max_lat": 31.43335,
      "max_lon": 75.73631
    },
    "color": "#00e5ff",
    "fill_opacity": 0.35
  },
  "candidate_zones": [
    {
      "zone_id": "Zone_02",
      "probability": 99.5,
      "is_primary": true,
      "bbox": {
        "min_lat": 31.21585,
        "min_lon": 75.44940,
        "max_lat": 31.43335,
        "max_lon": 75.73631
      },
      "color": "#00e5ff",
      "fill_opacity": 0.35
    }
  ],
  "zone_probabilities": {
    "Zone_01": 0.005,
    "Zone_02": 0.995,
    "Zone_03": 0.000,
    "Zone_04": 0.000,
    "Zone_05": 0.000,
    "Zone_06": 0.000,
    "Zone_07": 0.000,
    "Zone_08": 0.000,
    "Zone_09": 0.000,
    "Zone_10": 0.000
  },
  "ranked_atms": [
    {
      "rank": 1,
      "atm_id": "OSM-NODE-5094962456",
      "bank": "ICICI Bank",
      "operator": "ICICI",
      "city": "Jalandhar",
      "district": "Jalandhar",
      "state": "Punjab",
      "zone": "Zone_02",
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
      "explanation": "Assigned to Zone_02 (1.5 km from reference sector center).",
      "source": "OpenStreetMap",
      "latitude": 31.31659,
      "longitude": 75.58917,
      "is_top": true,
      "is_selected": false,
      "marker_color": "#10b981",
      "marker_radius": 10,
      "designation": "Highest-ranked candidate"
    }
  ],
  "total_atms_visible": 10,
  "total_atms_evaluated": 10,
  "cross_zone_search": false,
  "confidence_tier": "HIGH",
  "prediction_confidence": 0.995,
  "probability_margin": 0.99,
  "map_bounds": [
    [31.17585, 74.83800],
    [31.67342, 75.77631]
  ],
  "map_center": [31.42463, 75.30715],
  "map_zoom": 11,
  "osm_attribution": "&copy; <a href='https://www.openstreetmap.org/copyright' target='_blank'>OpenStreetMap</a> contributors"
}
```

---

## 3. Visual Semantics & Layer Hierarchy

| Map Element | Visual Treatment | Semantics & Role | Claim Boundary Rule |
| :--- | :--- | :--- | :--- |
| **Complaint Location** | 🔴 Solid red circle (`#ef4444`, radius 9px, white border) with pulsing aura | Reported location where the victim filed or initiated the complaint | Strictly labeled *"Complaint Location"*, never *"Withdrawal Point"* |
| **Primary Predicted Zone** | 🟦 Cyan dashed boundary (`#00e5ff`, weight 2.5), fill opacity $0.12 - 0.40$ proportional to probability | Model's highest-probability spatial sector | Strictly labeled *"Predicted Withdrawal Zone"*, never *"Confirmed Fraud Zone"* |
| **Cross-Zone Candidate Zone**| 🟨 Amber dashed boundary (`#f59e0b`, weight 2.0), fill opacity $0.10 - 0.25$ | Secondary sector evaluated when confidence is Medium ($0.15 \le \Delta < 0.30$) or Low ($\Delta < 0.15$) | Communicates spatial ambiguity across boundaries (e.g. NCR East/West) |
| **Highest-Ranked ATM (#1)** | 🟢 Emerald green marker (`#10b981`, radius 10px, bold white border) | Top-priority ATM candidate based on multi-criteria heuristic scoring | Labeled *"Highest-ranked candidate"*, never *"Actual ATM used"* |
| **Alternative Candidate ATMs** | 🟠 Amber markers (`#f59e0b`, radius 7px, dark border) | Runner-up candidate ATMs discovered in candidate sectors | Labeled *"Candidate #{rank}"* |
| **Selected ATM Marker** | 🟣 Vivid purple marker (`#a855f7`, radius 11px, white glow border) | Interactive candidate focused by user table selection | Auto-opens candidate score breakdown popup |

---

## 4. Map Control & Navigation API

1. **Auto-Centering & Bounds**: Map executes `map.fitBounds(map_bounds, { padding: [35, 35] })` dynamically upon case submission.
2. **Navigation Buttons**:
   - `⟲ Fit All`: Restores encompassing bounds covering complaint, zones, and all visible candidate ATMs.
   - `🚨 Complaint`: Pans and zooms (`flyTo`, zoom 13) to the complaint location marker and opens popup.
   - `⭐ Top ATM`: Pans and zooms (`flyTo`, zoom 14) to the #1 ranked candidate ATM marker and opens popup.
3. **Cross-Zone Banner**: Appears automatically in the top-right corner whenever `cross_zone_search == true`.
4. **Mandatory Attribution**: Bottom-right tile attribution: `Map data © OpenStreetMap contributors`.
