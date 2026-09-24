# CyberTrace Downstream Prediction Contract: ML to Geospatial ATM Ranking

## Contract Specification (Phase 6 to Phase 7 Interface)

Every inference produced by the selected `LocationClassifier` must output a standardized Python dictionary conforming to this schema:

```json
{
  "complaint_id": "CT012660",
  "predicted_zone": "Zone_01",
  "prediction_confidence": 0.9999,
  "second_best_zone": "Zone_03",
  "probability_margin": 0.9998,
  "model_id": "random_forest_full_none",
  "confidence_tier": "HIGH",
  "zone_probabilities": {
    "Zone_01": 0.9999,
    "Zone_02": 0.0000,
    "Zone_03": 0.0001,
    "Zone_04": 0.0000,
    "Zone_05": 0.0000,
    "Zone_06": 0.0000,
    "Zone_07": 0.0000,
    "Zone_08": 0.0000,
    "Zone_09": 0.0000,
    "Zone_10": 0.0000
  }
}
```

---

## Schema Contract Requirements

1. **`complaint_id`**: String matching source complaint.
2. **`predicted_zone`**: String $\in \{	ext{Zone\_01} \dots 	ext{Zone\_10}\}$, exactly matching the maximum probability class.
3. **`prediction_confidence`**: Float bounded in $[0.0, 1.0]$.
4. **`second_best_zone`**: String designating the runner-up candidate zone.
5. **`probability_margin`**: $	ext{prediction\_confidence} - P(	ext{second\_best\_zone})$.
6. **`zone_probabilities`**: Strict dictionary mapping all 10 canonical withdrawal zones to non-negative floats summing to $1.0 \pm 10^{-4}$.
7. **Downstream Consumption**: Phase 7 ATM ranker uses `zone_probabilities` to weight candidate ATM proximity scores across sector boundaries.
