# Interactive Geospatial & Mapping Pipeline

## 1. Zero-Cost, Open-Source Mapping Principle
CyberTrace adheres strictly to free and open-source geospatial standards:
- **Renderer**: Leaflet.js (v1.9.4)
- **Base Tile Provider**: OpenStreetMap Carto tiles (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`)
- **Attribution**: `&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors`
- **Candidate ATM Layer Data**: Sourced directly from `data/processed/atm_locations.csv` (Phase 2A OpenStreetMap ingestion)
- **Prohibited**: Google Maps API, Mapbox commercial tokens, or any paid mapping endpoints.

## 2. Visual Representation Hierarchy
The map visually conveys distinct spatial layers:
1. **Complaint Origin**: High-contrast red circle marker indicating where the victim reported the incident.
2. **Predicted Withdrawal Zone**: Bounding polygon (cyan border with semi-transparent blue fill) indicating the statistical perimeter predicted by the ML model.
3. **Candidate ATM Points**:
   - Ingested from `data/processed/atm_locations.csv`.
   - Filtered spatially by bounding box or Haversine radius from the predicted zone centroid.
   - **Highest-Ranked Candidate**: Emerald green marker (`#10b981`) with prominent icon.
   - **Alternative Candidates**: Amber markers (`#f59e0b`).
4. **Interactive Popups**: Display bank brand, operator, exact distance (km) from origin, and candidate compatibility score.

## 3. Streamlit Embedding Architecture
The map is embedded using `streamlit.components.v1.html`, dynamically injecting Leaflet scripts, responsive CSS styling, and GeoJSON/JSON data objects without requiring round-trip API tokens or exposing server secrets.

## 4. Evidence Integrity Notice
Markers on the Leaflet map denote candidate infrastructure discovered from OpenStreetMap. They represent geographic proximity candidates, not confirmed transaction locations.

---

## 5. Phase 7 Integration: Dynamic Map Input Contract

The interactive map in Phase 8 will consume the standardized output produced by `src/intelligence/case_analysis.py`:
- `complaint_point`: `(complaint_latitude, complaint_longitude, popup_label)`
- `predicted_zone_bbox`: Bounding box tuple `(min_lat, min_lon, max_lat, max_lon)` for the predicted zone.
- `candidate_zones`: List of all evaluated sectors (including cross-boundary candidate zones).
- `candidate_atms`: Top ranked candidate ATM objects containing `latitude`, `longitude`, `bank`, `operator`, `overall_score`, `rank`, `designation`, and `evidence_flags`.
- `cross_zone_status`: Flag indicating whether dual-sector rendering is active.


