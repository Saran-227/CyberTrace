# Interactive Leaflet / OpenStreetMap Geospatial Intelligence Map Pipeline (Phase 8)

## 1. Zero-Cost, Open-Source Mapping Principle
CyberTrace adheres strictly to free, open-source geospatial standards without reliance on commercial map SDKs:
- **Renderer**: Leaflet.js (v1.9.4)
- **Base Tile Provider**: OpenStreetMap Standard Carto Tiles (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`)
- **Attribution**: `&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors`
- **Candidate ATM Layer Data**: Sourced directly from `data/processed/atm_locations.csv` (333 verified OpenStreetMap ATM records ingested and cached locally)
- **Prohibited Technologies**: Google Maps API, Google Maps API keys, Mapbox tokens, or any paid map SDKs.

## 2. Dynamic Architecture & Data Flow
The map component functions as a pure visual intelligence consumer of the live Phase 7 pipeline (`src/intelligence/case_analysis.py`). It does not independently perform ML inference or rank ATM candidates.

```
USER INPUT (Complaint Form)
            ↓
  ANALYZE CASE TRIGGER
            ↓
  src/intelligence/case_analysis.py (analyze_case)
            ↓
  Case Analysis Result (st.session_state["case_analysis"])
            ↓
  src/geographic/map_data.py (prepare_map_data)
            ↓
  Standardized Map Contract (docs/map_contract.md)
            ↓
  app/components/map.py (render_investigation_map)
            ↓
  Leaflet.js + OpenStreetMap Tiles
```

When a user submits a new complaint (e.g. switching from Jalandhar to Gurugram), the previous map state is discarded, new bounds are computed, and Leaflet smoothly fits to the newly relevant operational area without leaving stale markers.

## 3. Visual Representation Hierarchy & Semantics

| Visual Element | Visual Style | Semantics & Evidence Language |
| :--- | :--- | :--- |
| **Complaint Location** | 🔴 Red circle marker (`#ef4444`, 9px radius, white outline, pulsing aura) | **"Complaint Location"** (Where victim reported the fraud incident; NOT confirmed cash-out point) |
| **Primary Predicted Zone** | 🟦 Cyan dashed boundary (`#00e5ff`, weight 2.5), probability-proportional fill opacity (0.10 to 0.40) | **"Predicted Withdrawal Zone"** (Statistical likelihood output from ML model; NOT confirmed area) |
| **Secondary Candidate Zone** | 🟨 Amber dashed boundary (`#f59e0b`, weight 2.0), fill opacity 0.12 | **"Secondary Candidate Zone"** (Evaluated under Medium/Low model confidence) |
| **Top Candidate ATM (#1)** | 🟢 Emerald green marker (`#10b981`, 10px radius, white outline, glowing green ring, star badge ⭐) | **"Highest-Ranked ATM Candidate (#1)"** (Top scored candidate by multi-criteria ranking engine) |
| **Alternative Candidate ATMs** | 🟠 Amber circle marker (`#f59e0b`, 7px radius, white outline) | **"Alternative ATM Candidate (#2-#N)"** (Secondary proximity & compatibility candidates) |
| **Selected ATM Candidate** | 🟣 Vivid purple marker (`#a855f7`, 11px radius, pulsing purple ring) | **"Selected Candidate"** (Active candidate currently focused in UI/table) |

## 4. Confidence-Driven Zone & Cross-Zone Visualization

The map dynamically adapts its boundary visualization based on the model confidence tier:
1. **HIGH Confidence ($\Delta \ge 0.30$)**:
   - Renders only the primary predicted zone boundary.
   - Fits bounds tightly around the complaint point and primary sector ATM candidates.
2. **MEDIUM Confidence ($0.15 \le \Delta < 0.30$) / Cross-Zone Search**:
   - Renders both the primary and secondary zone boundaries (e.g., Zone_07 ↔ Zone_08 NCR sector ambiguity).
   - Displays an overlay badge: `🔄 Cross-Zone Candidate Search Active`.
   - Bounds encompass both sectors and all relevant ATM candidates from both zones.
3. **LOW Confidence ($\Delta < 0.15$)**:
   - Renders all candidate zones evaluated by the Phase 7 engine.
   - Conveys multi-sector uncertainty to the investigator.

## 5. Rich Popups & Interactive Synchronizations

### Complaint Marker Popup
- Header: `COMPLAINT ORIGIN`
- Fields: Case ID, Reported City, Incident Date/Time, Reported Amount (INR)
- Disclaimer: `Reported cybercrime victim location. This is NOT a confirmed withdrawal point.`

### ATM Marker Popup
- Header: `ATM CANDIDATE #Rank`
- Fields: Bank Name, Operator, City / District, Overall Score (`XX.X / 100`), Distance (`X.XX km`), Withdrawal Zone
- Compatibility Chips: Bank Match (`EXACT MATCH` / `CROSS-NETWORK`), Operational Hours (`24x7 ACCESS` / `STANDARD`), Activity Score
- Disclaimer: `Ranked OpenStreetMap infrastructure candidate. Physical CCTV verification required.`

### Interactive Map Controls
- **`⟲ Fit All`**: Recalculates and smoothly flies to the full bounding box of complaint, zones, and ATMs.
- **`🚨 Complaint`**: Centers directly on the complaint origin point (zoom level 14).
- **`⭐ Top ATM`**: Centers directly on the highest-ranked ATM candidate (#1).

## 6. Edge Cases & Resilience
- **Missing Complaint Coordinates**: If complaint coordinates are missing or invalid (`NaN` / `0.0`), the map gracefully falls back to candidate zone centroids and displays ATM candidates without throwing an exception.
- **Panipat Edge Case (Zero OSM ATMs)**: Panipat has 0 OSM ATMs in the dataset. The map renders the adjacent search zone (`Zone_08`) and its candidate ATMs without crashing or fabricating ATM locations.
- **Offline Map Tile Handling**: If OSM Carto tiles fail to load or the user is offline, the container background displays an intelligence-themed fallback notice: `Map tiles are temporarily unavailable. Analytical results remain available.`
- **Filtering Controls**: Frontend controls (candidate count 5/10/25, bank filter, zone filter) filter the display without triggering re-inference or changing the underlying ranking calculations.

## 7. Evidence Integrity & Legal Disclaimers
Under no circumstances will CyberTrace label any map entity as:
- *"Fraud ATM"*
- *"Actual withdrawal location"*
- *"Confirmed criminal cash-out site"*

All markers represent open physical infrastructure prioritized according to statistical proximity, institutional compatibility, and simulated operational metrics.
