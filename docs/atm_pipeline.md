# ATM Ingestion & Candidate Ranking Pipeline

## 1. OpenStreetMap Geographic Data Ingestion

The candidate ATM infrastructure is sourced entirely from OpenStreetMap (OSM) without relying on proprietary, paid mapping APIs (strictly no Google Maps or Mapbox tokens).

### Ingestion Source & Query Tags
- **Entity Filter**: `amenity=atm` or `amenity=bank[atm=yes]`
- **Spatial Coverage**: 15 target study cities matching the CyberTrace complaint dataset across Punjab, Chandigarh, Haryana, Delhi NCR, Uttar Pradesh, and Rajasthan.
- **Protocol**: Queries are dispatched to Overpass QL endpoints (`https://overpass-api.de/api/interpreter`, mirrors) with an automated direct OpenStreetMap API (`api.openstreetmap.org/api/0.6/map?bbox=...`) fallback to guarantee resilience against Overpass server throttling or timeouts.

### Attribution
All geographic ATM data is derived from OpenStreetMap:
> *Data © OpenStreetMap contributors, licensed under the Open Database License (ODbL).*

---

## 2. Deterministic Caching Architecture

To prevent redundant network requests, comply with OSM Fair Use policies, and guarantee 100% offline demonstration capability:

1. **Deterministic Cache Key**:
   - Cache keys are generated based on the normalized query target or bounding box:
     `osm_atms_{normalized_city_name}.json` or `osm_atms_{md5_hash}.json`.
2. **Cache Storage Directory**:
   - Stored locally in `data/external/osm/`.
3. **Lookup Precedence**:
   - Before any network request is initiated, the system checks for the presence of the corresponding JSON cache file.
   - If present, the raw response is loaded immediately from disk.
4. **Offline Resilience**:
   - When offline or if network requests fail, the pipeline falls back to existing cache files or returns a clean fallback without crashing.

---

## 3. ATM Output Schema & Normalization Rules

The normalized dataset is written to:
`data/processed/atm_locations.csv`

### Exact Core Fields
| Field | Type | Description / Normalization Rules |
| :--- | :--- | :--- |
| `atm_id` | String | Deterministic stable identifier: `osm_node_{id}` or `atm_{md5_hash}` |
| `bank` | String | Explicit brand name (e.g., `State Bank of India`, `HDFC Bank`). If missing: `"unknown"` |
| `operator` | String | Explicit operator entity (e.g., `Tata Indicash`, `Hitachi`). If missing: `"unknown"` |
| `latitude` | Float | WGS-84 latitude bounded in $[-90.0, 90.0]$ |
| `longitude` | Float | WGS-84 longitude bounded in $[-180.0, 180.0]$ |
| `address` | String | Street name / house number / landmark if available. If missing: `"unknown"` |
| `city` | String | Target municipality name (e.g., `Ludhiana`, `Jaipur`) |
| `district` | String | Administrative district name |
| `state` | String | Administrative state name |
| `is_24x7` | Boolean / String | `True` only when source explicitly verifies (e.g., `opening_hours=24/7`). Otherwise `"unknown"` |
| `source` | String | Explicitly `"OpenStreetMap"` |

### Data Integrity & Non-Inference Rules
- **Zero Hallucination / Inference**: If OSM does not explicitly provide bank or operator tags, they are set to `"unknown"`. The system **never** infers bank identity based on nearby bank branches, road names, or assumed brand popularity.
- **Strict Hours Validation**: `is_24x7` is set to `True` only when `opening_hours` explicitly specifies `24/7` or `round-the-clock`. Otherwise, it remains `"unknown"`.
- **Deduplication**: Geographic points sharing identical coordinates within ~11 meters ($0.0001^\circ$) or identical OSM node IDs are deduplicated, keeping the entry with the richest metadata.

---

## 4. Known OSM Limitations & Research Caveats

1. **Coverage Density Variations**: OSM coverage is crowdsourced. Tier-1 metros and commercial hubs (e.g., Chandigarh, Ludhiana, Jaipur) have significantly higher tagged ATM density than peripheral or rural sub-districts.
2. **Metadata Sparsity**: Many community contributors tag physical nodes as `amenity=atm` without recording the `operator` or `bank` brand tags. Preserving `"unknown"` ensures research integrity rather than introducing synthetic bias.
3. **Overpass Gateway Latency**: Public Overpass endpoints experience high load and intermittent HTTP 429/504 errors. CyberTrace handles this via local caching and direct API 0.6 fallbacks.

---

## 5. Distinction: Geospatial Data vs. Transaction Evidence

> [!IMPORTANT]
> **Geospatial Point of Interest $\neq$ Criminal Evidence**:
> An OpenStreetMap record indicates only that an automated teller machine physically existed at that geographic coordinate at the time of mapping.
> It provides **zero evidence** that a criminal cash withdrawal occurred at that terminal.
> Legally actionable evidence requires formal bank audit logs, core banking system switch transaction logs, and authorized CCTV surveillance footage.

---

## 6. Multi-Criteria Candidate Scoring Algorithm (Upcoming Phase)

The candidate score $S \in [0, 1]$ prioritizes ATMs based on four weighted factors:

$$S = 0.40 \cdot S_{\text{dist}} + 0.30 \cdot S_{\text{bank}} + 0.15 \cdot S_{\text{access}} + 0.15 \cdot S_{\text{activity}}$$

1. **Proximity Score ($S_{\text{dist}}$)**:
   $$S_{\text{dist}} = e^{-0.15 \cdot d}$$
   Where $d$ is the Haversine distance in kilometers from the reference complaint / zone centroid.
2. **Bank Compatibility ($S_{\text{bank}}$)**:
   - Exact or partial brand match: $1.0$
   - Inter-bank rail compatibility: $0.35$
3. **Operational Accessibility ($S_{\text{access}}$)**:
   - 24/7 ATM: $1.0$
   - Daytime-only ATM during night incident: $0.25$
4. **Historical Activity ($S_{\text{activity}}$)**:
   - Normalized synthetic historical cash-out frequency: $\min(1.0, \frac{\text{fraud\_count}}{10})$.

---

## 7. Phase 2B: Synthetic Historical ATM Activity Dataset

To support ATM candidate ranking without compromising proprietary banking privacy or violating financial regulations, CyberTrace includes a dedicated historical activity generation engine in `src/atm/activity.py`.

### Academic Disclaimer & Synthetic Integrity
> [!WARNING]
> **SYNTHETIC DATA NOTICE — NOT REAL BANKING TRANSACTION DATA**:
> All ATM activity metrics (transaction counts, cash withdrawal volumes, estimated cash volumes, and activity scores) are **100% synthetic**.
> They are generated algorithmically for academic modeling, prototyping, and candidate-ranking demonstration only.
> Real-world investigative confirmation **strictly requires** official bank switch transaction logs, core banking system (CBS) audit trails, and authorized CCTV surveillance footage.

### Data Sources & Zero Target Leakage
- **Input**: `data/processed/atm_locations.csv` (strictly the 333 verified OSM ATM locations).
- **Output**: `data/processed/atm_activity.csv` (and raw version at `data/raw/atm_activity_raw.csv`).
- **Panipat Coverage**: Panipat has 0 OSM ATM records in `atm_locations.csv` (`NO_OSM_ATMS_FOUND`). In accordance with strict data integrity rules, **no fake ATMs or activity records are fabricated for Panipat**.
- **Zero Target Leakage**: The generation pipeline does **NOT** use `withdrawal_zone`, complaint coordinates, or fraud cluster labels. Activity levels are conditioned exclusively on physical ATM properties (city commercial weighting, bank tier, 24/7 operating hours, diurnal curves, and a deterministic hash of the `atm_id`).

### Activity Dataset Schema
| Field | Type | Description |
| :--- | :--- | :--- |
| `atm_id` | String | Foreign key matching verified OSM ATM node ID |
| `date` | String | ISO date (`YYYY-MM-DD`), spanning 90 days |
| `hour` | Integer | Operational hour ($0 \dots 23$) |
| `day_of_week` | Integer | Day of week index ($0 = \text{Monday}, \dots, 6 = \text{Sunday}$) |
| `is_weekend` | Integer | Binary flag: $1$ if Saturday or Sunday, $0$ otherwise |
| `is_night` | Integer | Binary flag: $1$ between 22:00 and 05:00, $0$ otherwise |
| `transaction_count` | Integer | Total synthetic interactions (inquiries + withdrawals + PIN ops) |
| `cash_withdrawal_count` | Integer | Synthetic cash dispensing transactions ($\le \text{transaction\_count}$) |
| `estimated_cash_volume` | Float | Plausible simulated cash volume dispensed in INR ($\ge 0.0$) |
| `activity_score` | Float | Multi-metric normalized activity score bounded in $[0.0, 100.0]$ |
| `fraud_withdrawal_count` | Integer | Synthetic anomalous cash-out signal for academic ranking |
| `high_value_withdrawal_count` | Integer | Synthetic frequency of maximum-denomination cash withdrawals |
| `average_amount` | Float | Synthetic mean transaction ticket in INR |

### Stochastic Modeling & Reproducibility
- **Reproducibility**: Parameterized with a fixed seed (`RANDOM_SEED = 42`).
- **Mathematical Invariant**: $\text{cash\_withdrawal\_count} \sim \text{Binomial}(\text{transaction\_count}, p)$ where $p \sim \text{Beta}(8, 3)$, guaranteeing $0 \le \text{cash\_withdrawal\_count} \le \text{transaction\_count}$ always.
- **Volume Distribution**: Dispensed cash is drawn from a plausible Gamma distribution ($\mu \approx ₹3,120$) reflecting typical ATM cash dispense tickets in India.

