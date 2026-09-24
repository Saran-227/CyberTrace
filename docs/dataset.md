# Dataset Architecture & Governance

## 1. Synthetic Data Philosophy
CyberTrace operates exclusively on synthetic data to preserve privacy, comply with academic data ethics, and prevent real victim disclosure.
All records are generated with realistic parameters reflecting digital banking crime profiles in North India.

## 2. Directory Separation
- `data/raw/`: Preserves untouched raw records with injected duplicates and missingness.
- `data/processed/`: Standardized, cleaned, and imputed datasets ready for model ingestion.
- `data/external/osm/`: Local cache directory holding JSON responses from OpenStreetMap Overpass queries.

## 3. Schemas

### A. Complaint Dataset (`cybercrime_complaints.csv`)
- `complaint_id`: Unique identifier
- `complaint_date`: YYYY-MM-DD
- `complaint_time`: HH:MM:SS
- `amount`: Transaction value (INR)
- `bank`: Complainant bank
- `transaction_type`: Payment rail (UPI, IMPS, NEFT, ATM, etc.)
- `fraud_type`: Modus operandi classification
- `city`: Reporting city
- `state`: Reporting state
- `district`: Administrative district
- `complaint_latitude`: Latitude of complaint incident
- `complaint_longitude`: Longitude of complaint incident
- `hour`: Hour of transaction (0-23)
- `day_of_week`: Day index (0-6)
- `day_name`: Day of week name
- `is_weekend`: 1 if weekend, 0 otherwise
- `is_night`: 1 if between 22:00 and 05:00
- `amount_category`: Low, Medium, High, Critical
- `withdrawal_zone`: **Target Variable** (Zone_01 through Zone_10)

### B. ATM Locations Dataset (`atm_locations.csv`)
The primary candidate ATM dataset created in **Phase 2A** by querying OpenStreetMap (`amenity=atm`) across the 15 study cities and normalizing tags with strict fidelity:
- `atm_id`: Deterministic stable identifier (`osm_node_{id}` or `atm_{md5}`)
- `bank`: Associated bank or brand (`"unknown"` if tag missing)
- `operator`: Financial institution or white-label operator (`"unknown"` if tag missing)
- `latitude`: WGS-84 latitude bounded in $[-90.0, 90.0]$
- `longitude`: WGS-84 longitude bounded in $[-180.0, 180.0]$
- `address`: Physical road or locality description (`"unknown"` if missing)
- `city`: Reporting city name
- `district`: Administrative district
- `state`: Administrative state
- `is_24x7`: Availability status (`True` if explicitly verified in source, otherwise `"unknown"`)
- `source`: `"OpenStreetMap"`

**Data Rules & Integrity:**
- **No Hallucination**: Bank and operator values are never inferred or guessed. Missing values are preserved as `"unknown"`.
- **Deduplication**: Spatial duplicates within $0.0001^\circ$ (~11 meters) and duplicate OSM node IDs are deduplicated.
- **Local Caching**: Raw query responses are cached deterministically in `data/external/osm/osm_atms_{city}.json` to allow offline execution and prevent rate-limiting.

### C. ATM Historical Activity Dataset (`atm_activity.csv`)
Generated in **Phase 2B & audited in Phase 2B.1** to simulate historical operational metrics across all 333 verified OSM ATM locations over a completed 90-day time window (`2026-06-26` through `2026-09-23`).

> [!WARNING]
> **Synthetic Operational Activity Notice**: ATM activity data is synthetic and represents simulated operational activity. It does not represent actual bank transaction logs or confirmed fraud activity. Real investigations require authorized bank CBS audit logs and switch journals.

**Schema Fields:**
- `atm_id`: Deterministic foreign key referencing `atm_locations.csv`
- `date`: Completed historical transaction date (`2026-06-26` to `2026-09-23`)
- `hour`: Operational hour of day ($0 \dots 23$)
- `day_of_week`: Day index ($0 = \text{Monday} \dots 6 = \text{Sunday}$)
- `is_weekend`: Flag ($1$ for Saturday/Sunday, $0$ for weekdays)
- `is_night`: Flag ($1$ between 22:00 and 05:00, $0$ otherwise)
- `transaction_count`: Simulated hourly total interactions (inquiries, balance checks, withdrawals)
- `cash_withdrawal_count`: Simulated cash dispensing operations ($\le \text{transaction\_count}$)
- `estimated_cash_volume`: Estimated total dispensed cash volume in INR ($\ge 0.0$)
- `activity_score`: Composite normalized activity metric bounded in $[0.0, 100.0]$
- `high_value_withdrawal_count`: Frequency of simulated high-value withdrawals ($\ge ₹15,000$, calculated purely from synthetic transaction amounts)
- `average_amount`: Simulated average transaction ticket size in INR

**Validation & Leakage Safeguards:**
- **No Fraud Variable**: `fraud_withdrawal_count` is absent. CyberTrace does not possess real fraud transaction logs.
- **Zero Target Leakage**: `withdrawal_zone` and complaint target variables are strictly omitted.
- **Panipat Status**: Panipat has 0 OSM ATMs in `atm_locations.csv`; thus, exactly 0 activity records are generated for Panipat (no fake ATMs fabricated).
- **Invariant Guarantee**: `cash_withdrawal_count <= transaction_count` holds strictly across all 719,280 generated records.

---

## 4. Phase 3 Geographic Target Validation & Statistical Intelligence

In **Phase 3**, the `withdrawal_zone` target variable was subjected to complete geographic and statistical validation before initiating ML preprocessing:

> [!IMPORTANT]
> **Academic Target Disclaimer**:
> `withdrawal_zone` is a synthetic target created for academic supervised-learning experimentation. It does not represent confirmed NCRP withdrawal locations.

- **Target Completeness**: Exactly 10 withdrawal zones (`Zone_01` to `Zone_10`) across 20,000 complaints.
- **Spatial Separation**: Mean pairwise centroid distance is **205.8 km** (max 532.5 km between Amritsar and Jaipur, min 21.1 km between East and West NCR).
- **Class Imbalance**: Imbalance ratio of **23.43:1** (`Zone_07` has 5,436 records [27.18%], while `Zone_09` has 232 records [1.16%]). Phase 4 must employ `StratifiedKFold` and class weighting.
- **Statistical Associations**:
  - `city` is the primary spatial predictor (Cramér's $V = 0.9732$, $p < 10^{-15}$).
  - Non-spatial features (`bank`, `transaction_type`, `amount`, `hour`, `fraud_type`) show negligible effect sizes ($V < 0.05$, $\eta^2 < 0.001$), confirming zero artificial synthetic leakage.
- **Hidden Feature Exclusion**: `synthetic_cashout_latitude` and `synthetic_cashout_longitude` are strictly verified to be excluded from processed data.



