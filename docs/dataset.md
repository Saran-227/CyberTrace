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
- `atm_id`: Foreign key referencing ATM
- `date`: YYYY-MM-DD
- `hour`: Operational hour (0-23)
- `transaction_count`: Number of completed transactions
- `fraud_withdrawal_count`: Synthetic flag for anomalous cash-outs
- `average_amount`: Mean withdrawal value
- `high_value_withdrawal_count`: Frequency of maximum denomination withdrawals
