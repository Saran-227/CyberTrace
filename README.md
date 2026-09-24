# CYBERTRACE: Cybercrime Cash-Withdrawal Location Intelligence Platform

CyberTrace is a supervised machine-learning based cybercrime cash-withdrawal location intelligence platform developed as an academic machine learning research system.

The system ingests synthetic cybercrime complaint metadata, predicts a likely geographic cash-out zone, discovers candidate ATM locations in that area using free, open-source geographic data (OpenStreetMap), ranks candidate ATMs through multi-criteria spatial scoring, visualizes the intelligence on an interactive Leaflet map, and generates a non-technical executive intelligence report for investigators.

---

## ⚠️ Academic Disclaimers & Ethics Policy

1. **Synthetic Data Disclosure**: All cybercrime complaint records, transaction histories, and ATM activity logs are **synthetic**. ATM activity data is synthetic and represents simulated operational activity. It does not represent actual bank transaction logs or confirmed fraud activity. No actual victim data or personal identifiable information (PII) is used.
2. **Zero Target Leakage**: The supervised machine learning model does not access hidden cash-out coordinates or target labels during inference or feature engineering.
3. **No Fabrication of Evidence**: The system clearly distinguishes:
   - **ML Prediction**: Statistical likelihood of geographic withdrawal zones.
   - **Candidate Ranking**: Proximity and compatibility ranking of open ATM infrastructure.
   - **Geographic Information**: Map layers from OpenStreetMap.
   - **Actual Evidence**: Formal banking audit trails and physical CCTV footage (which can only be verified by authorized law enforcement).
4. **Official Redirection**: CyberTrace does not process official criminal complaints. Users must register cases directly with the **National Cyber Crime Reporting Portal** ([cybercrime.gov.in](https://www.cybercrime.gov.in/)) or contact the Citizen Financial Cyber Fraud Helpline **1930**.

---

## 🏛️ System Architecture

```
CyberTrace/
├── README.md                           # Comprehensive documentation and roadmap
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Version control exclusions
├── .env.example                        # Environment configuration template
│
├── data/
│   ├── raw/                            # Ingested raw datasets (with noise/dupes)
│   │   ├── cybercrime_complaints_raw.csv
│   │   ├── atm_locations_raw.csv
│   │   └── atm_activity_raw.csv
│   │
│   ├── processed/                      # Preprocessed, deduplicated datasets
│   │   ├── cybercrime_complaints.csv
│   │   ├── atm_locations.csv
│   │   └── atm_activity.csv
│   │
│   └── external/
│       └── osm/                        # Local OpenStreetMap / Overpass JSON cache
│
├── notebooks/                          # Sequential research and validation notebooks
│   ├── 01_dataset_generation.ipynb
│   ├── 02_dataset_validation_eda.ipynb
│   ├── 03_geographic_clustering.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_location_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_atm_candidate_ranking.ipynb
│   └── 08_final_demo.ipynb
│
├── src/                                # Core Python package
│   ├── config.py                       # Paths, schemas, features, constants
│   ├── data/                           # Loaders, validation, synthetic generators
│   ├── preprocessing/                  # Cleaning, feature engineering, pipeline
│   ├── geographic/                     # Distance (Haversine), clustering, zones
│   ├── models/                         # Classifiers, training, evaluation, inference
│   ├── atm/                            # Overpass ingestion, caching, candidate ranking
│   ├── intelligence/                   # Score combination, non-technical explanation, reporting
│   └── utils/                          # Logging, schema validation
│
├── models/
│   ├── location_classifier/            # Serialized ML pipelines (.joblib)
│   └── atm_ranker/                     # Ranker configurations and weights
│
├── reports/
│   ├── generated/                      # Generated executive briefs (HTML/PDF)
│   └── templates/                      # Jinja2 report templates
│
├── app/                                # Streamlit Intelligence Application
│   ├── app.py                          # Shell entrypoint and page routing
│   ├── pages/                          # Dashboard, Investigation, Analytics, Reports
│   ├── components/                     # Leaflet map, prediction cards, ATM tables
│   └── assets/                         # CSS styling, JavaScript, visual assets
│
├── docs/                               # Architectural and technical documentation
│   ├── architecture.md
│   ├── dataset.md
│   ├── ml_pipeline.md
│   ├── map_pipeline.md
│   ├── atm_pipeline.md
│   └── reporting.md
│
└── tests/                              # Pytest test suite
    ├── test_data.py
    ├── test_preprocessing.py
    ├── test_geography.py
    ├── test_location_model.py
    ├── test_atm_ranking.py
    └── test_reports.py
```

---

## 🛠️ Technology Stack

- **Machine Learning**: Python 3.10+, Pandas, NumPy, Scikit-learn, Scipy, Joblib, Matplotlib, Seaborn
- **Geospatial & Mapping**: Leaflet.js, OpenStreetMap Carto tiles, Overpass API (free, open source; strictly no Google Maps or paid APIs)
- **Local Caching Layer**: JSON disk cache with MD5 spatial hashing to prevent repeated API calls
- **Application Interface**: Streamlit with custom CSS cyber-intelligence dark theme
- **Reporting Engine**: Jinja2 HTML templates and ReportLab PDF support
- **Testing & Quality Assurance**: Pytest unit test coverage

---

## 🚀 Installation & Getting Started

### 1. Clone & Set Up Virtual Environment
```powershell
git clone <repository-url>
cd CyberTrace

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Pytest Verification
```powershell
python -m pytest tests/ -v
```

### 4. Launch Streamlit Intelligence Application
```powershell
streamlit run app/app.py
```
Open browser at `http://localhost:8501`.

---

## 📊 Dataset Schema

The primary modeling dataset contains ~20,000 synthetic complaint records with 19 attributes:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `complaint_id` | String | Unique synthetic incident identifier |
| `complaint_date` | String (YYYY-MM-DD) | Date of registered incident |
| `complaint_time` | String (HH:MM:SS) | Time of incident |
| `amount` | Float | Financial sum involved |
| `bank` | String | Complainant financial institution |
| `transaction_type` | String | Rail: UPI, IMPS, NEFT, ATM, etc. |
| `fraud_type` | String | Modus operandi classification |
| `city` | String | Originating city |
| `state` | String | Jurisdiction state |
| `district` | String | Originating district |
| `complaint_latitude` | Float | Incident coordinate latitude |
| `complaint_longitude` | Float | Incident coordinate longitude |
| `hour` | Integer | Incident hour (0-23) |
| `day_of_week` | Integer | Day index (0=Monday, 6=Sunday) |
| `day_name` | String | Day name |
| `is_weekend` | Integer | Binary weekend flag (0 or 1) |
| `is_night` | Integer | Binary night flag (0 or 1) |
| `amount_category` | String | Analytical tier: Low, Medium, High, Critical |
| **`withdrawal_zone`** | String | **Target Label (Class 01-10). Strictly excluded from input features.** |

---

## 🗺️ Machine Learning & Spatial Ranking Flow

```
Complaint Metadata (Amount, Bank, City, Time, Coordinates)
               │
               ▼
  [ Scikit-Learn Preprocessing ] (StandardScaler + OneHotEncoder)
               │  (Zero Target Leakage)
               ▼
  [ Supervised Location Model ] (Random Forest / Logistic Reg / KNN / Decision Tree)
               │
               ▼
   Withdrawal Zone Probabilities (e.g. Zone_04: 81.4%, Zone_03: 10.2%)
               │
               ▼
  [ Candidate ATM Discovery ] (OpenStreetMap Overpass Query & Local Cache)
               │
               ▼
  [ Multi-Criteria ATM Ranking ]
       ├── Proximity (Haversine Distance Decay)
       ├── Bank Compatibility Matching
       ├── Operational Hours (24x7 vs Night)
       └── Historical Synthetic Activity Signals
               │
               ▼
  [ Interactive Leaflet Map & Executive Brief ]
```

---

## 🧭 Development Roadmap

```
PHASE 1: Dataset generation and validation
         STATUS: COMPLETE (Complaints & Activity pipeline ready, Streamlit launched)

PHASE 2A: ATM geographic dataset pipeline
          STATUS: COMPLETE (OpenStreetMap ingestion, caching, normalization, and validation)

PHASE 2B / 2B.1: Synthetic historical ATM activity dataset & methodology audit
          STATUS: COMPLETE (719,280 synthetic records, 2026-06-26 to 2026-09-23, zero target leakage, zero fraud variables)

PHASE 3: Geographic analysis and zone validation
          STATUS: COMPLETE (Target spatial validation, 10 zones verified, 20 reports & 10 figures generated, 66 tests passing)

PHASE 4: ML preprocessing
          STATUS: COMPLETE (Production-grade pipeline, dual feature sets [Full: 80 enc, Geo-Blind: 41 enc], stratified split, zero leakage, 83 tests passing)

PHASE 5: Location classification models
          STATUS: COMPLETE (14 experiments, 4 model families, 5-fold CV, Full [F1: 0.966] vs Geo-Blind [F1: 0.069], class weighting evaluated, 104 tests passing)

PHASE 6: Model evaluation, error analysis & pipeline selection
          STATUS: COMPLETE (14 models evaluated, NCR boundary analysis, uncertainty diagnostics, primary [Random Forest] & fallback [Logistic Regression] selected, 118 tests passing)

PHASE 7: ATM candidate ranking & live dynamic pipeline
          STATUS: COMPLETE (333 OSM ATMs, 719k activity records, 6-component scoring, dynamic cross-zone search, live orchestrator, 142 tests passing)

PHASE 8: Leaflet/OpenStreetMap interactive map
          STATUS: NEXT

PHASE 9: Executive intelligence report

PHASE 10: Streamlit integration

PHASE 11: Testing and refinement

PHASE 12: Final presentation/demo
```
