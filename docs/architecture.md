# CyberTrace System Architecture

## Overview
CyberTrace is structured as a modular, tiered intelligence platform with strict boundary separation between data access, feature preprocessing, machine learning inference, geospatial discovery, and user presentation.

```mermaid
graph TD
    A[Synthetic Complaint Data] --> B[Data Ingestion & Validation]
    B --> C[Preprocessing & Leakage Prevention]
    C --> D[Location Classification Models]
    D --> E[Predicted Cash-Out Zone]
    E --> F[OSM / Overpass ATM Discovery Layer]
    F --> G[Local Spatial Cache]
    F --> H[Multi-Criteria ATM Ranking Engine]
    H --> I[Leaflet.js Geospatial Visualization]
    H --> J[Executive Intelligence Report Generator]
    I --> K[Streamlit Intelligence UI]
    J --> K
```

## Layer Architecture

### 1. Data Layer (`src/data/`)
- `loader.py`: Safe ingestion abstraction with existence checking.
- `validator.py`: Comprehensive missingness, duplicate, and schema conformity profiling.
- `generator.py`: Synthetic benchmark generator with reproducible noise injection.

### 2. Preprocessing & Feature Engineering (`src/preprocessing/`)
- `cleaning.py`: Missing value handling, categorical normalization, numeric coordinate/amount validation, and deduplication with comprehensive auditing.
- `features.py`: Feature engineering including cyclical temporal transformations (`hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`) and monetary log transformation (`amount_log`).
- `pipeline.py`: Leakage-safe Skikit-Learn `ColumnTransformer` builder supporting dual feature sets (`FEATURE_SET_FULL` and `FEATURE_SET_GEOGRAPHIC_BLIND`), stratified train/test split, StratifiedKFold cross-validation preparation, and balanced class weight calculation. Detailed in [docs/preprocessing.md](file:///docs/preprocessing.md).

### 3. Machine Learning Layer (`src/models/`)
- `location_model.py`: Object-oriented wrapper around Scikit-Learn pipelines supporting pipeline loading, training, prediction, and probability estimation.
- `train.py`: Pipeline constructor, 5-fold Stratified Cross-Validation orchestrator, test evaluation engine, and backward-compatible model trainer.
- `evaluate.py`: Multi-metric evaluation (Accuracy, Balanced Accuracy, Macro/Weighted Precision, Recall, F1, Multiclass OVR ROC-AUC) and automated dual-panel confusion matrix visualization generator.
- `predict.py`: Inference engine returning standardized probability distributions and top-k zone candidate rankings. Detailed in [docs/model_training.md](file:///docs/model_training.md).
- **Phase 6 Model Evaluation & Selection**: Multi-criteria evaluation of 14 model experiments, error diagnostics, NCR boundary analysis, uncertainty profiling, and selection of Primary (`random_forest_full_none`) and Fallback (`logistic_full_none`) models. Formally specified in [docs/model_evaluation.md](file:///docs/model_evaluation.md) and connected downstream via [docs/location_prediction_contract.md](file:///docs/location_prediction_contract.md).


### 4. Geospatial & ATM Ranking Layer (`src/geographic/`, `src/atm/`)
- `distance.py`: Mathematical Haversine calculations.
- `zones.py`: Geographic bounding boxes and centroids for target operational sectors.
- `analysis.py`: Target validation, spatial separation, empirical centroid computation, bounding box overlap analysis, statistical association tests, and ATM coverage analytics (implemented in Phase 3).
- `osm_loader.py`: Overpass QL client with MD5 spatial caching on disk.
- `loader.py`: Validated, cached loader for 333 verified OpenStreetMap ATM locations and 719,280 hourly synthetic ATM operational activity records with zero target leakage.
- `ranking.py`: Phase 7 transparent multi-criteria ATM candidate scoring engine uniting zone likelihood, spatial proximity decay, bank matching, operational hours accessibility, simulated activity, and cash volume compatibility. Formally defined in [docs/atm_ranking.md](file:///docs/atm_ranking.md) and [docs/atm_ranking_contract.md](file:///docs/atm_ranking_contract.md).

> [!NOTE]
> **Synthetic Target Disclosure**: `withdrawal_zone` is a synthetic target created for academic supervised-learning experimentation. It does not represent confirmed NCRP withdrawal locations.


### 5. Intelligence & Presentation Layer (`src/intelligence/`, `app/`)
- `case_analysis.py`: Live dynamic orchestration service executing end-to-end case intake, feature preprocessing, model inference, candidate ranking, and explanation generation in real-time.
- `explanation.py`: Non-technical operational explanation and actionable investigative advice generator.
- `report.py`: HTML/PDF executive brief generator.
- `app/`: Multi-page Streamlit application workbench with live dynamic case analysis and embedded Leaflet.js maps.
