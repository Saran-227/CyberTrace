# CyberTrace Frontend Architecture & Workbench Integration (Phase 10)

## 1. Overview & Visual Identity

The CyberTrace frontend transforms the analytical ML and geospatial pipelines (Phases 1–9) into a cohesive, production-grade intelligence workbench. 

### Visual Identity & Inspiration
The UI explicitly breaks away from traditional "black hacker dashboards", matrix rain, and neon glow effects. Instead, it embodies:
- **Calm, Apple-level simplicity**: Expansive whitespace, clean typography, soft neutral backgrounds (`#F8FAFC`), near-black navy text (`#0F172A`), and restrained cyan/blue accents (`#0284C7`).
- **Floating pill navigation (Reference Style A - Haven)**: Floating rounded navigation bar with soft blur backdrop, real-time system readiness indicator, and navigation pills.
- **Geospatial network aesthetics (Reference Style B)**: Abstract SVG geospatial network visualization on the landing hero without fabricating fake case pins.
- **Modern SaaS card hierarchy (Reference Style C)**: Large border radius (`16px-20px`), soft depth shadows (`0 4px 20px -2px rgba(0, 0, 0, 0.05)`), and clean metric tags.

---

## 2. Global State Architecture

The application operates from a single authoritative source of truth stored in `st.session_state`:

```
                 USER COMPLAINT INPUT
                          │
                          ▼
                    analyze_case()
                          │
                          ▼
            st.session_state["case_analysis"]
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
  Prediction Card   Interactive Map    Candidate Table
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ▼
                  Reports Workbench
                          │
                     HTML & PDF
```

### Authoritative Keys:
- `st.session_state["case_analysis"]`: The master analytical result dictionary returned by `src.intelligence.case_analysis:analyze_case()`.
- `st.session_state["current_page"]`: Active page selection (`Overview`, `Investigate`, `Analytics`, `Reports`).
- `st.session_state["selected_atm_id"]`: Synchronized ATM candidate selection for interactive map highlighting.
- `st.session_state["report_status"]`: Status of the generated brief (`NOT_GENERATED`, `READY`, or `OUTDATED`).
- `st.session_state["current_report_path"]`: Path to the currently valid HTML brief.

### Cross-Page Persistence & Invalidation
- When a user analyzes a case in `Investigate`, the result persists across `Overview`, `Analytics`, and `Reports` without re-running models.
- When the user modifies inputs and triggers a new analysis, previous reports and selected ATM states are marked `OUTDATED`, guaranteeing zero stale data.

---

## 3. Component Architecture

```
app/
├── app.py                      # Main entrypoint, CSS injection, router, sidebar
├── assets/
│   └── css/
│       └── style.css           # Design tokens, typography, floating pills, cards
├── components/
│   ├── navbar.py               # Floating pill navbar with engine readiness
│   ├── atm_card.py             # Dedicated #1 candidate spotlight card & score bars
│   ├── atm_table.py            # Candidate ranking table with polymorphic evidence flags
│   ├── probability_chart.py    # Altair horizontal probability distribution bar chart
│   ├── prediction_card.py      # Primary sector prediction and confidence tier
│   └── map.py                  # Dynamic Leaflet/OSM map and neutral empty-state map
└── pages/
    ├── dashboard.py            # Overview, abstract hero visual, project stats, methodology
    ├── investigation.py        # Core workbench: case form, status bar, map, candidate table
    ├── analytics.py            # Model evaluation benchmarks, class imbalance, OSM coverage
    └── report.py               # Executive intelligence brief generator (HTML/PDF)
```

---

## 4. Performance & Caching Strategy

The frontend enforces strict performance protections to prevent lag and repeated compute:

1. **Model Loading (`@st.cache_resource`)**:
   - Machine learning pipelines (`random_forest_full_none`, `logistic_full_none`) are loaded once from disk and cached in memory. Never reloaded on widget reruns.
2. **Immutable Datasets (`@st.cache_data`)**:
   - The 333 OSM ATM records and zone centroids are cached on first load.
   - The 719,280 synthetic activity rows are aggregated once into hourly profiles and cached. The raw rows are never transferred to the frontend UI.
3. **Analytics Benchmarks (`@st.cache_data`)**:
   - Phase 6 evaluation benchmarks (`phase6_model_evaluation.csv`), class imbalance summaries, and ATM coverage tables are loaded from precomputed artifacts.
4. **Map Marker Optimization**:
   - The Leaflet map renders only the top $N$ candidates (default 10) requested by the user, avoiding heavy DOM rendering.

---

## 5. Map Integration & Edge Cases

- **OpenStreetMap & Leaflet.js**: Pure open-source zero-cost geospatial rendering via `streamlit.components.v1.html`. No Google Maps API keys or third-party paid tiles.
- **Empty State**: Renders a neutral, beautiful map centered over Northern India with a floating prompt banner before an investigation is initiated.
- **Panipat Edge Case**: Panipat has 0 verified OSM ATMs. The ranking engine searches neighboring candidate zones without crashing. The UI clearly displays candidates as analytical alternatives.
- **Unknown Bank Operators**: Displayed respectfully as `"Bank information unavailable"`, strictly avoiding speculative or negative inferences.

---

## 6. Analytical & Evidence Language Boundaries

In compliance with academic intelligence ethics:
- Candidates are designated as **"Highest-ranked ATM candidate"**, **"Predicted withdrawal zone"**, or **"Model-supported location"**.
- The terms **"Confirmed ATM"**, **"Criminal ATM"**, and **"Actual withdrawal location"** are strictly forbidden.
- The footer clearly provides the synthetic data disclosure and redirects citizens to the official National Cyber Crime Reporting Portal ([cybercrime.gov.in](https://www.cybercrime.gov.in/)) and Helpline **1930**.
