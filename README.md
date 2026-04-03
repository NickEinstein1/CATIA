# CATIA

**Catastrophe AI for Climate Risk Modeling** — a production-grade platform that unifies data, ML, actuarial simulation, and mitigation optimization for multi-peril climate risk.

---

## Why CATIA

- **End-to-end** — Ingest climate data, train risk models, run Monte Carlo simulations, and optimize mitigation in one pipeline.
- **Actuarially rigorous** — Frequency–severity models, VaR/TVaR, return periods, EVT tail modeling, and copula-based multi-peril correlation.
- **Explainable & auditable** — Optional SHAP feature importance, compliance reports (CAS/SOA/NAIC), run IDs, and config snapshots.
- **Production-ready** — REST API with health checks, rate limiting, async jobs, structured errors, and observability.

---

## Installation

```bash
cd CATIA
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]" -r requirements.txt
```

---

## Quick Start

**Full pipeline (data → model → simulation → mitigation → report):**

```python
from catia.main import run_catia_analysis

results = run_catia_analysis(
    region="US_Gulf_Coast",
    use_mock_data=True,
    perils=["hurricane", "flood"]
)
# Outputs: report JSON, dashboards, compliance report, optional feature importance
```

**REST API:**

```bash
uvicorn catia.api.app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

**CLI:** `catia --api --port 8000`

**System dashboard (Dash):** futuristic command-center UI with an **orthographic globe**, an **OpenStreetMap** 2D map (`dash-leaflet`) with the same markers, charts, and assumptions:

```bash
catia --dashboard
# or: catia-dashboard
# http://127.0.0.1:8050 — use --dashboard-port to change port
```

---

## Capabilities

| Area | Features |
| ---- | -------- |
| **Data** | NOAA/ECMWF/World Bank connectors; cache; mock data for development |
| **Risk model** | Probability & severity models (RF, GB, optional MLP); ensemble (`CATIA_USE_ENSEMBLE=1`); model registry |
| **Simulation** | Multi-peril Monte Carlo; Lognormal, Pareto, Weibull, Gamma, spliced severity; parallel runs; VaR/TVaR, return periods |
| **Tail & uncertainty** | EVT/GPD; bootstrap confidence intervals; correlation (Gaussian/t/Clayton/Gumbel copulas) |
| **Explainability** | SHAP feature importance (`CATIA_USE_SHAP=1`); written to `outputs/feature_importance.json` |
| **Mitigation** | Budget-constrained optimization; cost–benefit analysis; priority strategies |
| **API** | Health/ready; rate limiting; async jobs; request IDs; structured errors |

---

## API Overview

| Endpoint | Description |
| -------- | ----------- |
| `GET /api/v1/health` | Liveness |
| `GET /api/v1/ready` | Readiness |
| `GET /api/v1/perils/` | List perils and config |
| `POST /api/v1/simulation/run` | Multi-peril Monte Carlo |
| `POST /api/v1/analysis/run` | Full analysis pipeline |
| `POST /api/v1/analysis/jobs` | Submit async job; poll `GET .../jobs/{id}` and `.../jobs/{id}/result` |
| `POST /api/v1/analysis/stress` | Solvency-II-style stress scenarios (baseline or quick sim + stressed metrics) |
| `POST /api/v1/mitigation/optimize` | Mitigation recommendations |

---

## Documentation

- **[User guide](docs/USER_GUIDE.md)** — Concepts and pipeline
- **[Tutorial](notebooks/tutorial.ipynb)** — Step-by-step notebook
- **[Roadmap](docs/ROADMAP.md)** — Strategy and phases
- **[CAT modeling next](docs/CAT_MODELING_NEXT.md)** — What’s next for best-in-class catastrophe modeling (exposure, vulnerability, event set, spatial)
- **[Runbook](docs/RUNBOOK.md)** — Run API, env vars, troubleshooting
- **[Research and citation](docs/RESEARCH_AND_CITATION.md)** — Cite CATIA, reproducibility, model cards
- **[Docs index](docs/INDEX.md)** — Full documentation map

---

## Tests

```bash
pytest tests/ -v --tb=short
```

---

## Compliance

Designed for alignment with **CAS** catastrophe modeling guidelines, **SOA** risk management frameworks, and **NAIC** model act requirements for insurance applications.
