# CATIA

**Catastrophe AI System for Climate Risk Modeling**

CATIA is a production-ready Python library for catastrophe risk modeling. It combines climate data ingestion, ML-based risk prediction, actuarial loss simulation, and mitigation optimization into a unified framework.This system allows us to quantify climate risks and also find solutions in the leap of Environment Sustainable Goals (ESG).

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from catia.data_acquisition import fetch_climate_data
from catia.risk_prediction import train_risk_model
from catia.financial_impact import run_financial_impact_analysis
from catia.mitigation import generate_mitigation_recommendations

# Fetch climate data
climate_data = fetch_climate_data(use_mock=True)

# Train risk model
model = train_risk_model(climate_data)

# Run financial simulation
results = run_financial_impact_analysis(
    annual_frequency=2.5,
    mean_severity=50_000_000,
    n_simulations=10_000
)

# Get mitigation recommendations
recommendations = generate_mitigation_recommendations(
    expected_annual_loss=results['expected_loss'],
    budget=10_000_000 #This is an estimation based on the analysis, kindly feel free to adjust.
)
```

## Key Capabilities

| Module | Description |
|--------|-------------|
| `data_acquisition` | Climate data from NOAA, ECMWF; socioeconomic data from World Bank |
| `risk_prediction` | ML models for catastrophe probability and severity |
| `financial_impact` | Monte Carlo simulation with frequency-severity models |
| `extreme_value` | EVT/GPD tail modeling for 100-1000 year events |
| `uncertainty` | Bootstrap confidence intervals for all risk metrics |
| `correlation` | Copula-based multi-peril dependency modeling |
| `ensemble` | Voting and stacking ensembles for robust predictions |
| `explainability` | SHAP-based model interpretability |
| `backtesting` | Historical validation and model monitoring |
| `mitigation` | Budget-constrained optimization of risk reduction strategies |

## API & Production

Run the REST API with `catia --api --port 8000` or `uvicorn catia.api.app:app --reload`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Liveness probe |
| `GET /api/v1/ready` | Readiness (output dir, config) |
| `GET /api/v1/perils/` | List perils and config |
| `POST /api/v1/simulation/run` | Multi-peril Monte Carlo |
| `POST /api/v1/analysis/run` | Full analysis pipeline |
| `POST /api/v1/mitigation/optimize` | Mitigation recommendations |

All errors return a structured body with `error`, `message`, `request_id`, and `timestamp`. Send `X-Request-ID` for tracing. Every CLI/main run includes a **run ID** and **config snapshot** in the report for reproducibility and audit.

## Phase C (Best-in-Class)

- **Async jobs**: `POST /api/v1/analysis/jobs` to submit long runs; poll `GET /api/v1/analysis/jobs/{id}` and `GET /api/v1/analysis/jobs/{id}/result`.
- **Compliance report**: Every run writes `outputs/compliance_report.html` (CAS/SOA/NAIC alignment).
- **Uncertainty in pipeline**: Multi-peril analysis includes bootstrap uncertainty by default.
- **Ensemble**: Set `CATIA_USE_ENSEMBLE=1` to train a voting ensemble risk model.
- **User guide**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md). **Tutorial**: [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb).
- **Drought peril** and **optional DL** (MLP): Set `CATIA_USE_DL=1` or `model_type: NeuralNetwork` in config; add `drought` to perils.

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for the full plan. Phase A and B and C are complete.

## Running Tests

```bash
pytest tests/ -v
```

## Compliance

- CAS Catastrophe Modeling Guidelines
- SOA Risk Management Framework
- NAIC Model Act (insurance applications)

