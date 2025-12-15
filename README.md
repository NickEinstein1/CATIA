# CATIA

**Catastrophe AI System for Climate Risk Modeling**

CATIA is a production-ready Python library for catastrophe risk modeling. It combines climate data ingestion, ML-based risk prediction, actuarial loss simulation, and mitigation optimization into a unified framework.

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
    budget=10_000_000
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

## Running Tests

```bash
pytest tests/ -v
```

## Compliance

- CAS Catastrophe Modeling Guidelines
- SOA Risk Management Framework
- NAIC Model Act (insurance applications)

