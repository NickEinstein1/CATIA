# CATIA Implementation Guide

## Executive Summary

CATIA is a production-ready catastrophe AI system for climate risk modeling. This guide provides detailed implementation instructions for actuaries, risk analysts, and climate scientists.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CATIA System Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Data Acquisition │  │ Risk Prediction  │  │ Financial    │  │
│  │ (NOAA, ECMWF,   │→ │ (ML Model)       │→ │ Impact       │  │
│  │  World Bank)     │  │                  │  │ (Monte Carlo)│  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│           ↓                     ↓                     ↓           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Mitigation Recommendations & Optimization        │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │    Visualization & Reporting (Interactive Dashboards)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation & Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd CATIA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.py` to customize:
- API endpoints and credentials
- ML model hyperparameters
- Monte Carlo simulation parameters
- Mitigation budget constraints
- Output directories

### 3. Mock Data Mode

For development/testing without real API credentials:
```python
from config import set_mock_data_mode
set_mock_data_mode(True)  # Use mock data
```

## Real-World API Integration

### NOAA Integration

```python
# Register at: https://www.ncei.noaa.gov/
# Get API key from NOAA portal

from src.data_acquisition import DataAcquisition

da = DataAcquisition(use_mock_data=False)
climate_data = da.fetch_climate_data(
    region="US_Gulf_Coast",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

**Required Credentials:**
- NOAA API Key (free registration)
- Endpoint: `https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso`

### ECMWF Integration

```python
# Register at: https://www.ecmwf.int/
# Get API credentials from ECMWF portal

# Set environment variables:
# export ECMWF_API_KEY=<your-key>
# export ECMWF_API_URL=<your-url>
```

**Required Credentials:**
- ECMWF API Key (free registration)
- ECMWF API URL

### World Bank Integration

```python
# No credentials required - free public API
# Endpoint: https://api.worldbank.org/v2

socioeconomic_data = da.fetch_socioeconomic_data(region="US_Gulf_Coast")
```

## Running the System

### Quick Start

```bash
# Run complete analysis with mock data
python main.py

# Output: outputs/catia_report.json + visualizations
```

### Advanced Usage

```python
from main import run_catia_analysis

# Run analysis for specific region
results = run_catia_analysis(
    region="US_Gulf_Coast",
    use_mock_data=False  # Use real APIs
)

# Access results
print(f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}")
print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
```

## Actuarial Methodology

### 1. Risk Prediction Model

**Algorithm:** Random Forest Classifier + Regressor
- **Probability Model:** Predicts event occurrence (binary classification)
- **Severity Model:** Predicts loss magnitude (regression)
- **Validation:** Cross-validation with actuarial metrics (loss ratio accuracy)

**Features:**
- Climate variables: temperature, precipitation, wind speed, pressure, humidity
- Socioeconomic variables: population density, GDP, infrastructure index, poverty rate

### 2. Financial Impact Simulation

**Frequency Model:** Poisson distribution
- λ (lambda) = expected events per year
- Calibrated from historical event data

**Severity Model:** Lognormal distribution
- μ (mu) = mean of log-normal
- σ (sigma) = standard deviation of log-normal
- Calibrated from historical loss data

**Monte Carlo Process:**
1. Simulate number of events: N ~ Poisson(λ)
2. For each event, simulate loss: L ~ Lognormal(μ, σ)
3. Annual aggregate loss = Σ L
4. Repeat 10,000 iterations

### 3. Risk Metrics

**Value-at-Risk (VaR):**
- 95th percentile of loss distribution
- Interpretation: 95% probability loss won't exceed VaR

**Tail Value-at-Risk (TVaR):**
- Average loss exceeding VaR
- More conservative than VaR
- Captures tail risk

**Return Periods:**
- Loss level with 1/n probability of occurring in any year
- Example: 100-year loss = loss with 1% annual probability

### 4. Mitigation Optimization

**Objective:** Maximize risk reduction under budget constraint

**Constraints:**
- Total cost ≤ Budget
- Each strategy selected 0 or 1 times

**Optimization Method:** Linear Programming
- Maximize: Σ (risk_reduction_i × x_i)
- Subject to: Σ (cost_i × x_i) ≤ Budget

**Cost-Benefit Analysis:**
- NPV = Σ (annual_benefit / (1+r)^t) - implementation_cost
- Benefit-Cost Ratio = NPV / Cost
- Payback Period = Cost / Annual Benefit

## Compliance & Standards

### CAS Standards

CATIA adheres to Casualty Actuarial Society guidelines:
- **CAS Catastrophe Modeling Guidelines:** Transparent methodology, documented assumptions
- **CAS Actuarial Standards of Practice:** Professional standards for modeling

### SOA Standards

Alignment with Society of Actuaries:
- **SOA Risk Management Framework:** Comprehensive risk assessment
- **SOA Governance Standards:** Transparent decision-making

### Regulatory Compliance

- **NAIC Model Act:** Insurance regulatory compliance
- **Solvency II (EU):** Capital adequacy requirements
- **IAIS Standards:** International insurance standards

## Validation & Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_financial_impact.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Model Validation

**Backtesting:**
```python
# Compare predicted vs actual losses
from src.risk_prediction import RiskPredictor

predictor = RiskPredictor()
predictor.load_model()

# Validate on historical data
predictions = predictor.predict(X_test)
loss_ratio = predicted_losses.sum() / actual_losses.sum()
print(f"Loss Ratio: {loss_ratio:.4f}")  # Should be close to 1.0
```

**Stress Testing:**
```python
# Test model under extreme scenarios
extreme_climate = climate_data.quantile(0.99)
predictions = predictor.predict(extreme_climate)
```

## Deployment

### Local Deployment

```bash
# Run as standalone application
python main.py

# Run as web service (Dash)
python -m src.visualization
# Access at http://localhost:8050
```

### Cloud Deployment

**AWS:**
```bash
# Package as Lambda function
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda_function.zip . && cd ..
zip lambda_function.zip main.py src/

# Deploy to Lambda
aws lambda create-function --function-name catia \
  --runtime python3.11 --role <role-arn> \
  --handler main.run_catia_analysis --zip-file fileb://lambda_function.zip
```

**Google Cloud:**
```bash
# Deploy to Cloud Functions
gcloud functions deploy catia \
  --runtime python311 \
  --trigger-http \
  --entry-point run_catia_analysis
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## Performance Optimization

### Parallel Processing

```python
from multiprocessing import Pool

# Parallelize Monte Carlo simulations
with Pool(processes=4) as pool:
    results = pool.map(simulate_annual_losses, range(10000))
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_climate_data(region, date):
    # Cached results
    pass
```

## Troubleshooting

### Common Issues

**Issue:** API connection timeout
- **Solution:** Increase timeout in config.py, check network connectivity

**Issue:** Model training fails
- **Solution:** Verify data quality, check for missing values

**Issue:** Monte Carlo simulation slow
- **Solution:** Reduce iterations in config.py, use parallel processing

## Next Steps

1. **Integrate Real APIs:** Replace mock data with actual NOAA/ECMWF/World Bank APIs
2. **Calibrate Models:** Use historical data specific to your region
3. **Validate Results:** Backtest against historical events
4. **Deploy to Production:** Use cloud platform of choice
5. **Monitor Performance:** Set up alerts for model drift

## Support & Documentation

- **Technical Documentation:** See docstrings in each module
- **Configuration Guide:** See config.py comments
- **API Documentation:** See data_acquisition.py
- **Model Documentation:** See risk_prediction.py

## References

- CAS Catastrophe Modeling Guidelines
- SOA Risk Management Framework
- IPCC Climate Change Reports
- NAIC Insurance Regulatory Standards

