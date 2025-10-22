# CATIA Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd CATIA

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the System

```bash
# Run complete analysis with mock data
python main.py
```

**Expected Output:**
```
================================================================================
CATIA: Catastrophe AI System for Climate Risk Modeling
================================================================================
Analysis Region: US_Gulf_Coast
Timestamp: 2024-01-15T10:30:45.123456

[STEP 1] DATA ACQUISITION
✓ Climate data: 1461 records
✓ Socioeconomic data: 1 records
✓ Historical events: 12 records

[STEP 2] RISK PREDICTION MODEL
✓ Risk prediction model trained and saved

[STEP 3] FINANCIAL IMPACT SIMULATION
✓ Monte Carlo simulations: 10000 iterations
✓ Mean annual loss: $45,234,567
✓ VaR (95%): $78,901,234
✓ TVaR (95%): $92,345,678

[STEP 4] MITIGATION RECOMMENDATIONS
✓ Baseline loss: $45,234,567
✓ Mitigated loss: $27,140,740
✓ Risk reduction: 40.00%
✓ Priority strategies: relocation, infrastructure_hardening, land_use_planning

[STEP 5] VISUALIZATION & REPORTING
✓ Dashboard created: outputs/
  - loss_exceedance_curve.html
  - risk_distribution.html
  - return_period_curve.html
  - mitigation_comparison.html

================================================================================
ANALYSIS COMPLETE
================================================================================
```

### Step 3: View Results

**JSON Report:**
```bash
# Open the comprehensive report
cat outputs/catia_report.json
```

**Interactive Visualizations:**
```bash
# Open in web browser
open outputs/loss_exceedance_curve.html
open outputs/risk_distribution.html
open outputs/return_period_curve.html
open outputs/mitigation_comparison.html
```

## Understanding the Output

### Key Metrics Explained

**Mean Annual Loss (MAL):** $45.2M
- Average expected loss per year
- Used for budgeting and insurance pricing

**Value-at-Risk (VaR) at 95%:** $78.9M
- Loss level with 95% probability of not being exceeded
- Regulatory capital requirement

**Tail Value-at-Risk (TVaR) at 95%:** $92.3M
- Average loss when exceeding VaR
- More conservative risk measure

**100-Year Loss:** $150M+
- Loss level with 1% annual probability
- Used for long-term planning

### Visualizations

**Loss Exceedance Curve:**
- X-axis: Loss amount ($ millions)
- Y-axis: Probability of exceeding that loss
- Shows tail risk and extreme events

**Risk Distribution:**
- Histogram of simulated annual losses
- Shows most likely outcomes and tail behavior
- Red line = mean loss

**Return Period Curve:**
- X-axis: Return period (years, log scale)
- Y-axis: Loss level
- Shows increasing severity for rarer events

**Mitigation Comparison:**
- Left: Cost vs Risk Reduction scatter plot
- Right: Benefit-Cost Ratio for each strategy
- Helps prioritize investments

## Customization

### Change Analysis Region

Edit `main.py`:
```python
results = run_catia_analysis(
    region="California_Wildfire_Zone",  # Change region
    use_mock_data=True
)
```

### Adjust Simulation Parameters

Edit `config.py`:
```python
SIMULATION_CONFIG = {
    "monte_carlo_iterations": 50000,  # Increase for more accuracy
    "confidence_level": 0.99,  # Change to 99%
    # ... other parameters
}
```

### Modify Mitigation Budget

Edit `config.py`:
```python
MITIGATION_CONFIG = {
    "budget_constraint": 5_000_000,  # $5M budget
    # ... other parameters
}
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_financial_impact.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'plotly'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Permission denied" on macOS/Linux

**Solution:**
```bash
chmod +x main.py
python main.py
```

### Issue: Slow execution

**Solution:** Reduce Monte Carlo iterations in `config.py`:
```python
SIMULATION_CONFIG = {
    "monte_carlo_iterations": 1000,  # Reduced from 10000
}
```

## Next Steps

### 1. Integrate Real Data

Replace mock data with real APIs:

```python
# In main.py
results = run_catia_analysis(
    region="US_Gulf_Coast",
    use_mock_data=False  # Use real APIs
)
```

**Required:**
- NOAA API key: https://www.ncei.noaa.gov/
- ECMWF API key: https://www.ecmwf.int/
- World Bank API (free, no key needed)

### 2. Calibrate Models

Use your own historical data:

```python
from src.risk_prediction import train_risk_model

# Load your data
climate_data = pd.read_csv("your_climate_data.csv")
socioeconomic_data = pd.read_csv("your_socioeconomic_data.csv")
historical_events = pd.read_csv("your_historical_events.csv")

# Train model
predictor = train_risk_model(
    climate_data,
    socioeconomic_data,
    historical_events
)
```

### 3. Deploy to Production

```bash
# Containerize with Docker
docker build -t catia:latest .
docker run -p 8050:8050 catia:latest

# Deploy to AWS Lambda
aws lambda create-function --function-name catia \
  --runtime python3.11 --role <role-arn> \
  --handler main.run_catia_analysis
```

### 4. Set Up Monitoring

```python
# Add logging and alerts
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("catia.log"),
        logging.StreamHandler()
    ]
)
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, orchestrates workflow |
| `config.py` | Configuration settings |
| `src/data_acquisition.py` | Data fetching and validation |
| `src/risk_prediction.py` | ML model training and prediction |
| `src/financial_impact.py` | Monte Carlo simulation |
| `src/mitigation.py` | Optimization and recommendations |
| `src/visualization.py` | Interactive dashboards |
| `tests/` | Unit tests |
| `outputs/` | Generated reports and visualizations |

## Support

- **Documentation:** See README.md, IMPLEMENTATION_GUIDE.md, TECHNICAL_ARCHITECTURE.md
- **Issues:** Check troubleshooting section above
- **Questions:** Review docstrings in source code

## Example: Custom Analysis

```python
from src.data_acquisition import fetch_all_data
from src.risk_prediction import train_risk_model
from src.financial_impact import run_financial_impact_analysis
from src.mitigation import generate_mitigation_recommendations

# 1. Fetch data
data = fetch_all_data("US_Gulf_Coast", use_mock=True)

# 2. Train model
predictor = train_risk_model(
    data['climate'],
    data['socioeconomic'],
    data['historical_events']
)

# 3. Run financial analysis
financial_results = run_financial_impact_analysis(
    event_frequency=0.5,
    severity_params={'mu': 15, 'sigma': 2}
)

# 4. Generate recommendations
baseline_loss = financial_results['metrics']['descriptive_stats']['mean']
recommendations = generate_mitigation_recommendations(baseline_loss)

# 5. Print results
print(f"Mean Loss: ${baseline_loss:,.0f}")
print(f"VaR (95%): ${financial_results['metrics']['risk_metrics']['var']:,.0f}")
print(f"Top Strategy: {recommendations['priority_order'][0]}")
```

## Performance Tips

1. **Reduce iterations for testing:** Set `monte_carlo_iterations: 1000`
2. **Use mock data initially:** Faster than API calls
3. **Cache results:** Save intermediate results to disk
4. **Parallelize:** Use multiprocessing for large simulations
5. **Monitor memory:** Watch for large data structures

Enjoy using CATIA! 🌍📊

