# CATIA API Reference

## Data Acquisition Module

### DataAcquisition Class

```python
from src.data_acquisition import DataAcquisition

# Initialize
da = DataAcquisition(use_mock_data=True)
```

#### Methods

**fetch_climate_data(region, start_date, end_date)**
```python
df = da.fetch_climate_data(
    region="US_Gulf_Coast",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
# Returns: DataFrame with columns:
# - date, temperature, precipitation, wind_speed, 
#   sea_level_pressure, humidity, region
```

**fetch_socioeconomic_data(region)**
```python
df = da.fetch_socioeconomic_data(region="US_Gulf_Coast")
# Returns: DataFrame with columns:
# - region, population_density, gdp_per_capita,
#   infrastructure_index, poverty_rate
```

**fetch_historical_events(region, event_type)**
```python
df = da.fetch_historical_events(
    region="US_Gulf_Coast",
    event_type="hurricane"
)
# Returns: DataFrame with columns:
# - year, event_type, region, magnitude, loss_usd, affected_population
```

**validate_data(df, data_type)**
```python
cleaned_df, report = da.validate_data(df, data_type="climate")
# Returns: (cleaned_df, validation_report_dict)
```

### Helper Functions

**fetch_all_data(region, use_mock)**
```python
from src.data_acquisition import fetch_all_data

data = fetch_all_data("US_Gulf_Coast", use_mock=True)
# Returns: Dict with keys: 'climate', 'socioeconomic', 'historical_events'
```

---

## Risk Prediction Module

### RiskPredictor Class

```python
from src.risk_prediction import RiskPredictor

predictor = RiskPredictor()
```

#### Methods

**prepare_features(climate_data, socioeconomic_data, historical_events)**
```python
X, y_prob, y_sev = predictor.prepare_features(
    climate_data=climate_df,
    socioeconomic_data=socio_df,
    historical_events=events_df
)
# Returns: (features_df, probability_target, severity_target)
```

**train(X, y_probability, y_severity)**
```python
predictor.train(X, y_prob, y_sev)
# Trains both probability and severity models
# Performs cross-validation and validation
```

**predict(X)**
```python
probabilities, severities = predictor.predict(X)
# Returns: (probability_array, severity_array)
# probabilities: ∈ [0, 1]
# severities: Loss amounts in USD
```

**save_model(path)**
```python
predictor.save_model("models/risk_model.pkl")
# Saves trained models to disk
```

**load_model(path)**
```python
predictor.load_model("models/risk_model.pkl")
# Loads trained models from disk
```

### Helper Functions

**train_risk_model(climate_data, socioeconomic_data, historical_events)**
```python
from src.risk_prediction import train_risk_model

predictor = train_risk_model(climate_df, socio_df, events_df)
# Returns: Trained RiskPredictor instance
```

---

## Financial Impact Module

### FinancialImpactSimulator Class

```python
from src.financial_impact import FinancialImpactSimulator

simulator = FinancialImpactSimulator(
    event_frequency=0.5,
    severity_params={'mu': 15, 'sigma': 2}
)
```

#### Methods

**simulate_annual_losses(num_years)**
```python
losses = simulator.simulate_annual_losses(num_years=100)
# Returns: Array of annual aggregate losses
# Shape: (num_years,)
```

**monte_carlo_simulation()**
```python
results = simulator.monte_carlo_simulation()
# Returns: Dict with keys:
# - all_losses: Array of simulated losses
# - mean_loss, median_loss, std_loss
# - min_loss, max_loss
```

**calculate_var_tvar(losses)**
```python
metrics = simulator.calculate_var_tvar(losses)
# Returns: Dict with keys:
# - var: Value-at-Risk at 95%
# - tvar: Tail Value-at-Risk at 95%
# - var_confidence: 0.95
# - tail_losses_count: Number of losses ≥ VaR
```

**calculate_loss_exceedance_curve(losses)**
```python
loss_levels, exceedance_probs = simulator.calculate_loss_exceedance_curve(losses)
# Returns: (sorted_losses, exceedance_probabilities)
# exceedance_probs: P(Loss ≥ loss_level)
```

**calculate_return_periods(losses)**
```python
rp = simulator.calculate_return_periods(losses)
# Returns: Dict with keys:
# - '10_year': Loss level for 10-year event
# - '100_year': Loss level for 100-year event
# - '1000_year': Loss level for 1000-year event
# etc.
```

**calculate_aggregate_metrics(losses)**
```python
metrics = simulator.calculate_aggregate_metrics(losses)
# Returns: Dict with keys:
# - descriptive_stats: mean, median, std, skewness, kurtosis
# - risk_metrics: var, tvar, var_confidence
# - return_periods: 10-year, 100-year, etc.
# - percentiles: 50th, 75th, 90th, 95th, 99th
```

### Helper Functions

**run_financial_impact_analysis(event_frequency, severity_params)**
```python
from src.financial_impact import run_financial_impact_analysis

results = run_financial_impact_analysis(
    event_frequency=0.5,
    severity_params={'mu': 15, 'sigma': 2}
)
# Returns: Dict with keys:
# - simulation_results
# - metrics
# - loss_exceedance_curve
```

---

## Mitigation Module

### MitigationStrategy Class

```python
from src.mitigation import MitigationStrategy

strategy = MitigationStrategy(
    name="Infrastructure Hardening",
    cost=500_000,
    risk_reduction=0.25,
    implementation_time=2,
    effectiveness=0.85
)
```

### MitigationOptimizer Class

```python
from src.mitigation import MitigationOptimizer

optimizer = MitigationOptimizer(budget=1_000_000)
```

#### Methods

**add_strategy(strategy)**
```python
optimizer.add_strategy(strategy)
# Adds a MitigationStrategy to the optimizer
```

**add_strategies_from_dict(strategies_dict)**
```python
strategies = {
    'infrastructure_hardening': {
        'cost': 500_000,
        'risk_reduction': 0.25,
        'implementation_time': 2,
        'effectiveness': 0.85
    },
    # ... more strategies
}
optimizer.add_strategies_from_dict(strategies)
```

**optimize_greedy()**
```python
selected = optimizer.optimize_greedy()
# Returns: List of selected MitigationStrategy objects
# Uses greedy algorithm (best cost-benefit ratio first)
```

**optimize_linear_programming()**
```python
selected = optimizer.optimize_linear_programming()
# Returns: List of selected MitigationStrategy objects
# Uses linear programming (optimal solution)
```

**calculate_cost_benefit_analysis(baseline_loss)**
```python
cba_df = optimizer.calculate_cost_benefit_analysis(baseline_loss=50_000_000)
# Returns: DataFrame with columns:
# - Strategy, Cost, Annual_Benefit, NPV
# - Benefit_Cost_Ratio, Payback_Period_Years
# - Risk_Reduction, Effectiveness
```

**generate_recommendations(baseline_loss)**
```python
recommendations = optimizer.generate_recommendations(baseline_loss=50_000_000)
# Returns: Dict with keys:
# - summary: total_budget, total_cost, total_risk_reduction, etc.
# - strategies: List of strategy details
# - priority_order: Strategies sorted by benefit-cost ratio
```

### Helper Functions

**generate_mitigation_recommendations(baseline_loss, budget)**
```python
from src.mitigation import generate_mitigation_recommendations

recommendations = generate_mitigation_recommendations(
    baseline_loss=50_000_000,
    budget=1_000_000
)
# Returns: Dict with recommendations
```

---

## Visualization Module

### CATIAVisualizer Class

```python
from src.visualization import CATIAVisualizer

visualizer = CATIAVisualizer(output_dir="outputs/")
```

#### Methods

**plot_loss_exceedance_curve(loss_levels, exceedance_probs, var_95, tvar_95)**
```python
fig = visualizer.plot_loss_exceedance_curve(
    loss_levels=loss_array,
    exceedance_probs=prob_array,
    var_95=78_900_000,
    tvar_95=92_345_678
)
# Returns: Plotly Figure object
```

**plot_risk_distribution(losses)**
```python
fig = visualizer.plot_risk_distribution(losses=loss_array)
# Returns: Plotly Figure object (histogram)
```

**plot_return_period_curve(return_periods)**
```python
fig = visualizer.plot_return_period_curve(
    return_periods={'10_year': 50M, '100_year': 150M, ...}
)
# Returns: Plotly Figure object
```

**plot_mitigation_comparison(cba_df)**
```python
fig = visualizer.plot_mitigation_comparison(cba_df=cba_dataframe)
# Returns: Plotly Figure object (subplots)
```

**plot_climate_trends(climate_data)**
```python
fig = visualizer.plot_climate_trends(climate_data=climate_df)
# Returns: Plotly Figure object (4 subplots)
```

**save_figure(fig, filename)**
```python
visualizer.save_figure(fig, "loss_exceedance_curve.html")
# Saves Plotly figure to HTML file
```

### Helper Functions

**create_dashboard(analysis_results, climate_data, cba_df)**
```python
from src.visualization import create_dashboard

dashboard_dir = create_dashboard(
    analysis_results=financial_results,
    climate_data=climate_df,
    cba_df=cba_dataframe
)
# Returns: Path to output directory
# Creates 4 interactive HTML visualizations
```

---

## Main Workflow

### run_catia_analysis(region, use_mock_data)

```python
from main import run_catia_analysis

results = run_catia_analysis(
    region="US_Gulf_Coast",
    use_mock_data=True
)
# Returns: Dict with complete analysis results
# Includes: metadata, data_summary, risk_metrics, 
#           mitigation_summary, strategies, priority_order
```

---

## Configuration

### Access Configuration

```python
from config import get_config

api_config = get_config("api")
ml_config = get_config("ml")
simulation_config = get_config("simulation")
risk_metrics = get_config("risk_metrics")
mitigation_config = get_config("mitigation")
data_config = get_config("data")
```

### Modify Configuration

```python
from config import set_mock_data_mode

set_mock_data_mode(True)   # Use mock data
set_mock_data_mode(False)  # Use real APIs
```

---

## Data Types

### Climate Data DataFrame
```
Columns: date, temperature, precipitation, wind_speed, 
         sea_level_pressure, humidity, region
Types: datetime64, float64, float64, float64, float64, float64, object
```

### Socioeconomic Data DataFrame
```
Columns: region, population_density, gdp_per_capita, 
         infrastructure_index, poverty_rate
Types: object, float64, float64, float64, float64
```

### Historical Events DataFrame
```
Columns: year, event_type, region, magnitude, loss_usd, affected_population
Types: int64, object, object, float64, float64, int64
```

---

## Error Handling

### Common Exceptions

```python
# Data validation error
try:
    data = fetch_all_data("region")
except ValueError as e:
    print(f"Data validation failed: {e}")

# Model training error
try:
    predictor.train(X, y_prob, y_sev)
except Exception as e:
    print(f"Training failed: {e}")

# Optimization error
try:
    selected = optimizer.optimize_linear_programming()
except Exception as e:
    print(f"Optimization failed: {e}")
```

---

## Performance Tips

1. **Reduce iterations:** Set `monte_carlo_iterations: 1000` for testing
2. **Use mock data:** Faster than API calls
3. **Cache results:** Save intermediate results
4. **Parallelize:** Use multiprocessing for large simulations
5. **Monitor memory:** Watch for large data structures

---

## Examples

See `example_analysis.py` for comprehensive usage examples including:
- Basic analysis
- Sensitivity analysis
- Multi-region analysis
- Custom mitigation strategies

