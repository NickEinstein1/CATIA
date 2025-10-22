# CATIA Technical Architecture

## System Design

### Module Breakdown

#### 1. Data Acquisition Module (`src/data_acquisition.py`)

**Purpose:** Fetch and validate climate and socioeconomic data

**Key Classes:**
- `DataAcquisition`: Main class for data fetching
  - `fetch_climate_data()`: Retrieves climate variables
  - `fetch_socioeconomic_data()`: Retrieves socioeconomic indicators
  - `fetch_historical_events()`: Retrieves historical catastrophe events
  - `validate_data()`: Cleans and validates data

**Data Sources:**
- NOAA: Climate data (temperature, precipitation, wind speed, pressure)
- ECMWF: Weather forecasts and reanalysis
- World Bank: Socioeconomic indicators (GDP, population, poverty)

**Error Handling:**
- Retry logic with exponential backoff
- Missing value detection and handling
- Outlier detection (3-sigma rule)
- Data type validation

#### 2. Risk Prediction Module (`src/risk_prediction.py`)

**Purpose:** ML-based prediction of catastrophe probability and severity

**Key Classes:**
- `RiskPredictor`: Main ML model class
  - `prepare_features()`: Feature engineering
  - `train()`: Model training with cross-validation
  - `predict()`: Generate predictions
  - `save_model()` / `load_model()`: Model persistence

**Models:**
- **Probability Model:** RandomForestClassifier
  - Predicts: P(event occurs)
  - Output: 0-1 probability
  
- **Severity Model:** RandomForestRegressor
  - Predicts: Expected loss magnitude
  - Output: Loss in USD

**Validation Metrics:**
- Accuracy, Precision, Recall, F1-Score (probability model)
- RMSE, MAE (severity model)
- Loss Ratio Accuracy (actuarial metric)

**Feature Engineering:**
- Climate aggregation (monthly)
- Socioeconomic normalization
- Historical event encoding

#### 3. Financial Impact Module (`src/financial_impact.py`)

**Purpose:** Actuarial catastrophe modeling and Monte Carlo simulation

**Key Classes:**
- `FinancialImpactSimulator`: Main simulation engine
  - `simulate_annual_losses()`: Single-year simulation
  - `monte_carlo_simulation()`: Multi-iteration simulation
  - `calculate_var_tvar()`: Risk metrics
  - `calculate_loss_exceedance_curve()`: Exceedance probabilities
  - `calculate_return_periods()`: Return period analysis

**Stochastic Models:**

**Frequency Model (Poisson):**
```
N ~ Poisson(λ)
where λ = expected events per year
```

**Severity Model (Lognormal):**
```
L ~ Lognormal(μ, σ)
where μ = mean of log-normal
      σ = std of log-normal
```

**Aggregate Loss:**
```
Annual Loss = Σ L_i for i = 1 to N
```

**Risk Metrics:**
- VaR(95%) = 95th percentile of loss distribution
- TVaR(95%) = E[Loss | Loss ≥ VaR(95%)]
- Return Periods: Loss levels for 10, 25, 50, 100, 250, 500, 1000-year events

#### 4. Mitigation Module (`src/mitigation.py`)

**Purpose:** Optimization and recommendation of mitigation strategies

**Key Classes:**
- `MitigationStrategy`: Individual strategy representation
- `MitigationOptimizer`: Optimization engine
  - `optimize_greedy()`: Greedy algorithm
  - `optimize_linear_programming()`: LP optimization
  - `calculate_cost_benefit_analysis()`: CBA
  - `generate_recommendations()`: Final recommendations

**Optimization Problem:**
```
Maximize: Σ (risk_reduction_i × x_i)
Subject to: Σ (cost_i × x_i) ≤ Budget
           x_i ∈ {0, 1}
```

**Cost-Benefit Analysis:**
- NPV calculation with discount rate
- Benefit-Cost Ratio
- Payback Period
- Sensitivity analysis

#### 5. Visualization Module (`src/visualization.py`)

**Purpose:** Interactive dashboards and charts

**Key Classes:**
- `CATIAVisualizer`: Main visualization engine
  - `plot_loss_exceedance_curve()`: LEC chart
  - `plot_risk_distribution()`: Loss histogram
  - `plot_return_period_curve()`: Return period analysis
  - `plot_mitigation_comparison()`: Strategy comparison
  - `plot_climate_trends()`: Climate variable trends

**Output Formats:**
- Interactive HTML (Plotly)
- Static PNG/PDF (optional)
- JSON reports

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION                                         │
│    ├─ NOAA API → Climate Data                              │
│    ├─ ECMWF API → Weather Data                             │
│    ├─ World Bank API → Socioeconomic Data                  │
│    └─ Historical Database → Event Data                     │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA VALIDATION & CLEANING                               │
│    ├─ Missing value handling                               │
│    ├─ Outlier detection                                    │
│    ├─ Type validation                                      │
│    └─ Range checking                                       │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                                      │
│    ├─ Climate aggregation                                  │
│    ├─ Socioeconomic normalization                          │
│    └─ Historical event encoding                            │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING & VALIDATION                              │
│    ├─ Probability model (RandomForest)                     │
│    ├─ Severity model (RandomForest)                        │
│    ├─ Cross-validation                                     │
│    └─ Actuarial metrics                                    │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RISK PREDICTION                                          │
│    ├─ Event probability prediction                         │
│    └─ Loss severity prediction                             │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. MONTE CARLO SIMULATION                                   │
│    ├─ Frequency simulation (Poisson)                       │
│    ├─ Severity simulation (Lognormal)                      │
│    ├─ Aggregate loss calculation                           │
│    └─ 10,000 iterations                                    │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. RISK METRICS CALCULATION                                 │
│    ├─ VaR (95%)                                            │
│    ├─ TVaR (95%)                                           │
│    ├─ Return periods                                       │
│    └─ Loss exceedance curve                                │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. MITIGATION OPTIMIZATION                                  │
│    ├─ Strategy selection                                   │
│    ├─ Cost-benefit analysis                                │
│    └─ Prioritization                                       │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. VISUALIZATION & REPORTING                                │
│    ├─ Interactive dashboards                               │
│    ├─ JSON reports                                         │
│    └─ Recommendations                                      │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Management

**config.py** contains:
- API endpoints and credentials
- ML hyperparameters
- Simulation parameters
- Risk metrics thresholds
- Mitigation strategies
- Output settings

**Environment Variables:**
```bash
NOAA_API_KEY=<key>
ECMWF_API_KEY=<key>
ECMWF_API_URL=<url>
CATIA_MOCK_DATA=true/false
CATIA_OUTPUT_DIR=outputs/
```

## Error Handling Strategy

**Levels:**
1. **API Level:** Retry logic, timeout handling
2. **Data Level:** Validation, cleaning, imputation
3. **Model Level:** Cross-validation, error metrics
4. **Simulation Level:** Convergence checks, bounds validation
5. **Application Level:** Logging, exception handling

## Performance Characteristics

**Computational Complexity:**
- Data Acquisition: O(n) where n = data points
- Model Training: O(n × m × log(n)) where m = features
- Monte Carlo: O(k × n) where k = iterations, n = events per iteration
- Optimization: O(m²) where m = strategies

**Memory Usage:**
- Climate data: ~100 MB (4 years daily)
- Model: ~50 MB (serialized)
- Simulation results: ~500 MB (10,000 iterations)

**Execution Time (Typical):**
- Data acquisition: 5-30 seconds
- Model training: 10-60 seconds
- Monte Carlo (10,000 iterations): 30-120 seconds
- Optimization: 1-5 seconds
- Visualization: 5-15 seconds
- **Total: ~2-5 minutes**

## Scalability Considerations

**Horizontal Scaling:**
- Parallelize Monte Carlo iterations
- Distribute data fetching across regions
- Use cloud functions for API calls

**Vertical Scaling:**
- Increase simulation iterations
- Add more features to ML model
- Extend historical data period

**Cloud Deployment:**
- AWS Lambda: Serverless execution
- Google Cloud Functions: Event-driven
- Azure Functions: Integrated with enterprise tools

