"""
CATIA: Catastrophe AI System for Climate Risk Modeling

A comprehensive Python-based system for climate change and catastrophe modeling,
including data acquisition, risk prediction, financial impact simulation, and
mitigation recommendations.

Supported Perils:
    - Hurricane: Tropical cyclones with sustained winds > 74 mph
    - Flood: River and flash flooding events
    - Wildfire: Uncontrolled fires in wildland areas
    - Earthquake: Seismic events causing ground shaking

Modules:
    - config: Configuration for perils, models, and simulations
    - data_acquisition: Fetch and validate climate and socioeconomic data
    - risk_prediction: ML-based risk prediction models
    - financial_impact: Actuarial catastrophe modeling and Monte Carlo simulation
    - extreme_value: Extreme Value Theory (EVT) with GPD for tail risk modeling
    - uncertainty: Bootstrap-based uncertainty quantification for risk metrics
    - correlation: Copula-based peril correlation modeling
    - ensemble: Ensemble ML models for robust risk predictions
    - explainability: SHAP-based model interpretability
    - backtesting: Historical validation and performance monitoring
    - mitigation: Optimization and recommendation of mitigation strategies
    - visualization: Interactive dashboards and charts
    - sensitivity_analysis: Parameter sensitivity and tornado charts
    - scenario_analysis: Climate scenario stress testing (peril-specific)
    - model_comparison: Compare different model configurations
    - risk_alerts: Threshold-based risk monitoring
    - export: Report generation in multiple formats

Example:
    >>> from catia.data_acquisition import fetch_all_data
    >>> from catia.risk_prediction import train_risk_model
    >>> from catia.financial_impact import run_multi_peril_analysis
    >>> from catia.extreme_value import analyze_tail_risk
    >>>
    >>> # Multi-peril analysis with EVT
    >>> results = run_multi_peril_analysis(["hurricane", "flood"], include_evt=True)
    >>> print(f"EVT VaR (95%): ${results['aggregate_metrics']['evt']['gpd_var_95']:,.0f}")
"""

__version__ = "1.7.0"  # Updated for backtesting framework
__author__ = "CATIA Development Team"
__all__ = [
    "config",
    "data_acquisition",
    "risk_prediction",
    "financial_impact",
    "extreme_value",
    "uncertainty",
    "correlation",
    "ensemble",
    "explainability",
    "backtesting",
    "mitigation",
    "visualization",
    "sensitivity_analysis",
    "scenario_analysis",
    "model_comparison",
    "risk_alerts",
    "export",
]

# Convenience imports for commonly used items
from catia.config import PERIL_CONFIG, DEFAULT_PERILS

