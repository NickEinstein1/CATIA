"""
CATIA: Catastrophe AI System for Climate Risk Modeling

A comprehensive Python-based system for climate change and catastrophe modeling,
including data acquisition, risk prediction, financial impact simulation, and
mitigation recommendations.

Modules:
    - data_acquisition: Fetch and validate climate and socioeconomic data
    - risk_prediction: ML-based risk prediction models
    - financial_impact: Actuarial catastrophe modeling and Monte Carlo simulation
    - mitigation: Optimization and recommendation of mitigation strategies
    - visualization: Interactive dashboards and charts

Example:
    >>> from src.data_acquisition import fetch_all_data
    >>> from src.risk_prediction import train_risk_model
    >>> from src.financial_impact import run_financial_impact_analysis
    >>> 
    >>> data = fetch_all_data("US_Gulf_Coast", use_mock=True)
    >>> predictor = train_risk_model(data['climate'], data['socioeconomic'], data['historical_events'])
    >>> results = run_financial_impact_analysis(0.5, {'mu': 15, 'sigma': 2})
"""

__version__ = "1.0.0"
__author__ = "CATIA Development Team"
__all__ = [
    "data_acquisition",
    "risk_prediction",
    "financial_impact",
    "mitigation",
    "visualization"
]

