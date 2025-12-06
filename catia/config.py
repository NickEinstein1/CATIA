"""
Configuration settings for CATIA system.
Manages API endpoints, model parameters, and simulation settings.
"""

import os
from typing import Dict, Any

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    "NOAA": {
        "base_url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso",
        "endpoints": {
            "climate_data": "/noaa/NOAA-NCEI-0208688",
            "historical_events": "/noaa/NOAA-NCEI-0208688"
        },
        "timeout": 30,
        "retry_attempts": 3
    },
    "ECMWF": {
        "base_url": "https://api.ecmwf.int/v1",
        "endpoints": {
            "forecast": "/forecasts",
            "reanalysis": "/reanalysis"
        },
        "timeout": 60,
        "retry_attempts": 3
    },
    "WORLD_BANK": {
        "base_url": "https://api.worldbank.org/v2",
        "endpoints": {
            "population": "/country",
            "gdp": "/country"
        },
        "timeout": 30,
        "retry_attempts": 3
    }
}

# ============================================================================
# MACHINE LEARNING MODEL CONFIGURATION
# ============================================================================

ML_CONFIG = {
    "model_type": "RandomForest",  # RandomForest, GradientBoosting, NeuralNetwork
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "train_test_split": 0.8,
    "cross_validation_folds": 5,
    "feature_scaling": "StandardScaler",
    "model_path": "models/risk_model.pkl"
}

# ============================================================================
# ACTUARIAL SIMULATION CONFIGURATION
# ============================================================================

SIMULATION_CONFIG = {
    "monte_carlo_iterations": 10000,
    "confidence_level": 0.95,  # 95% for VaR/TVaR
    "frequency_distribution": "Poisson",  # Poisson for event frequency
    "severity_distribution": "Lognormal",  # Lognormal or Pareto for losses
    "random_seed": 42,
    "correlation_matrix_path": "data/peril_correlations.csv"
}

# ============================================================================
# RISK METRICS CONFIGURATION
# ============================================================================

RISK_METRICS = {
    "var_confidence": 0.95,  # Value-at-Risk at 95%
    "tvar_confidence": 0.95,  # Tail Value-at-Risk at 95%
    "return_periods": [10, 25, 50, 100, 250, 500, 1000],  # Years
    "loss_ratio_threshold": 0.75  # For model validation
}

# ============================================================================
# MITIGATION CONFIGURATION
# ============================================================================

MITIGATION_CONFIG = {
    "strategies": [
        "infrastructure_hardening",
        "insurance_coverage",
        "relocation",
        "early_warning_systems",
        "land_use_planning"
    ],
    "budget_constraint": 1_000_000,  # USD
    "optimization_method": "linear_programming",  # or "genetic_algorithm"
    "cost_benefit_discount_rate": 0.03  # 3% annual discount
}

# ============================================================================
# PERIL CONFIGURATION
# ============================================================================

PERIL_CONFIG = {
    "hurricane": {
        "name": "Hurricane",
        "description": "Tropical cyclones with sustained winds > 74 mph",
        "frequency_base": 0.5,  # Expected events per year
        "severity_params": {"mu": 15, "sigma": 2},  # Lognormal parameters
        "climate_drivers": ["wind_speed", "sea_level_pressure", "humidity"],
        "regions": ["US_Gulf_Coast", "US_East_Coast", "Caribbean", "Southeast_Asia"],
        "seasonality": [6, 7, 8, 9, 10, 11],  # June - November
        "magnitude_scale": "Saffir-Simpson (1-5)"
    },
    "flood": {
        "name": "Flood",
        "description": "River and flash flooding events",
        "frequency_base": 1.2,  # More frequent than hurricanes
        "severity_params": {"mu": 13, "sigma": 1.8},
        "climate_drivers": ["precipitation", "humidity"],
        "regions": ["US_Gulf_Coast", "US_Midwest", "Europe", "South_Asia"],
        "seasonality": [3, 4, 5, 6, 7, 8, 9],  # Spring-Summer
        "magnitude_scale": "Flood severity (1-5)"
    },
    "wildfire": {
        "name": "Wildfire",
        "description": "Uncontrolled fires in wildland areas",
        "frequency_base": 0.8,
        "severity_params": {"mu": 14, "sigma": 1.5},
        "climate_drivers": ["temperature", "humidity", "wind_speed"],
        "regions": ["US_West_Coast", "Australia", "Mediterranean", "South_America"],
        "seasonality": [6, 7, 8, 9, 10],  # Summer-Fall (fire season)
        "magnitude_scale": "Fire severity (1-5)"
    },
    "earthquake": {
        "name": "Earthquake",
        "description": "Seismic events causing ground shaking",
        "frequency_base": 0.3,  # Less frequent, but higher severity
        "severity_params": {"mu": 16, "sigma": 2.5},
        "climate_drivers": [],  # Not climate-driven
        "regions": ["US_West_Coast", "Japan", "Turkey", "Chile", "Indonesia"],
        "seasonality": list(range(1, 13)),  # Year-round
        "magnitude_scale": "Richter/Moment Magnitude (1-10)"
    }
}

# Default perils to analyze
DEFAULT_PERILS = ["hurricane", "flood", "wildfire", "earthquake"]

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    "mock_data_enabled": True,  # Set to False for real API calls
    "mock_data_path": "data/",
    "climate_variables": [
        "temperature",
        "precipitation",
        "wind_speed",
        "sea_level_pressure",
        "humidity"
    ],
    "socioeconomic_variables": [
        "population_density",
        "gdp_per_capita",
        "infrastructure_index",
        "poverty_rate"
    ],
    "data_validation": {
        "check_missing_values": True,
        "check_outliers": True,
        "outlier_threshold": 3.0  # Standard deviations
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/catia.log"
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    "output_dir": "outputs/",
    "report_format": "json",  # json, csv, html
    "visualization_format": "html",  # html, png, pdf
    "save_intermediate_results": True
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(section: str) -> Dict[str, Any]:
    """Retrieve configuration for a specific section."""
    config_map = {
        "api": API_CONFIG,
        "ml": ML_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "risk_metrics": RISK_METRICS,
        "mitigation": MITIGATION_CONFIG,
        "data": DATA_CONFIG,
        "logging": LOGGING_CONFIG,
        "output": OUTPUT_CONFIG,
        "peril": PERIL_CONFIG
    }
    return config_map.get(section, {})

def get_peril_config(peril_type: str) -> Dict[str, Any]:
    """Retrieve configuration for a specific peril type."""
    return PERIL_CONFIG.get(peril_type, {})

def set_mock_data_mode(enabled: bool):
    """Toggle mock data mode for development."""
    DATA_CONFIG["mock_data_enabled"] = enabled

