"""
Scenario Analysis Module for CATIA
Provides scenario-based stress testing for catastrophe models.
"""

import logging
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from catia.config import LOGGING_CONFIG, RISK_METRICS, PERIL_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


# Predefined scenarios for climate catastrophe modeling
PREDEFINED_SCENARIOS = {
    'baseline': {
        'name': 'Baseline',
        'description': 'Current climate conditions',
        'frequency_multiplier': 1.0,
        'severity_multiplier': 1.0
    },
    'moderate_climate_change': {
        'name': 'Moderate Climate Change (RCP 4.5)',
        'description': '2°C warming scenario by 2100',
        'frequency_multiplier': 1.3,
        'severity_multiplier': 1.2
    },
    'severe_climate_change': {
        'name': 'Severe Climate Change (RCP 8.5)',
        'description': '4°C warming scenario by 2100',
        'frequency_multiplier': 1.8,
        'severity_multiplier': 1.5
    },
    'extreme_event': {
        'name': 'Extreme Event Year',
        'description': '1-in-100 year event frequency',
        'frequency_multiplier': 3.0,
        'severity_multiplier': 2.0
    },
    'favorable': {
        'name': 'Favorable Conditions',
        'description': 'Below-average catastrophe activity',
        'frequency_multiplier': 0.6,
        'severity_multiplier': 0.8
    }
}

# Peril-specific scenarios with tailored multipliers
PERIL_SCENARIOS = {
    'hurricane': {
        'baseline': {'frequency_multiplier': 1.0, 'severity_multiplier': 1.0},
        'active_atlantic': {
            'name': 'Active Atlantic Season',
            'description': 'Above-normal Atlantic hurricane activity (La Niña)',
            'frequency_multiplier': 1.5,
            'severity_multiplier': 1.3
        },
        'quiet_season': {
            'name': 'Quiet Hurricane Season',
            'description': 'Below-normal activity (El Niño)',
            'frequency_multiplier': 0.6,
            'severity_multiplier': 0.9
        },
        'major_hurricane': {
            'name': 'Major Hurricane Landfall',
            'description': 'Category 4-5 hurricane makes landfall',
            'frequency_multiplier': 1.0,
            'severity_multiplier': 3.0
        }
    },
    'flood': {
        'baseline': {'frequency_multiplier': 1.0, 'severity_multiplier': 1.0},
        'atmospheric_river': {
            'name': 'Atmospheric River Event',
            'description': 'Prolonged heavy precipitation',
            'frequency_multiplier': 2.0,
            'severity_multiplier': 1.8
        },
        'flash_flood': {
            'name': 'Flash Flood Outbreak',
            'description': 'Multiple flash flood events',
            'frequency_multiplier': 2.5,
            'severity_multiplier': 1.5
        },
        'drought_break': {
            'name': 'Drought-Breaking Rains',
            'description': 'Heavy rain on dry ground',
            'frequency_multiplier': 1.2,
            'severity_multiplier': 1.4
        }
    },
    'wildfire': {
        'baseline': {'frequency_multiplier': 1.0, 'severity_multiplier': 1.0},
        'extreme_fire_weather': {
            'name': 'Extreme Fire Weather',
            'description': 'Dry, hot, windy conditions',
            'frequency_multiplier': 2.0,
            'severity_multiplier': 2.5
        },
        'megafire': {
            'name': 'Megafire Event',
            'description': 'Single fire > 100,000 acres',
            'frequency_multiplier': 0.8,
            'severity_multiplier': 4.0
        },
        'wui_expansion': {
            'name': 'WUI Expansion',
            'description': 'Increased wildland-urban interface exposure',
            'frequency_multiplier': 1.3,
            'severity_multiplier': 1.8
        }
    },
    'earthquake': {
        'baseline': {'frequency_multiplier': 1.0, 'severity_multiplier': 1.0},
        'major_quake': {
            'name': 'Major Earthquake (M7+)',
            'description': 'Magnitude 7+ event in urban area',
            'frequency_multiplier': 0.5,
            'severity_multiplier': 5.0
        },
        'aftershock_sequence': {
            'name': 'Aftershock Sequence',
            'description': 'Extended aftershock activity',
            'frequency_multiplier': 3.0,
            'severity_multiplier': 0.6
        },
        'cascading_failure': {
            'name': 'Cascading Infrastructure Failure',
            'description': 'Dam, pipeline, or building collapse',
            'frequency_multiplier': 1.0,
            'severity_multiplier': 2.5
        }
    }
}


def get_scenarios_for_peril(peril: str) -> Dict:
    """Get scenarios specific to a peril type."""
    base_scenarios = PREDEFINED_SCENARIOS.copy()
    peril_specific = PERIL_SCENARIOS.get(peril, {})

    # Merge peril-specific scenarios
    for key, value in peril_specific.items():
        if key != 'baseline':
            base_scenarios[f"{peril}_{key}"] = {
                'name': value.get('name', key),
                'description': value.get('description', ''),
                'frequency_multiplier': value.get('frequency_multiplier', 1.0),
                'severity_multiplier': value.get('severity_multiplier', 1.0)
            }

    return base_scenarios


class ScenarioAnalyzer:
    """Scenario-based stress testing for catastrophe models."""
    
    def __init__(self, simulator, scenarios: Dict = None):
        """
        Initialize scenario analyzer.
        
        Args:
            simulator: FinancialImpactSimulator instance
            scenarios: Custom scenarios dict (uses PREDEFINED_SCENARIOS if None)
        """
        self.simulator = simulator
        self.base_frequency = simulator.event_frequency
        self.base_severity_params = simulator.severity_params.copy()
        self.scenarios = scenarios or PREDEFINED_SCENARIOS
        logger.info(f"ScenarioAnalyzer initialized with {len(self.scenarios)} scenarios")
    
    def run_scenarios(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run all scenarios and collect results.
        
        Args:
            num_simulations: Number of Monte Carlo iterations per scenario
        
        Returns:
            Dictionary with scenario results
        """
        logger.info(f"Running {len(self.scenarios)} scenarios...")
        results = {}
        
        for scenario_id, scenario in self.scenarios.items():
            # Apply scenario multipliers
            self.simulator.event_frequency = (
                self.base_frequency * scenario['frequency_multiplier']
            )
            self.simulator.severity_params['mu'] = (
                self.base_severity_params['mu'] + np.log(scenario['severity_multiplier'])
            )
            
            # Run simulation
            losses = self.simulator.simulate_annual_losses(num_years=num_simulations)
            
            # Calculate metrics
            results[scenario_id] = {
                'name': scenario['name'],
                'description': scenario['description'],
                'mean_loss': np.mean(losses),
                'median_loss': np.median(losses),
                'std_loss': np.std(losses),
                'var_95': np.percentile(losses, 95),
                'tvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
                'max_loss': np.max(losses),
                'losses': losses,  # Store for return period analysis
                'return_periods': self._calculate_return_periods(losses)
            }
            
            logger.info(f"  {scenario['name']}: Mean=${results[scenario_id]['mean_loss']:,.0f}")
        
        # Restore original parameters
        self.simulator.event_frequency = self.base_frequency
        self.simulator.severity_params = self.base_severity_params.copy()
        
        return results
    
    def _calculate_return_periods(self, losses: np.ndarray) -> Dict[str, float]:
        """Calculate losses for standard return periods."""
        return_periods = RISK_METRICS.get("return_periods", [10, 25, 50, 100, 250, 500])
        results = {}
        
        for rp in return_periods:
            percentile = (1 - 1/rp) * 100
            results[f"{rp}_year"] = np.percentile(losses, percentile)
        
        return results
    
    def plot_scenarios(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create comparison chart for all scenarios.
        
        Args:
            results: Results from run_scenarios()
        
        Returns:
            Plotly figure
        """
        scenarios = list(results.keys())
        names = [results[s]['name'] for s in scenarios]
        means = [results[s]['mean_loss'] / 1e6 for s in scenarios]
        var_95s = [results[s]['var_95'] / 1e6 for s in scenarios]
        tvar_95s = [results[s]['tvar_95'] / 1e6 for s in scenarios]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Mean Loss',
            x=names,
            y=means,
            marker_color='#636efa'
        ))
        
        fig.add_trace(go.Bar(
            name='VaR (95%)',
            x=names,
            y=var_95s,
            marker_color='#ef553b'
        ))
        
        fig.add_trace(go.Bar(
            name='TVaR (95%)',
            x=names,
            y=tvar_95s,
            marker_color='#00cc96'
        ))
        
        fig.update_layout(
            title="Scenario Comparison: Loss Metrics",
            xaxis_title="Scenario",
            yaxis_title="Loss ($ Millions)",
            barmode='group',
            template='plotly_white',
            height=500,
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        return fig

    def plot_return_periods(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create return period curves for all scenarios.

        Args:
            results: Results from run_scenarios()

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a']

        for i, (scenario_id, scenario_data) in enumerate(results.items()):
            rp_data = scenario_data['return_periods']
            periods = []
            losses = []

            for key, value in sorted(rp_data.items()):
                if 'year' in key:
                    period = int(key.split('_')[0])
                    periods.append(period)
                    losses.append(value / 1e6)

            fig.add_trace(go.Scatter(
                x=periods,
                y=losses,
                mode='lines+markers',
                name=scenario_data['name'],
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title="Return Period Curves by Scenario",
            xaxis_title="Return Period (Years)",
            yaxis_title="Loss Level ($ Millions)",
            xaxis_type="log",
            template='plotly_white',
            height=500,
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        return fig

    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate text summary of scenario analysis.

        Args:
            results: Results from run_scenarios()

        Returns:
            Summary string
        """
        lines = ["\n  Scenario Analysis Summary:"]

        # Find baseline for comparison
        baseline_mean = results.get('baseline', {}).get('mean_loss', 0)

        for scenario_id, data in results.items():
            mean = data['mean_loss']
            if baseline_mean > 0 and scenario_id != 'baseline':
                pct_change = (mean - baseline_mean) / baseline_mean * 100
                lines.append(
                    f"  {data['name']}: Mean=${mean/1e6:.1f}M ({pct_change:+.1f}% vs baseline)"
                )
            else:
                lines.append(f"  {data['name']}: Mean=${mean/1e6:.1f}M")

        return "\n".join(lines)

