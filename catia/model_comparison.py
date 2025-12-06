"""
Model Comparison Module for CATIA
Compares different model configurations and distributions.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, pareto, weibull_min

from catia.config import LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare different catastrophe model configurations."""
    
    def __init__(self, base_simulator=None):
        """
        Initialize model comparison.
        
        Args:
            base_simulator: Optional base FinancialImpactSimulator for comparison
        """
        self.base_simulator = base_simulator
        self.model_results = {}
        logger.info("ModelComparison initialized")
    
    def add_model(self, name: str, simulator, description: str = ""):
        """
        Add a model configuration for comparison.
        
        Args:
            name: Model identifier
            simulator: FinancialImpactSimulator instance
            description: Model description
        """
        self.model_results[name] = {
            'simulator': simulator,
            'description': description,
            'results': None
        }
        logger.info(f"Added model: {name}")
    
    def run_comparison(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run all models and compare results.
        
        Args:
            num_simulations: Number of Monte Carlo iterations
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Running comparison across {len(self.model_results)} models...")
        
        for name, model_data in self.model_results.items():
            simulator = model_data['simulator']
            losses = simulator.simulate_annual_losses(num_years=num_simulations)
            
            model_data['results'] = {
                'losses': losses,
                'mean': np.mean(losses),
                'median': np.median(losses),
                'std': np.std(losses),
                'var_95': np.percentile(losses, 95),
                'tvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
                'skewness': self._calculate_skewness(losses),
                'kurtosis': self._calculate_kurtosis(losses)
            }
            
            logger.info(f"  {name}: Mean=${model_data['results']['mean']:,.0f}")
        
        return self._compile_comparison()
    
    def _compile_comparison(self) -> Dict[str, Any]:
        """Compile comparison metrics across all models."""
        comparison = {
            'models': {},
            'ranking': {}
        }
        
        for name, model_data in self.model_results.items():
            if model_data['results']:
                comparison['models'][name] = {
                    'description': model_data['description'],
                    **{k: v for k, v in model_data['results'].items() if k != 'losses'}
                }
        
        # Rank models by different metrics
        if comparison['models']:
            models = list(comparison['models'].keys())
            comparison['ranking']['by_mean'] = sorted(
                models, key=lambda m: comparison['models'][m]['mean']
            )
            comparison['ranking']['by_var'] = sorted(
                models, key=lambda m: comparison['models'][m]['var_95']
            )
        
        return comparison
    
    def plot_comparison(self) -> go.Figure:
        """
        Create comparison visualization.
        
        Returns:
            Plotly figure
        """
        models = []
        means = []
        var_95s = []
        tvar_95s = []
        
        for name, model_data in self.model_results.items():
            if model_data['results']:
                models.append(name)
                means.append(model_data['results']['mean'] / 1e6)
                var_95s.append(model_data['results']['var_95'] / 1e6)
                tvar_95s.append(model_data['results']['tvar_95'] / 1e6)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Mean Loss Comparison", "Risk Metrics Comparison")
        )
        
        # Mean loss bar chart
        fig.add_trace(
            go.Bar(x=models, y=means, name="Mean Loss", marker_color='#636efa'),
            row=1, col=1
        )
        
        # VaR/TVaR comparison
        fig.add_trace(
            go.Bar(x=models, y=var_95s, name="VaR (95%)", marker_color='#ef553b'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=tvar_95s, name="TVaR (95%)", marker_color='#00cc96'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Model Comparison",
            template='plotly_white',
            height=450,
            width=1000,
            barmode='group'
        )
        
        fig.update_yaxes(title_text="$ Millions", row=1, col=1)
        fig.update_yaxes(title_text="$ Millions", row=1, col=2)
        
        return fig
    
    def plot_distributions(self) -> go.Figure:
        """
        Plot loss distributions for all models.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a']
        
        for i, (name, model_data) in enumerate(self.model_results.items()):
            if model_data['results']:
                losses = model_data['results']['losses']
                fig.add_trace(go.Histogram(
                    x=losses / 1e6,
                    name=name,
                    opacity=0.6,
                    marker_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            title="Loss Distribution Comparison",
            xaxis_title="Loss ($ Millions)",
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_white',
            height=450,
            width=900
        )

        return fig

    def generate_summary(self) -> str:
        """
        Generate text summary of model comparison.

        Returns:
            Summary string
        """
        lines = ["\n  Model Comparison Summary:"]

        for name, model_data in self.model_results.items():
            if model_data['results']:
                r = model_data['results']
                lines.append(
                    f"  {name}: Mean=${r['mean']/1e6:.1f}M, "
                    f"VaR=${r['var_95']/1e6:.1f}M, "
                    f"TVaR=${r['tvar_95']/1e6:.1f}M"
                )

        return "\n".join(lines)

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

