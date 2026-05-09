"""
Sensitivity Analysis Module for CATIA
Provides sensitivity and tornado chart analysis for risk parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from catia.config import LOGGING_CONFIG, SIMULATION_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class QuickSensitivityAnalysis:
    """Sensitivity analysis for catastrophe model parameters."""
    
    def __init__(self, simulator):
        """
        Initialize sensitivity analyzer.
        
        Args:
            simulator: FinancialImpactSimulator instance
        """
        self.simulator = simulator
        self.base_frequency = simulator.event_frequency
        self.base_severity_params = simulator.severity_params.copy()
        logger.info("QuickSensitivityAnalysis initialized")
    
    def analyze(self, parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Run sensitivity analysis across parameter ranges.
        
        Args:
            parameter_ranges: Dict mapping parameter names to lists of values
                e.g., {'event_frequency': [0.3, 0.5, 0.7], 'severity_mu': [14, 15, 16]}
        
        Returns:
            Dictionary with sensitivity results
        """
        logger.info("Running sensitivity analysis...")
        results = {
            'parameters': {},
            'base_case': None
        }
        
        # Calculate base case
        base_losses = self.simulator.simulate_annual_losses(num_years=1000)
        base_mean = np.mean(base_losses)
        results['base_case'] = {'mean_loss': base_mean}
        
        # Analyze each parameter
        for param_name, values in parameter_ranges.items():
            param_results = []
            
            for value in values:
                # Temporarily modify simulator
                if param_name == 'event_frequency':
                    original = self.simulator.event_frequency
                    self.simulator.event_frequency = value
                elif param_name == 'severity_mu':
                    original = self.simulator.severity_params['mu']
                    self.simulator.severity_params['mu'] = value
                elif param_name == 'severity_sigma':
                    original = self.simulator.severity_params['sigma']
                    self.simulator.severity_params['sigma'] = value
                else:
                    continue
                
                # Run simulation
                losses = self.simulator.simulate_annual_losses(num_years=500)
                mean_loss = np.mean(losses)
                var_95 = np.percentile(losses, 95)
                
                param_results.append({
                    'value': value,
                    'mean_loss': mean_loss,
                    'var_95': var_95,
                    'pct_change': (mean_loss - base_mean) / base_mean * 100
                })
                
                # Restore original value
                if param_name == 'event_frequency':
                    self.simulator.event_frequency = original
                elif param_name == 'severity_mu':
                    self.simulator.severity_params['mu'] = original
                elif param_name == 'severity_sigma':
                    self.simulator.severity_params['sigma'] = original
            
            results['parameters'][param_name] = param_results
        
        logger.info(f"Sensitivity analysis complete for {len(parameter_ranges)} parameters")
        return results
    
    def plot_tornado(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create tornado chart showing parameter sensitivity.
        
        Args:
            results: Results from analyze()
        
        Returns:
            Plotly figure
        """
        base_mean = results['base_case']['mean_loss']
        
        # Calculate impact ranges for each parameter
        impacts = []
        for param_name, param_results in results['parameters'].items():
            pct_changes = [r['pct_change'] for r in param_results]
            impacts.append({
                'parameter': param_name,
                'min_impact': min(pct_changes),
                'max_impact': max(pct_changes),
                'range': max(pct_changes) - min(pct_changes)
            })
        
        # Sort by range (largest impact first)
        impacts.sort(key=lambda x: x['range'], reverse=True)
        
        fig = go.Figure()
        
        for i, impact in enumerate(impacts):
            # Low bar (negative side)
            fig.add_trace(go.Bar(
                y=[impact['parameter']],
                x=[impact['min_impact']],
                orientation='h',
                name=f"{impact['parameter']} (low)",
                marker_color='#ef553b',
                showlegend=False
            ))
            # High bar (positive side)
            fig.add_trace(go.Bar(
                y=[impact['parameter']],
                x=[impact['max_impact']],
                orientation='h',
                name=f"{impact['parameter']} (high)",
                marker_color='#636efa',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Tornado Chart: Parameter Sensitivity",
            xaxis_title="% Change in Mean Loss from Base Case",
            yaxis_title="Parameter",
            barmode='overlay',
            template='plotly_white',
            height=400,
            width=800
        )
        
        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def plot_sensitivity_heatmap(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create heatmap showing sensitivity across parameters.
        
        Args:
            results: Results from analyze()
        
        Returns:
            Plotly figure
        """
        # Build matrix for heatmap
        params = list(results['parameters'].keys())
        
        if len(params) < 2:
            # Single parameter - create simple bar chart instead
            param = params[0]
            values = [r['value'] for r in results['parameters'][param]]
            pct_changes = [r['pct_change'] for r in results['parameters'][param]]
            
            fig = go.Figure(go.Bar(x=values, y=pct_changes))
            fig.update_layout(
                title=f"Sensitivity: {param}",
                xaxis_title=param,
                yaxis_title="% Change in Mean Loss"
            )
            return fig
        
        # Two parameters - create proper heatmap
        param1, param2 = params[0], params[1]
        values1 = [r['value'] for r in results['parameters'][param1]]
        values2 = [r['value'] for r in results['parameters'][param2]]
        
        # Create matrix (simplified - just use the diagonal)
        z_matrix = []
        for r1 in results['parameters'][param1]:
            row = [r1['pct_change'] + r2['pct_change'] 
                   for r2 in results['parameters'][param2]]
            z_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=[str(v) for v in values2],
            y=[str(v) for v in values1],
            colorscale='RdYlGn_r',
            colorbar=dict(title="% Change")
        ))
        
        fig.update_layout(
            title="Sensitivity Heatmap: Combined Parameter Impact",
            xaxis_title=param2,
            yaxis_title=param1,
            template='plotly_white',
            height=500,
            width=700
        )
        
        return fig
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate text summary of sensitivity analysis.
        
        Args:
            results: Results from analyze()
        
        Returns:
            Summary string
        """
        lines = ["\n  Sensitivity Analysis Summary:"]
        lines.append(f"  Base Case Mean Loss: ${results['base_case']['mean_loss']:,.0f}")
        
        for param_name, param_results in results['parameters'].items():
            pct_changes = [r['pct_change'] for r in param_results]
            lines.append(f"  {param_name}: {min(pct_changes):+.1f}% to {max(pct_changes):+.1f}%")
        
        return "\n".join(lines)

