"""
Visualization Module for CATIA
Creates interactive dashboards and charts for risk analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class CATIAVisualizer:
    """Creates interactive visualizations for CATIA analysis."""
    
    def __init__(self, output_dir: str = OUTPUT_CONFIG["output_dir"]):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"CATIAVisualizer initialized (output_dir={output_dir})")
    
    def plot_loss_exceedance_curve(self, loss_levels: np.ndarray, 
                                   exceedance_probs: np.ndarray,
                                   var_95: float = None,
                                   tvar_95: float = None) -> go.Figure:
        """
        Create loss exceedance curve visualization.
        
        Args:
            loss_levels: Array of loss levels
            exceedance_probs: Array of exceedance probabilities
            var_95: Value-at-Risk at 95% (optional)
            tvar_95: Tail Value-at-Risk at 95% (optional)
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add main curve
        fig.add_trace(go.Scatter(
            x=loss_levels / 1e6,  # Convert to millions
            y=exceedance_probs * 100,  # Convert to percentage
            mode='lines',
            name='Loss Exceedance Curve',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add VaR line
        if var_95 is not None:
            fig.add_vline(
                x=var_95 / 1e6,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR (95%): ${var_95/1e6:.1f}M",
                annotation_position="top right"
            )
        
        # Add TVaR line
        if tvar_95 is not None:
            fig.add_vline(
                x=tvar_95 / 1e6,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"TVaR (95%): ${tvar_95/1e6:.1f}M",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Loss Exceedance Curve (Annual Losses)",
            xaxis_title="Loss Level ($ Millions)",
            yaxis_title="Probability of Exceedance (%)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1000
        )
        
        return fig
    
    def plot_risk_distribution(self, losses: np.ndarray) -> go.Figure:
        """
        Create risk distribution histogram.
        
        Args:
            losses: Array of simulated losses
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=losses / 1e6,
            nbinsx=50,
            name='Loss Distribution',
            marker_color='#2ca02c'
        ))
        
        # Add mean line
        mean_loss = np.mean(losses)
        fig.add_vline(
            x=mean_loss / 1e6,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_loss/1e6:.1f}M"
        )
        
        fig.update_layout(
            title="Annual Loss Distribution (Monte Carlo)",
            xaxis_title="Loss ($ Millions)",
            yaxis_title="Frequency",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1000
        )
        
        return fig
    
    def plot_return_period_curve(self, return_periods: Dict) -> go.Figure:
        """
        Create return period curve.
        
        Args:
            return_periods: Dictionary with return periods and loss levels
        
        Returns:
            Plotly figure
        """
        # Extract data
        periods = []
        losses = []
        
        for key, value in sorted(return_periods.items()):
            if isinstance(key, str) and 'year' in key:
                period = int(key.split('_')[0])
                periods.append(period)
                losses.append(value / 1e6)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=losses,
            mode='lines+markers',
            name='Return Period',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Return Period Analysis",
            xaxis_title="Return Period (Years)",
            yaxis_title="Loss Level ($ Millions)",
            xaxis_type="log",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1000
        )
        
        return fig
    
    def plot_mitigation_comparison(self, cba_df: pd.DataFrame) -> go.Figure:
        """
        Create mitigation strategy comparison chart.
        
        Args:
            cba_df: Cost-benefit analysis DataFrame
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Cost vs Risk Reduction", "Benefit-Cost Ratio")
        )
        
        # Cost vs Risk Reduction
        fig.add_trace(
            go.Scatter(
                x=cba_df['Risk_Reduction'] * 100,
                y=cba_df['Cost'] / 1e6,
                mode='markers+text',
                text=cba_df['Strategy'],
                textposition="top center",
                marker=dict(size=12, color='#1f77b4'),
                name='Strategies'
            ),
            row=1, col=1
        )
        
        # Benefit-Cost Ratio
        fig.add_trace(
            go.Bar(
                x=cba_df['Strategy'],
                y=cba_df['Benefit_Cost_Ratio'],
                marker_color='#2ca02c',
                name='BCR'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Risk Reduction (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cost ($ Millions)", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Benefit-Cost Ratio", row=1, col=2)
        
        fig.update_layout(
            title_text="Mitigation Strategy Analysis",
            height=600,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_climate_trends(self, climate_data: pd.DataFrame) -> go.Figure:
        """
        Create climate trends visualization.
        
        Args:
            climate_data: Climate data DataFrame
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperature", "Precipitation", "Wind Speed", "Pressure")
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=climate_data['date'], y=climate_data['temperature'],
                      name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        # Precipitation
        fig.add_trace(
            go.Bar(x=climate_data['date'], y=climate_data['precipitation'],
                   name='Precipitation', marker_color='blue'),
            row=1, col=2
        )
        
        # Wind Speed
        fig.add_trace(
            go.Scatter(x=climate_data['date'], y=climate_data['wind_speed'],
                      name='Wind Speed', line=dict(color='green')),
            row=2, col=1
        )
        
        # Pressure
        fig.add_trace(
            go.Scatter(x=climate_data['date'], y=climate_data['sea_level_pressure'],
                      name='Pressure', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="mm", row=1, col=2)
        fig.update_yaxes(title_text="km/h", row=2, col=1)
        fig.update_yaxes(title_text="hPa", row=2, col=2)
        
        fig.update_layout(
            title_text="Climate Variables Over Time",
            height=800,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str):
        """Save figure to HTML file."""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        logger.info(f"Figure saved: {filepath}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_dashboard(analysis_results: Dict, climate_data: pd.DataFrame, 
                    cba_df: pd.DataFrame) -> str:
    """
    Create comprehensive dashboard.
    
    Args:
        analysis_results: Financial impact analysis results
        climate_data: Climate data
        cba_df: Cost-benefit analysis DataFrame
    
    Returns:
        Path to dashboard HTML file
    """
    visualizer = CATIAVisualizer()
    
    # Extract data
    losses = analysis_results['simulation_results']['all_losses']
    loss_levels, exceedance_probs = analysis_results['loss_exceedance_curve'].values()
    var_95 = analysis_results['metrics']['risk_metrics']['var']
    tvar_95 = analysis_results['metrics']['risk_metrics']['tvar']
    
    # Create figures
    fig1 = visualizer.plot_loss_exceedance_curve(loss_levels, exceedance_probs, var_95, tvar_95)
    fig2 = visualizer.plot_risk_distribution(losses)
    fig3 = visualizer.plot_return_period_curve(analysis_results['metrics']['return_periods'])
    fig4 = visualizer.plot_mitigation_comparison(cba_df)
    
    # Save figures
    visualizer.save_figure(fig1, "loss_exceedance_curve.html")
    visualizer.save_figure(fig2, "risk_distribution.html")
    visualizer.save_figure(fig3, "return_period_curve.html")
    visualizer.save_figure(fig4, "mitigation_comparison.html")
    
    logger.info("Dashboard created successfully")
    return visualizer.output_dir

