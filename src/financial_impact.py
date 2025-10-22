"""
Financial Impact Simulation Module for CATIA
Actuarial catastrophe modeling using frequency-severity models.
Monte Carlo simulations for loss exceedance curves and risk metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.stats import poisson, lognorm, pareto
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMULATION_CONFIG, RISK_METRICS, LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# ============================================================================
# FINANCIAL IMPACT SIMULATOR CLASS
# ============================================================================

class FinancialImpactSimulator:
    """Actuarial catastrophe modeling with Monte Carlo simulations."""
    
    def __init__(self, event_frequency: float, severity_params: Dict):
        """
        Initialize simulator.
        
        Args:
            event_frequency: Expected number of events per year (lambda for Poisson)
            severity_params: Parameters for severity distribution
                - For lognormal: {'mu': mean_log, 'sigma': std_log}
                - For pareto: {'scale': scale, 'shape': shape}
        """
        self.event_frequency = event_frequency
        self.severity_params = severity_params
        self.severity_dist = SIMULATION_CONFIG["severity_distribution"]
        self.random_seed = SIMULATION_CONFIG["random_seed"]
        np.random.seed(self.random_seed)
        logger.info(f"FinancialImpactSimulator initialized (frequency={event_frequency})")
    
    def simulate_annual_losses(self, num_years: int = 1) -> np.ndarray:
        """
        Simulate annual aggregate losses.
        
        Args:
            num_years: Number of years to simulate
        
        Returns:
            Array of annual aggregate losses
        """
        annual_losses = np.zeros(num_years)
        
        for year in range(num_years):
            # Simulate number of events (Poisson)
            num_events = poisson.rvs(self.event_frequency)
            
            # Simulate loss for each event
            if num_events > 0:
                if self.severity_dist == "Lognormal":
                    losses = lognorm.rvs(
                        s=self.severity_params['sigma'],
                        scale=np.exp(self.severity_params['mu']),
                        size=num_events
                    )
                elif self.severity_dist == "Pareto":
                    losses = pareto.rvs(
                        a=self.severity_params['shape'],
                        scale=self.severity_params['scale'],
                        size=num_events
                    )
                else:
                    raise ValueError(f"Unknown severity distribution: {self.severity_dist}")
                
                annual_losses[year] = losses.sum()
        
        return annual_losses
    
    def monte_carlo_simulation(self) -> Dict:
        """
        Run Monte Carlo simulation for loss exceedance curves.
        
        Returns:
            Dictionary with simulation results
        """
        num_iterations = SIMULATION_CONFIG["monte_carlo_iterations"]
        num_years = 1  # Annual losses
        
        logger.info(f"Running Monte Carlo simulation ({num_iterations} iterations)...")
        
        # Run simulations
        all_losses = []
        for i in range(num_iterations):
            annual_loss = self.simulate_annual_losses(num_years)
            all_losses.extend(annual_loss)
            
            if (i + 1) % (num_iterations // 10) == 0:
                logger.info(f"  Completed {i + 1}/{num_iterations} iterations")
        
        all_losses = np.array(all_losses)
        
        # Calculate risk metrics
        results = {
            'all_losses': all_losses,
            'mean_loss': np.mean(all_losses),
            'median_loss': np.median(all_losses),
            'std_loss': np.std(all_losses),
            'min_loss': np.min(all_losses),
            'max_loss': np.max(all_losses)
        }
        
        logger.info(f"Simulation complete. Mean loss: ${results['mean_loss']:,.0f}")
        return results
    
    def calculate_var_tvar(self, losses: np.ndarray) -> Dict:
        """
        Calculate Value-at-Risk (VaR) and Tail Value-at-Risk (TVaR).
        
        Args:
            losses: Array of simulated losses
        
        Returns:
            Dictionary with VaR and TVaR metrics
        """
        confidence = RISK_METRICS["var_confidence"]
        percentile = int(confidence * 100)
        
        var = np.percentile(losses, percentile)
        
        # TVaR: average of losses exceeding VaR
        tail_losses = losses[losses >= var]
        tvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        results = {
            'var_confidence': confidence,
            'var': var,
            'tvar': tvar,
            'var_percentile': percentile,
            'tail_losses_count': len(tail_losses)
        }
        
        logger.info(f"VaR ({percentile}%): ${var:,.0f}")
        logger.info(f"TVaR ({percentile}%): ${tvar:,.0f}")
        
        return results
    
    def calculate_loss_exceedance_curve(self, losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate loss exceedance probability curve.
        
        Args:
            losses: Array of simulated losses
        
        Returns:
            Tuple of (loss_levels, exceedance_probabilities)
        """
        # Sort losses
        sorted_losses = np.sort(losses)
        
        # Calculate exceedance probabilities
        n = len(sorted_losses)
        exceedance_probs = np.arange(1, n + 1) / n
        
        # Reverse for exceedance curve (probability of exceeding loss level)
        exceedance_probs = 1 - exceedance_probs
        
        return sorted_losses, exceedance_probs
    
    def calculate_return_periods(self, losses: np.ndarray) -> Dict:
        """
        Calculate return periods for specific loss levels.
        
        Args:
            losses: Array of simulated losses
        
        Returns:
            Dictionary with return periods
        """
        return_periods = RISK_METRICS["return_periods"]
        results = {}
        
        for rp in return_periods:
            # Return period = 1 / (1 - percentile)
            percentile = 1 - (1 / rp)
            loss_level = np.percentile(losses, percentile * 100)
            results[f"{rp}_year"] = loss_level
        
        logger.info("Return Period Analysis:")
        for rp, loss in results.items():
            logger.info(f"  {rp}: ${loss:,.0f}")
        
        return results
    
    def calculate_aggregate_metrics(self, losses: np.ndarray) -> Dict:
        """
        Calculate comprehensive aggregate metrics.
        
        Args:
            losses: Array of simulated losses
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'descriptive_stats': {
                'mean': np.mean(losses),
                'median': np.median(losses),
                'std': np.std(losses),
                'skewness': self._calculate_skewness(losses),
                'kurtosis': self._calculate_kurtosis(losses)
            },
            'risk_metrics': self.calculate_var_tvar(losses),
            'return_periods': self.calculate_return_periods(losses),
            'percentiles': {
                '50th': np.percentile(losses, 50),
                '75th': np.percentile(losses, 75),
                '90th': np.percentile(losses, 90),
                '95th': np.percentile(losses, 95),
                '99th': np.percentile(losses, 99)
            }
        }
        
        return metrics
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_financial_impact_analysis(event_frequency: float, 
                                  severity_params: Dict) -> Dict:
    """
    Run complete financial impact analysis.
    
    Args:
        event_frequency: Expected events per year
        severity_params: Severity distribution parameters
    
    Returns:
        Dictionary with all analysis results
    """
    simulator = FinancialImpactSimulator(event_frequency, severity_params)
    
    # Run Monte Carlo simulation
    sim_results = simulator.monte_carlo_simulation()
    
    # Calculate metrics
    metrics = simulator.calculate_aggregate_metrics(sim_results['all_losses'])
    
    # Calculate loss exceedance curve
    loss_levels, exceedance_probs = simulator.calculate_loss_exceedance_curve(sim_results['all_losses'])
    
    return {
        'simulation_results': sim_results,
        'metrics': metrics,
        'loss_exceedance_curve': {
            'loss_levels': loss_levels,
            'exceedance_probabilities': exceedance_probs
        }
    }

if __name__ == "__main__":
    # Example: Hurricane modeling
    event_frequency = 0.5  # 0.5 hurricanes per year on average
    severity_params = {
        'mu': 15,      # Mean of log-normal
        'sigma': 2     # Std of log-normal
    }
    
    results = run_financial_impact_analysis(event_frequency, severity_params)
    
    print("\nFinancial Impact Analysis Results:")
    print(f"Mean Annual Loss: ${results['metrics']['descriptive_stats']['mean']:,.0f}")
    print(f"VaR (95%): ${results['metrics']['risk_metrics']['var']:,.0f}")
    print(f"TVaR (95%): ${results['metrics']['risk_metrics']['tvar']:,.0f}")
    print(f"100-year loss: ${results['metrics']['return_periods']['100_year']:,.0f}")

