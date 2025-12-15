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

from catia.config import SIMULATION_CONFIG, RISK_METRICS, LOGGING_CONFIG, PERIL_CONFIG
from catia.extreme_value import ExtremeValueAnalyzer, analyze_tail_risk
from catia.uncertainty import RiskMetricUncertainty, quantify_risk_uncertainty
from catia.correlation import PerilCorrelationSimulator, simulate_correlated_perils

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

    def calculate_evt_metrics(self, losses: np.ndarray,
                              threshold_percentile: float = 90.0) -> Dict:
        """
        Calculate EVT-based risk metrics using Generalized Pareto Distribution.

        This provides more accurate tail risk estimates than empirical methods,
        especially for rare events (100+ year return periods).

        Args:
            losses: Array of simulated losses
            threshold_percentile: Percentile for GPD threshold selection

        Returns:
            Dictionary with EVT-based risk metrics
        """
        logger.info("Calculating EVT-based risk metrics...")
        evt_results = analyze_tail_risk(losses, threshold_percentile)

        logger.info(f"  GPD Shape (ξ): {evt_results['gpd_parameters']['shape_xi']:.4f}")
        logger.info(f"  Tail Type: {evt_results['gpd_parameters']['tail_type']}")

        return evt_results

    def calculate_comprehensive_metrics(self, losses: np.ndarray) -> Dict:
        """
        Calculate all risk metrics including both empirical and EVT-based.

        Args:
            losses: Array of simulated losses

        Returns:
            Dictionary with comprehensive risk metrics
        """
        # Standard empirical metrics
        empirical_metrics = self.calculate_aggregate_metrics(losses)

        # EVT-based metrics for tail risk
        evt_metrics = self.calculate_evt_metrics(losses)

        # Combine into comprehensive result
        return {
            'empirical': empirical_metrics,
            'evt': evt_metrics,
            'summary': {
                'mean_annual_loss': empirical_metrics['descriptive_stats']['mean'],
                'empirical_var_95': empirical_metrics['risk_metrics']['var'],
                'evt_var_95': evt_metrics['var_estimates']['var_95'],
                'empirical_tvar_95': empirical_metrics['risk_metrics']['tvar'],
                'evt_tvar_95': evt_metrics['tvar_estimates']['tvar_95'],
                'tail_type': evt_metrics['gpd_parameters']['tail_type'],
                'gpd_shape': evt_metrics['gpd_parameters']['shape_xi'],
                'recommendation': self._get_var_recommendation(
                    empirical_metrics['risk_metrics']['var'],
                    evt_metrics['var_estimates']['var_95'],
                    evt_metrics['gpd_parameters']['shape_xi']
                )
            }
        }

    def _get_var_recommendation(self, emp_var: float, evt_var: float, shape: float) -> str:
        """Generate recommendation based on VaR comparison."""
        diff_pct = (evt_var - emp_var) / emp_var * 100

        if shape > 0.2 and diff_pct > 10:
            return (f"Heavy tail detected (ξ={shape:.3f}). EVT VaR is {diff_pct:.1f}% higher. "
                    "USE EVT ESTIMATES for capital reserves and reinsurance pricing.")
        elif shape > 0:
            return (f"Moderate tail (ξ={shape:.3f}). Consider EVT for 100+ year return periods.")
        else:
            return "Light tail - empirical estimates are adequate."

    def calculate_uncertainty(self, losses: np.ndarray,
                             n_bootstrap: int = 500,
                             include_gpd: bool = True) -> Dict:
        """
        Quantify uncertainty in risk metrics using bootstrap methods.

        Provides confidence intervals for VaR, TVaR, return periods,
        and GPD parameters.

        Args:
            losses: Array of simulated losses
            n_bootstrap: Number of bootstrap samples (more = slower but more accurate)
            include_gpd: Whether to include GPD-based uncertainty (slower)

        Returns:
            Dictionary with uncertainty quantification results
        """
        logger.info(f"Quantifying uncertainty with {n_bootstrap} bootstrap samples...")

        results = quantify_risk_uncertainty(
            losses,
            n_bootstrap=n_bootstrap,
            include_gpd=include_gpd
        )

        logger.info(f"Estimation quality: {results['summary']['estimation_quality']}")
        logger.info(f"Average CI width: {results['summary']['mean_relative_width_pct']:.1f}%")

        return results

    def calculate_full_analysis(self, losses: np.ndarray,
                                include_uncertainty: bool = True,
                                n_bootstrap: int = 300) -> Dict:
        """
        Perform complete analysis including metrics, EVT, and uncertainty.

        Args:
            losses: Array of simulated losses
            include_uncertainty: Whether to run bootstrap uncertainty analysis
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with complete analysis results
        """
        # Empirical metrics
        empirical = self.calculate_aggregate_metrics(losses)

        # EVT analysis
        evt = self.calculate_evt_metrics(losses)

        result = {
            'empirical': empirical,
            'evt': evt
        }

        # Uncertainty quantification (optional due to computational cost)
        if include_uncertainty:
            result['uncertainty'] = self.calculate_uncertainty(
                losses, n_bootstrap, include_gpd=False  # GPD uncertainty is slow
            )

        return result

# ============================================================================
# MULTI-PERIL SIMULATOR CLASS
# ============================================================================

class MultiPerilSimulator:
    """Simulates losses across multiple peril types with correlations."""

    def __init__(self, perils: List[str] = None,
                 use_correlation: bool = True,
                 copula_type: str = "t"):
        """
        Initialize multi-peril simulator.

        Args:
            perils: List of peril types to simulate
            use_correlation: Whether to use copula-based correlation
            copula_type: Type of copula ('gaussian', 't', 'gumbel', 'clayton')
        """
        self.perils = perils or list(PERIL_CONFIG.keys())
        self.use_correlation = use_correlation
        self.copula_type = copula_type
        self.simulators = {}
        self.correlation_simulator = None

        # Create simulators for each peril
        for peril in self.perils:
            config = PERIL_CONFIG.get(peril, {})
            self.simulators[peril] = FinancialImpactSimulator(
                event_frequency=config.get('frequency_base', 0.5),
                severity_params=config.get('severity_params', {'mu': 15, 'sigma': 2})
            )

        # Create correlation simulator if enabled
        if use_correlation and len(self.perils) > 1:
            self.correlation_simulator = PerilCorrelationSimulator(
                self.perils, copula_type=copula_type
            )

        logger.info(f"MultiPerilSimulator initialized with {len(self.perils)} perils")
        if use_correlation:
            logger.info(f"  Correlation: {copula_type}-copula enabled")

    def simulate_all_perils(self, num_iterations: int = None) -> Dict:
        """
        Run simulation for all perils.

        Args:
            num_iterations: Number of Monte Carlo iterations

        Returns:
            Dictionary with results for each peril and aggregate
        """
        num_iterations = num_iterations or SIMULATION_CONFIG["monte_carlo_iterations"]
        results = {'by_peril': {}, 'aggregate': {}}

        if self.use_correlation and self.correlation_simulator:
            # Use correlated simulation
            return self._simulate_correlated(num_iterations, results)
        else:
            # Use independent simulation (original behavior)
            return self._simulate_independent(num_iterations, results)

    def _simulate_independent(self, num_iterations: int, results: Dict) -> Dict:
        """Simulate perils independently (no correlation)."""
        total_losses = np.zeros(num_iterations)

        for peril, simulator in self.simulators.items():
            losses = simulator.simulate_annual_losses(num_years=num_iterations)
            metrics = simulator.calculate_aggregate_metrics(losses)

            results['by_peril'][peril] = {
                'name': PERIL_CONFIG[peril]['name'],
                'losses': losses,
                'metrics': metrics
            }

            total_losses += losses
            logger.info(f"  {peril}: Mean=${metrics['descriptive_stats']['mean']:,.0f}")

        aggregate_simulator = FinancialImpactSimulator(1.0, {'mu': 15, 'sigma': 2})
        results['aggregate'] = {
            'losses': total_losses,
            'metrics': aggregate_simulator.calculate_aggregate_metrics(total_losses)
        }
        results['correlation_used'] = False

        logger.info(f"Aggregate Mean Loss: ${results['aggregate']['metrics']['descriptive_stats']['mean']:,.0f}")
        return results

    def _simulate_correlated(self, num_iterations: int, results: Dict) -> Dict:
        """Simulate perils with copula-based correlation."""
        # Build marginal parameters from config
        marginal_params = {}
        for peril in self.perils:
            cfg = PERIL_CONFIG.get(peril, {})
            sev = cfg.get('severity_params', {'mu': 15, 'sigma': 2})
            mu, sigma = sev.get('mu', 15), sev.get('sigma', 2)
            mean = np.exp(mu + sigma**2 / 2)
            std = mean * np.sqrt(np.exp(sigma**2) - 1)
            marginal_params[peril] = {
                'mean': mean,
                'std': std,
                'distribution': 'lognormal'
            }

        # Generate correlated losses
        correlated_losses = self.correlation_simulator.simulate_correlated_losses(
            num_iterations, marginal_params
        )

        # Apply frequency adjustment (not all years have events)
        total_losses = np.zeros(num_iterations)

        for peril in self.perils:
            cfg = PERIL_CONFIG.get(peril, {})
            frequency = cfg.get('frequency_base', 0.5)

            # Apply frequency mask
            rng = np.random.default_rng()
            event_mask = rng.random(num_iterations) < frequency
            losses = np.where(event_mask, correlated_losses[peril], 0)

            simulator = self.simulators[peril]
            metrics = simulator.calculate_aggregate_metrics(losses)

            results['by_peril'][peril] = {
                'name': PERIL_CONFIG[peril]['name'],
                'losses': losses,
                'metrics': metrics
            }

            total_losses += losses
            logger.info(f"  {peril}: Mean=${metrics['descriptive_stats']['mean']:,.0f}")

        aggregate_simulator = FinancialImpactSimulator(1.0, {'mu': 15, 'sigma': 2})
        results['aggregate'] = {
            'losses': total_losses,
            'metrics': aggregate_simulator.calculate_aggregate_metrics(total_losses)
        }
        results['correlation_used'] = True
        results['correlation_info'] = self.correlation_simulator.get_correlation_summary()

        logger.info(f"Aggregate Mean Loss: ${results['aggregate']['metrics']['descriptive_stats']['mean']:,.0f}")
        logger.info(f"  Tail dependence: {self.correlation_simulator.copula.tail_dependence}")

        return results

    def get_peril_contribution(self, results: Dict) -> pd.DataFrame:
        """
        Calculate each peril's contribution to total loss.

        Args:
            results: Results from simulate_all_perils()

        Returns:
            DataFrame with peril contributions
        """
        contributions = []
        total_mean = results['aggregate']['metrics']['descriptive_stats']['mean']

        for peril, data in results['by_peril'].items():
            mean_loss = data['metrics']['descriptive_stats']['mean']
            contributions.append({
                'peril': peril,
                'peril_name': data['name'],
                'mean_loss': mean_loss,
                'contribution_pct': (mean_loss / total_mean * 100) if total_mean > 0 else 0,
                'var_95': data['metrics']['risk_metrics']['var'],
                'tvar_95': data['metrics']['risk_metrics']['tvar']
            })

        return pd.DataFrame(contributions).sort_values('mean_loss', ascending=False)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_financial_impact_analysis(event_frequency: float,
                                  severity_params: Dict,
                                  peril: str = None) -> Dict:
    """
    Run complete financial impact analysis.

    Args:
        event_frequency: Expected events per year
        severity_params: Severity distribution parameters
        peril: Optional peril name for labeling

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

    result = {
        'simulation_results': sim_results,
        'metrics': metrics,
        'loss_exceedance_curve': {
            'loss_levels': loss_levels,
            'exceedance_probabilities': exceedance_probs
        }
    }

    if peril:
        result['peril'] = peril
        result['peril_name'] = PERIL_CONFIG.get(peril, {}).get('name', peril)

    return result


def run_multi_peril_analysis(perils: List[str] = None,
                             include_evt: bool = True,
                             include_uncertainty: bool = False,
                             include_correlation: bool = True,
                             copula_type: str = "t",
                             n_bootstrap: int = 300) -> Dict:
    """
    Run financial impact analysis across multiple perils.

    Args:
        perils: List of peril types (uses all if None)
        include_evt: Whether to include EVT-based tail risk analysis
        include_uncertainty: Whether to include uncertainty quantification
        include_correlation: Whether to use copula-based peril correlation
        copula_type: Type of copula ('gaussian', 't', 'gumbel', 'clayton')
        n_bootstrap: Number of bootstrap samples for uncertainty

    Returns:
        Dictionary with multi-peril analysis results
    """
    simulator = MultiPerilSimulator(
        perils,
        use_correlation=include_correlation,
        copula_type=copula_type
    )
    results = simulator.simulate_all_perils()
    contributions = simulator.get_peril_contribution(results)

    output = {
        'perils': simulator.perils,
        'results': results,
        'contributions': contributions.to_dict('records'),
        'aggregate_metrics': results['aggregate']['metrics'],
        'correlation_used': results.get('correlation_used', False)
    }

    # Add correlation info if used
    if results.get('correlation_used') and 'correlation_info' in results:
        output['correlation_info'] = results['correlation_info']

    aggregate_losses = results['aggregate']['losses']

    # Add EVT analysis for aggregate losses
    if include_evt:
        logger.info("Running EVT tail risk analysis on aggregate losses...")
        evt_results = analyze_tail_risk(aggregate_losses)
        output['evt_analysis'] = evt_results
        output['aggregate_metrics']['evt'] = {
            'gpd_var_95': evt_results['var_estimates']['var_95'],
            'gpd_tvar_95': evt_results['tvar_estimates']['tvar_95'],
            'gpd_shape': evt_results['gpd_parameters']['shape_xi'],
            'tail_type': evt_results['gpd_parameters']['tail_type'],
            'return_periods_evt': evt_results['return_period_levels']
        }

    # Add uncertainty quantification
    if include_uncertainty:
        logger.info(f"Running uncertainty analysis ({n_bootstrap} bootstrap samples)...")
        uncertainty_results = quantify_risk_uncertainty(
            aggregate_losses,
            n_bootstrap=n_bootstrap,
            include_gpd=False  # GPD bootstrap is slow, skip for speed
        )
        output['uncertainty'] = uncertainty_results
        output['aggregate_metrics']['confidence_intervals'] = {
            'var_95': uncertainty_results['confidence_intervals']['var_95'],
            'tvar_95': uncertainty_results['confidence_intervals']['tvar_95'],
            '100_year': uncertainty_results['confidence_intervals']['100_year'],
            'estimation_quality': uncertainty_results['summary']['estimation_quality']
        }

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Peril Financial Impact Analysis")
    print("=" * 60)

    # Run multi-peril analysis
    results = run_multi_peril_analysis()

    print(f"\nPerils Analyzed: {results['perils']}")
    print("\nPer-Peril Results:")
    for contrib in results['contributions']:
        print(f"  {contrib['peril_name']}: "
              f"Mean=${contrib['mean_loss']:,.0f} "
              f"({contrib['contribution_pct']:.1f}%)")

    print("\nAggregate Results:")
    agg = results['aggregate_metrics']
    print(f"  Mean Annual Loss: ${agg['descriptive_stats']['mean']:,.0f}")
    print(f"  VaR (95%): ${agg['risk_metrics']['var']:,.0f}")
    print(f"  TVaR (95%): ${agg['risk_metrics']['tvar']:,.0f}")
    print(f"  100-year loss: ${agg['return_periods']['100_year']:,.0f}")

