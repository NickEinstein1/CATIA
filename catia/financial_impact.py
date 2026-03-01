"""
Financial Impact Simulation Module for CATIA
Actuarial catastrophe modeling using frequency-severity models.
Monte Carlo simulations for loss exceedance curves and risk metrics.
"""

import logging
import time

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from catia.exposure import ExposureStore
    from catia.vulnerability import VulnerabilitySet

from scipy.stats import gamma, lognorm, pareto, poisson, weibull_min

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    Parallel = delayed = None

try:
    from catia.metrics import record_simulation_duration
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    record_simulation_duration = None

from catia.config import (
    INTENSITY_DISTRIBUTION,
    PERIL_CONFIG,
    LOGGING_CONFIG,
    RISK_METRICS,
    SIMULATION_CONFIG,
)
from catia.extreme_value import ExtremeValueAnalyzer, analyze_tail_risk
from catia.uncertainty import RiskMetricUncertainty, quantify_risk_uncertainty
from catia.correlation import PerilCorrelationSimulator, simulate_correlated_perils

try:
    from catia.climate_scenarios import apply_scenario_to_peril_config
    _CLIMATE_SCENARIOS_AVAILABLE = True
except ImportError:
    _CLIMATE_SCENARIOS_AVAILABLE = False
    apply_scenario_to_peril_config = None  # type: ignore

try:
    from catia.exposure import ExposureStore
    from catia.vulnerability import VulnerabilitySet
    _EXPOSURE_AVAILABLE = True
except ImportError:
    _EXPOSURE_AVAILABLE = False
    ExposureStore = None  # type: ignore
    VulnerabilitySet = None  # type: ignore

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


def _spliced_threshold_from_config(severity_params: Dict) -> float:
    """Compute spliced threshold as percentile of body (lognormal) from config."""
    body_s = severity_params.get("body_sigma", severity_params.get("sigma", 2))
    body_scale = np.exp(severity_params.get("body_mu", severity_params.get("mu", 15)))
    q = SIMULATION_CONFIG.get("spliced_threshold_percentile", 90) / 100.0
    return float(lognorm.ppf(q, s=body_s, scale=body_scale))


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
                - Lognormal: {'mu': mean_log, 'sigma': std_log}
                - Pareto: {'scale': scale, 'shape': shape}
                - Weibull: {'c': shape, 'scale': scale}
                - Gamma: {'a': shape, 'scale': scale}
                - Spliced: {'body_mu', 'body_sigma', 'tail_shape', 'tail_scale', 'threshold'} or from config
        """
        self.event_frequency = event_frequency
        self.severity_params = severity_params
        self.severity_dist = SIMULATION_CONFIG.get("severity_distribution", "Lognormal")
        self.random_seed = SIMULATION_CONFIG["random_seed"]
        np.random.seed(self.random_seed)
        logger.info(f"FinancialImpactSimulator initialized (frequency={event_frequency})")

    def _sample_severity(self, size: int, rng=None) -> np.ndarray:
        """Sample severity losses from the configured distribution."""
        kw = {} if rng is None else {"random_state": rng}
        p = self.severity_params
        if self.severity_dist == "Lognormal":
            return lognorm.rvs(
                s=p["sigma"],
                scale=np.exp(p["mu"]),
                size=size,
                **kw,
            )
        if self.severity_dist == "Pareto":
            return pareto.rvs(
                p["shape"],
                scale=p["scale"],
                size=size,
                **kw,
            )
        if self.severity_dist == "Weibull":
            return weibull_min.rvs(
                c=p["c"],
                scale=p["scale"],
                size=size,
                **kw,
            )
        if self.severity_dist == "Gamma":
            return gamma.rvs(
                a=p["a"],
                scale=p["scale"],
                size=size,
                **kw,
            )
        if self.severity_dist == "Spliced":
            return self._sample_spliced(size, rng)
        raise ValueError(f"Unknown severity distribution: {self.severity_dist}")

    def _sample_spliced(self, size: int, rng=None) -> np.ndarray:
        """Spliced: body (lognormal) below threshold, tail (Pareto) above."""
        p = self.severity_params
        threshold = p.get("threshold")
        if threshold is None:
            threshold = _spliced_threshold_from_config(p)
        body_s = self.severity_params.get("body_sigma", self.severity_params.get("sigma", 2))
        body_scale = np.exp(self.severity_params.get("body_mu", self.severity_params.get("mu", 15)))
        p_body = float(lognorm.cdf(threshold, s=body_s, scale=body_scale))
        kw = {} if rng is None else {"random_state": rng}
        U = (rng.uniform(0, 1, size=size) if rng is not None else np.random.uniform(0, 1, size=size))
        out = np.empty(size)
        n_body = (U <= p_body).sum()
        n_tail = size - n_body
        if n_body > 0:
            u_body = (U[U <= p_body] / p_body) if p_body > 0 else np.zeros(n_body)
            out[U <= p_body] = lognorm.ppf(np.clip(u_body, 1e-10, 1 - 1e-10), s=body_s, scale=body_scale)
        if n_tail > 0:
            tail_shape = self.severity_params.get("tail_shape", 2.0)
            out[U > p_body] = pareto.rvs(tail_shape, scale=threshold, size=n_tail, **kw)
        return out
    
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
                losses = self._sample_severity(num_events, rng=np.random)
                
                annual_losses[year] = losses.sum()
        
        return annual_losses
    
    def _simulate_chunk(self, chunk_size: int, seed: int) -> np.ndarray:
        """Run a chunk of annual loss simulations with fixed seed (for parallel reproducibility)."""
        rng = np.random.default_rng(seed)
        annual_losses = np.zeros(chunk_size)
        for i in range(chunk_size):
            num_events = poisson.rvs(self.event_frequency, random_state=rng)
            if num_events > 0:
                losses = self._sample_severity(num_events, rng=rng)
                annual_losses[i] = losses.sum()
        return annual_losses

    def monte_carlo_simulation(self) -> Dict:
        """
        Run Monte Carlo simulation for loss exceedance curves.
        Uses joblib for parallel execution when n_jobs is set in SIMULATION_CONFIG.
        Records duration metric if enabled.
        """
        start = time.time()
        num_iterations = SIMULATION_CONFIG["monte_carlo_iterations"]
        n_jobs = SIMULATION_CONFIG.get("n_jobs", 1)

        use_parallel = _JOBLIB_AVAILABLE and n_jobs != 1 and num_iterations >= 100
        if use_parallel:
            n_jobs = n_jobs if n_jobs > 0 else -1
            chunk_size = max(1, num_iterations // (n_jobs * 4 if n_jobs > 0 else 32))
            chunks = []
            it = 0
            while it < num_iterations:
                size = min(chunk_size, num_iterations - it)
                chunks.append((size, it))
                it += size
            base_seed = self.random_seed
            logger.info("Running Monte Carlo simulation (%s iterations, n_jobs=%s)...", num_iterations, n_jobs)
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(self._simulate_chunk)(size, base_seed + idx)
                for idx, (size, _) in enumerate(chunks)
            )
            all_losses = np.concatenate(results_list)
        else:
            num_years = 1
            logger.info("Running Monte Carlo simulation (%s iterations)...", num_iterations)
            all_losses = []
            for i in range(num_iterations):
                annual_loss = self.simulate_annual_losses(num_years)
                all_losses.extend(annual_loss)
                if (i + 1) % max(1, num_iterations // 10) == 0:
                    logger.info("  Completed %s/%s iterations", i + 1, num_iterations)
            all_losses = np.array(all_losses)

        duration = time.time() - start
        if _METRICS_AVAILABLE and record_simulation_duration:
            record_simulation_duration(duration, perils=None)
        results = {
            "all_losses": all_losses,
            "mean_loss": float(np.mean(all_losses)),
            "median_loss": float(np.median(all_losses)),
            "std_loss": float(np.std(all_losses)),
            "min_loss": float(np.min(all_losses)),
            "max_loss": float(np.max(all_losses)),
        }
        logger.info("Simulation complete. Mean loss: $%s (duration: %.2fs)", f"{results['mean_loss']:,.0f}", duration)
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
                 copula_type: str = "t",
                 scenario_id: Optional[str] = None):
        """
        Initialize multi-peril simulator.

        Args:
            perils: List of peril types to simulate
            use_correlation: Whether to use copula-based correlation
            copula_type: Type of copula ('gaussian', 't', 'gumbel', 'clayton')
            scenario_id: Optional climate scenario (e.g. RCP4.5_mid, SSP2_2050, high_stress)
        """
        self.perils = perils or list(PERIL_CONFIG.keys())
        self.use_correlation = use_correlation
        self.copula_type = copula_type
        self.scenario_id = scenario_id
        self.simulators = {}
        self.correlation_simulator = None

        # Create simulators for each peril (apply scenario if set)
        for peril in self.perils:
            config = PERIL_CONFIG.get(peril, {})
            freq = config.get('frequency_base', 0.5)
            sev = dict(config.get('severity_params', {'mu': 15, 'sigma': 2}))
            if _CLIMATE_SCENARIOS_AVAILABLE and scenario_id and apply_scenario_to_peril_config:
                freq, sev = apply_scenario_to_peril_config(peril, freq, sev, scenario_id)
            self.simulators[peril] = FinancialImpactSimulator(
                event_frequency=freq,
                severity_params=sev,
            )

        # Create correlation simulator if enabled
        if use_correlation and len(self.perils) > 1:
            self.correlation_simulator = PerilCorrelationSimulator(
                self.perils, copula_type=copula_type
            )

        logger.info(f"MultiPerilSimulator initialized with {len(self.perils)} perils")
        if use_correlation:
            logger.info(f"  Correlation: {copula_type}-copula enabled")
        if scenario_id and scenario_id != "baseline":
            logger.info(f"  Climate scenario: {scenario_id}")

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
# EXPOSURE-BASED SIMULATION (loss = exposure × vulnerability)
# ============================================================================

def _sample_intensity(peril: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample hazard intensity for a peril using INTENSITY_DISTRIBUTION (e.g. Weibull)."""
    cfg = INTENSITY_DISTRIBUTION.get(peril, {"dist": "weibull", "scale": 50, "shape": 2.0})
    if cfg.get("dist") == "weibull":
        scale = cfg.get("scale", 50)
        shape = cfg.get("shape", 2.0)
        return weibull_min.rvs(shape, scale=scale, size=size, random_state=rng)
    # Fallback: uniform over a range
    return rng.uniform(0, 100, size=size)


def _simulate_exposure_based_one_year(
    exposure_store: "ExposureStore",
    vulnerability_set: "VulnerabilitySet",
    perils: List[str],
    rng: np.random.Generator,
    scenario_id: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Simulate one year of losses using exposure × vulnerability.
    Returns (peril_annual_losses dict, aggregate_annual_loss).
    """
    records = exposure_store.records()
    if not records:
        return {p: 0.0 for p in perils}, 0.0

    scenario_adj = {}
    if scenario_id and scenario_id != "baseline" and _CLIMATE_SCENARIOS_AVAILABLE:
        try:
            from catia.climate_scenarios import get_scenario_adjustments
            scenario_adj = get_scenario_adjustments(scenario_id)
        except Exception:
            pass

    total_tiv = exposure_store.get_total_tiv()
    peril_losses = {p: 0.0 for p in perils}

    for peril in perils:
        freq = PERIL_CONFIG.get(peril, {}).get("frequency_base", 0.5)
        freq_mult = scenario_adj.get(peril, {}).get("frequency_multiplier", 1.0)
        sev_mult = scenario_adj.get(peril, {}).get("severity_multiplier", 1.0)
        n_events = rng.poisson(freq * freq_mult)
        if n_events == 0:
            continue
        intensities = _sample_intensity(peril, n_events, rng)
        for intensity in intensities:
            damage_ratio = vulnerability_set.damage_ratio(peril, float(intensity))
            # Event loss = sum over exposure of TIV * damage_ratio (single intensity applied to all)
            event_loss = total_tiv * damage_ratio * sev_mult
            peril_losses[peril] += event_loss

    aggregate = sum(peril_losses.values())
    return peril_losses, aggregate


def run_exposure_based_simulation(
    exposure_store: "ExposureStore",
    vulnerability_set: "VulnerabilitySet",
    perils: List[str],
    num_iterations: int = None,
    random_seed: int = None,
    scenario_id: Optional[str] = None,
) -> Dict:
    """
    Run Monte Carlo simulation using exposure × vulnerability.
    Returns same structure as MultiPerilSimulator.simulate_all_perils() for downstream compatibility.
    """
    if not _EXPOSURE_AVAILABLE or ExposureStore is None or VulnerabilitySet is None:
        raise RuntimeError("Exposure and vulnerability modules required for exposure-based simulation")
    num_iterations = num_iterations or SIMULATION_CONFIG["monte_carlo_iterations"]
    random_seed = random_seed or SIMULATION_CONFIG["random_seed"]
    rng = np.random.default_rng(random_seed)

    by_peril_losses = {p: np.zeros(num_iterations) for p in perils}
    aggregate_losses = np.zeros(num_iterations)

    for i in range(num_iterations):
        peril_annual, agg = _simulate_exposure_based_one_year(
            exposure_store, vulnerability_set, perils, rng, scenario_id=scenario_id
        )
        for p in perils:
            by_peril_losses[p][i] = peril_annual[p]
        aggregate_losses[i] = agg

    results = {"by_peril": {}, "aggregate": {}, "correlation_used": False}
    aggregate_simulator = FinancialImpactSimulator(1.0, {"mu": 15, "sigma": 2})

    for peril in perils:
        losses = by_peril_losses[peril]
        sim = FinancialImpactSimulator(
            PERIL_CONFIG.get(peril, {}).get("frequency_base", 0.5),
            PERIL_CONFIG.get(peril, {}).get("severity_params", {"mu": 15, "sigma": 2}),
        )
        results["by_peril"][peril] = {
            "name": PERIL_CONFIG.get(peril, {}).get("name", peril),
            "losses": losses,
            "metrics": sim.calculate_aggregate_metrics(losses),
        }

    results["aggregate"] = {
        "losses": aggregate_losses,
        "metrics": aggregate_simulator.calculate_aggregate_metrics(aggregate_losses),
    }
    logger.info(
        "Exposure-based simulation complete. Mean aggregate loss: $%s",
        f"{results['aggregate']['metrics']['descriptive_stats']['mean']:,.0f}",
    )
    return results


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
                             n_bootstrap: int = 300,
                             num_iterations: Optional[int] = None,
                             scenario_id: Optional[str] = None,
                             exposure_store: Optional["ExposureStore"] = None,
                             vulnerability_set: Optional["VulnerabilitySet"] = None) -> Dict:
    """
    Run financial impact analysis across multiple perils.

    Args:
        perils: List of peril types (uses all if None)
        include_evt: Whether to include EVT-based tail risk analysis
        include_uncertainty: Whether to include uncertainty quantification
        include_correlation: Whether to use copula-based peril correlation
        copula_type: Type of copula ('gaussian', 't', 'gumbel', 'clayton')
        n_bootstrap: Number of bootstrap samples for uncertainty
        num_iterations: Override Monte Carlo iterations (uses config default if None)
        scenario_id: Optional climate scenario (e.g. RCP4.5_mid, SSP2_2050, high_stress)
        exposure_store: If set with vulnerability_set, use exposure × vulnerability loss
        vulnerability_set: If set with exposure_store, use exposure-based simulation

    Returns:
        Dictionary with multi-peril analysis results
    """
    perils = perils or list(PERIL_CONFIG.keys())

    if exposure_store is not None and vulnerability_set is not None and _EXPOSURE_AVAILABLE:
        results = run_exposure_based_simulation(
            exposure_store, vulnerability_set, perils,
            num_iterations=num_iterations,
            scenario_id=scenario_id,
        )
        contributions = MultiPerilSimulator(
            perils, use_correlation=False, scenario_id=scenario_id
        ).get_peril_contribution(results)
    else:
        simulator = MultiPerilSimulator(
            perils,
            use_correlation=include_correlation,
            copula_type=copula_type,
            scenario_id=scenario_id,
        )
        results = simulator.simulate_all_perils()
        contributions = simulator.get_peril_contribution(results)

    output = {
        'perils': perils,
        'results': results,
        'contributions': contributions.to_dict('records'),
        'aggregate_metrics': results['aggregate']['metrics'],
        'correlation_used': results.get('correlation_used', False),
    }
    if scenario_id and scenario_id != "baseline":
        output['scenario_id'] = scenario_id

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

