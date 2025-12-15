"""
Uncertainty Quantification Module for CATIA

Provides methods for quantifying uncertainty in catastrophe model predictions:
- Bootstrap confidence intervals for risk metrics (VaR, TVaR, return periods)
- Parameter uncertainty for GPD fitting
- Prediction intervals for loss estimates
- Model uncertainty through ensemble approaches

Key concepts:
- Bootstrap resampling: Non-parametric method for uncertainty estimation
- Confidence intervals: Range containing true value with specified probability
- Prediction intervals: Range for future observations (wider than CI)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

from catia.config import LOGGING_CONFIG, RISK_METRICS
from catia.extreme_value import ExtremeValueAnalyzer

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    estimate: float  # Point estimate
    lower: float  # Lower bound
    upper: float  # Upper bound
    confidence: float  # Confidence level (e.g., 0.95)
    method: str  # Method used (bootstrap, parametric, etc.)
    
    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower
    
    @property
    def relative_width(self) -> float:
        """Width as percentage of estimate."""
        return (self.width / self.estimate * 100) if self.estimate != 0 else np.inf


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification result."""
    metric_name: str
    point_estimate: float
    confidence_intervals: Dict[float, ConfidenceInterval]  # keyed by confidence level
    bootstrap_distribution: np.ndarray
    std_error: float
    coefficient_of_variation: float


class BootstrapAnalyzer:
    """
    Bootstrap-based uncertainty quantification.
    
    Uses non-parametric bootstrap resampling to estimate confidence intervals
    for any statistic computed from loss data.
    """
    
    def __init__(self, data: np.ndarray, n_bootstrap: int = 1000, 
                 random_state: Optional[int] = None):
        """
        Initialize bootstrap analyzer.
        
        Args:
            data: Original loss observations
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.data = np.asarray(data)
        self.n = len(self.data)
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_state)
        
        logger.info(f"BootstrapAnalyzer initialized: n={self.n}, "
                   f"n_bootstrap={n_bootstrap}")
    
    def _generate_bootstrap_samples(self) -> np.ndarray:
        """Generate bootstrap resamples."""
        return self.rng.choice(self.data, size=(self.n_bootstrap, self.n), 
                               replace=True)
    
    def compute_bootstrap_distribution(self, 
                                       statistic: Callable[[np.ndarray], float]
                                       ) -> np.ndarray:
        """
        Compute bootstrap distribution of a statistic.
        
        Args:
            statistic: Function that computes a scalar from an array
        
        Returns:
            Array of bootstrap statistic values
        """
        samples = self._generate_bootstrap_samples()
        bootstrap_stats = np.array([statistic(sample) for sample in samples])
        return bootstrap_stats
    
    def confidence_interval(self, 
                           statistic: Callable[[np.ndarray], float],
                           confidence: float = 0.95,
                           method: str = "percentile"
                           ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval.
        
        Args:
            statistic: Function to compute statistic
            confidence: Confidence level (default 0.95)
            method: 'percentile', 'basic', or 'bca'
        
        Returns:
            ConfidenceInterval with bounds
        """
        point_estimate = statistic(self.data)
        bootstrap_dist = self.compute_bootstrap_distribution(statistic)
        
        alpha = 1 - confidence
        
        if method == "percentile":
            lower = np.percentile(bootstrap_dist, alpha/2 * 100)
            upper = np.percentile(bootstrap_dist, (1 - alpha/2) * 100)
        elif method == "basic":
            # Basic bootstrap (pivot method)
            lower = 2 * point_estimate - np.percentile(bootstrap_dist, (1 - alpha/2) * 100)
            upper = 2 * point_estimate - np.percentile(bootstrap_dist, alpha/2 * 100)
        elif method == "bca":
            # Bias-corrected and accelerated (simplified version)
            lower, upper = self._bca_interval(bootstrap_dist, point_estimate, 
                                              statistic, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ConfidenceInterval(
            estimate=point_estimate,
            lower=lower,
            upper=upper,
            confidence=confidence,
            method=f"bootstrap_{method}"
        )
    
    def _bca_interval(self, bootstrap_dist: np.ndarray, 
                      point_estimate: float,
                      statistic: Callable, 
                      alpha: float) -> Tuple[float, float]:
        """Bias-corrected and accelerated bootstrap interval."""
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_dist < point_estimate))
        
        # Acceleration (jackknife estimate)
        jackknife_stats = np.array([
            statistic(np.delete(self.data, i)) 
            for i in range(min(self.n, 100))  # Limit for performance
        ])
        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats)**3) / \
            (6 * np.sum((jack_mean - jackknife_stats)**2)**1.5 + 1e-10)
        
        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha/2)
        z_1_alpha = stats.norm.ppf(1 - alpha/2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a*(z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a*(z0 + z_1_alpha)))
        
        return (np.percentile(bootstrap_dist, p_lower * 100),
                np.percentile(bootstrap_dist, p_upper * 100))


class RiskMetricUncertainty:
    """
    Uncertainty quantification for catastrophe risk metrics.

    Provides confidence intervals for VaR, TVaR, return periods, and
    GPD parameters using bootstrap methods.
    """

    def __init__(self, losses: np.ndarray, n_bootstrap: int = 500,
                 random_state: Optional[int] = None):
        """
        Initialize risk metric uncertainty analyzer.

        Args:
            losses: Array of loss observations
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
        """
        self.losses = np.asarray(losses)
        self.n_bootstrap = n_bootstrap
        self.bootstrap = BootstrapAnalyzer(losses, n_bootstrap, random_state)

        logger.info(f"RiskMetricUncertainty initialized with {len(losses)} observations")

    def var_confidence_interval(self, percentile: float = 95.0,
                                confidence: float = 0.95) -> ConfidenceInterval:
        """
        Compute confidence interval for VaR.

        Args:
            percentile: VaR percentile (e.g., 95 for 95% VaR)
            confidence: Confidence level for the interval

        Returns:
            ConfidenceInterval for VaR
        """
        def var_statistic(data):
            return np.percentile(data, percentile)

        ci = self.bootstrap.confidence_interval(var_statistic, confidence, "bca")
        logger.info(f"VaR({percentile}%) CI: [{ci.lower:,.0f}, {ci.upper:,.0f}] "
                   f"({confidence*100:.0f}% conf)")
        return ci

    def tvar_confidence_interval(self, percentile: float = 95.0,
                                 confidence: float = 0.95) -> ConfidenceInterval:
        """
        Compute confidence interval for TVaR (Expected Shortfall).

        Args:
            percentile: VaR percentile for tail threshold
            confidence: Confidence level for the interval

        Returns:
            ConfidenceInterval for TVaR
        """
        def tvar_statistic(data):
            var = np.percentile(data, percentile)
            tail = data[data >= var]
            return np.mean(tail) if len(tail) > 0 else var

        ci = self.bootstrap.confidence_interval(tvar_statistic, confidence, "bca")
        logger.info(f"TVaR({percentile}%) CI: [{ci.lower:,.0f}, {ci.upper:,.0f}] "
                   f"({confidence*100:.0f}% conf)")
        return ci

    def return_period_confidence_interval(self, return_period: int,
                                          confidence: float = 0.95
                                          ) -> ConfidenceInterval:
        """
        Compute confidence interval for return period loss level.

        Args:
            return_period: Return period in years (e.g., 100)
            confidence: Confidence level

        Returns:
            ConfidenceInterval for return period level
        """
        percentile = (1 - 1/return_period) * 100

        def rp_statistic(data):
            return np.percentile(data, percentile)

        ci = self.bootstrap.confidence_interval(rp_statistic, confidence, "percentile")
        logger.info(f"{return_period}-year RP CI: [{ci.lower:,.0f}, {ci.upper:,.0f}]")
        return ci

    def gpd_parameter_uncertainty(self, threshold_percentile: float = 90.0,
                                  confidence: float = 0.95) -> Dict:
        """
        Compute confidence intervals for GPD parameters.

        Args:
            threshold_percentile: Percentile for threshold selection
            confidence: Confidence level

        Returns:
            Dictionary with parameter CIs
        """
        def fit_gpd_and_get_shape(data):
            try:
                analyzer = ExtremeValueAnalyzer(data)
                threshold = np.percentile(data, threshold_percentile)
                result = analyzer.fit_gpd(threshold=threshold, method="mle")
                return result.shape
            except Exception:
                return np.nan

        def fit_gpd_and_get_scale(data):
            try:
                analyzer = ExtremeValueAnalyzer(data)
                threshold = np.percentile(data, threshold_percentile)
                result = analyzer.fit_gpd(threshold=threshold, method="mle")
                return result.scale
            except Exception:
                return np.nan

        shape_ci = self.bootstrap.confidence_interval(
            fit_gpd_and_get_shape, confidence, "percentile")
        scale_ci = self.bootstrap.confidence_interval(
            fit_gpd_and_get_scale, confidence, "percentile")

        logger.info(f"GPD Shape CI: [{shape_ci.lower:.4f}, {shape_ci.upper:.4f}]")
        logger.info(f"GPD Scale CI: [{scale_ci.lower:,.0f}, {scale_ci.upper:,.0f}]")

        return {
            'shape': shape_ci,
            'scale': scale_ci
        }

    def gpd_var_confidence_interval(self, var_confidence: float = 0.95,
                                    interval_confidence: float = 0.95,
                                    threshold_percentile: float = 90.0
                                    ) -> ConfidenceInterval:
        """
        Compute confidence interval for GPD-based VaR.

        Args:
            var_confidence: VaR confidence level (e.g., 0.95 for 95% VaR)
            interval_confidence: Confidence level for the CI
            threshold_percentile: GPD threshold percentile

        Returns:
            ConfidenceInterval for GPD VaR
        """
        def gpd_var_statistic(data):
            try:
                analyzer = ExtremeValueAnalyzer(data)
                threshold = np.percentile(data, threshold_percentile)
                analyzer.fit_gpd(threshold=threshold, method="mle")
                return analyzer.gpd_var(var_confidence)
            except Exception:
                return np.nan

        ci = self.bootstrap.confidence_interval(
            gpd_var_statistic, interval_confidence, "percentile")

        logger.info(f"GPD VaR({var_confidence*100:.0f}%) CI: "
                   f"[{ci.lower:,.0f}, {ci.upper:,.0f}]")
        return ci

    def comprehensive_uncertainty_analysis(self,
                                           include_gpd: bool = True
                                           ) -> Dict:
        """
        Compute comprehensive uncertainty analysis for all key metrics.

        Args:
            include_gpd: Whether to include GPD-based metrics (slower)

        Returns:
            Dictionary with all uncertainty results
        """
        logger.info("Running comprehensive uncertainty analysis...")

        results = {
            'n_observations': len(self.losses),
            'n_bootstrap': self.n_bootstrap,
            'empirical_metrics': {},
            'confidence_intervals': {}
        }

        # VaR confidence intervals
        for pct in [90, 95, 99]:
            ci = self.var_confidence_interval(pct, 0.95)
            results['confidence_intervals'][f'var_{pct}'] = {
                'estimate': ci.estimate,
                'lower': ci.lower,
                'upper': ci.upper,
                'relative_width_pct': ci.relative_width
            }

        # TVaR confidence intervals
        for pct in [90, 95, 99]:
            ci = self.tvar_confidence_interval(pct, 0.95)
            results['confidence_intervals'][f'tvar_{pct}'] = {
                'estimate': ci.estimate,
                'lower': ci.lower,
                'upper': ci.upper,
                'relative_width_pct': ci.relative_width
            }

        # Return period confidence intervals
        for rp in [10, 25, 50, 100, 250]:
            ci = self.return_period_confidence_interval(rp, 0.95)
            results['confidence_intervals'][f'{rp}_year'] = {
                'estimate': ci.estimate,
                'lower': ci.lower,
                'upper': ci.upper,
                'relative_width_pct': ci.relative_width
            }

        # GPD-based metrics (if requested)
        if include_gpd:
            logger.info("Computing GPD parameter uncertainty...")
            gpd_params = self.gpd_parameter_uncertainty()
            results['gpd_parameters'] = {
                'shape': {
                    'estimate': gpd_params['shape'].estimate,
                    'lower': gpd_params['shape'].lower,
                    'upper': gpd_params['shape'].upper
                },
                'scale': {
                    'estimate': gpd_params['scale'].estimate,
                    'lower': gpd_params['scale'].lower,
                    'upper': gpd_params['scale'].upper
                }
            }

            # GPD VaR
            gpd_var_ci = self.gpd_var_confidence_interval(0.95, 0.95)
            results['confidence_intervals']['gpd_var_95'] = {
                'estimate': gpd_var_ci.estimate,
                'lower': gpd_var_ci.lower,
                'upper': gpd_var_ci.upper,
                'relative_width_pct': gpd_var_ci.relative_width
            }

        # Summary statistics
        results['summary'] = {
            'mean_relative_width_pct': np.mean([
                v['relative_width_pct']
                for v in results['confidence_intervals'].values()
                if not np.isinf(v['relative_width_pct'])
            ]),
            'max_relative_width_pct': np.max([
                v['relative_width_pct']
                for v in results['confidence_intervals'].values()
                if not np.isinf(v['relative_width_pct'])
            ]),
            'estimation_quality': self._assess_quality(results['confidence_intervals'])
        }

        return results

    def _assess_quality(self, cis: Dict) -> str:
        """Assess overall estimation quality based on CI widths."""
        avg_width = np.mean([
            v['relative_width_pct']
            for v in cis.values()
            if not np.isinf(v['relative_width_pct'])
        ])

        if avg_width < 20:
            return "Excellent - narrow confidence intervals indicate reliable estimates"
        elif avg_width < 40:
            return "Good - moderate uncertainty, estimates are reasonably reliable"
        elif avg_width < 60:
            return "Fair - substantial uncertainty, interpret with caution"
        else:
            return "Poor - wide confidence intervals suggest more data needed"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quantify_risk_uncertainty(losses: np.ndarray,
                              n_bootstrap: int = 500,
                              include_gpd: bool = True) -> Dict:
    """
    Convenience function for comprehensive risk uncertainty analysis.

    Args:
        losses: Array of loss observations
        n_bootstrap: Number of bootstrap samples
        include_gpd: Whether to include GPD-based metrics

    Returns:
        Dictionary with all uncertainty results
    """
    analyzer = RiskMetricUncertainty(losses, n_bootstrap)
    return analyzer.comprehensive_uncertainty_analysis(include_gpd)


def format_uncertainty_report(results: Dict) -> str:
    """
    Format uncertainty results as a readable report.

    Args:
        results: Output from quantify_risk_uncertainty

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        "UNCERTAINTY QUANTIFICATION REPORT",
        "=" * 70,
        f"\nData: {results['n_observations']} observations",
        f"Bootstrap samples: {results['n_bootstrap']}",
        f"\nEstimation Quality: {results['summary']['estimation_quality']}",
        f"Average CI Width: {results['summary']['mean_relative_width_pct']:.1f}%",
        "\n" + "-" * 70,
        "CONFIDENCE INTERVALS (95%)",
        "-" * 70,
    ]

    for metric, ci in results['confidence_intervals'].items():
        lines.append(
            f"  {metric:15s}: ${ci['estimate']:>15,.0f}  "
            f"[{ci['lower']:>12,.0f}, {ci['upper']:>12,.0f}]  "
            f"(±{ci['relative_width_pct']/2:.1f}%)"
        )

    if 'gpd_parameters' in results:
        lines.extend([
            "\n" + "-" * 70,
            "GPD PARAMETERS",
            "-" * 70,
            f"  Shape (ξ): {results['gpd_parameters']['shape']['estimate']:.4f}  "
            f"[{results['gpd_parameters']['shape']['lower']:.4f}, "
            f"{results['gpd_parameters']['shape']['upper']:.4f}]",
            f"  Scale (σ): ${results['gpd_parameters']['scale']['estimate']:,.0f}  "
            f"[${results['gpd_parameters']['scale']['lower']:,.0f}, "
            f"${results['gpd_parameters']['scale']['upper']:,.0f}]"
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    from scipy.stats import lognorm

    np.random.seed(42)
    losses = lognorm.rvs(s=2, scale=np.exp(14), size=5000)

    print("Running uncertainty analysis...")
    results = quantify_risk_uncertainty(losses, n_bootstrap=200, include_gpd=True)
    print(format_uncertainty_report(results))

