"""
Extreme Value Theory (EVT) Module for CATIA

Implements Generalized Pareto Distribution (GPD) for robust tail risk modeling.
Uses Peaks Over Threshold (POT) approach for extreme loss estimation.

Key concepts:
- GPD is theoretically justified for modeling exceedances over high thresholds
- Pickands-Balkema-de Haan theorem: exceedances follow GPD asymptotically
- More accurate VaR/TVaR estimates for rare events (100+ year return periods)
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize

from catia.config import LOGGING_CONFIG, RISK_METRICS

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@dataclass
class GPDFitResult:
    """Results from GPD fitting."""
    shape: float  # xi (shape parameter): xi > 0 heavy tail, xi < 0 bounded, xi = 0 exponential
    scale: float  # sigma (scale parameter)
    threshold: float  # u (threshold)
    n_exceedances: int  # Number of observations above threshold
    n_total: int  # Total number of observations
    exceedance_rate: float  # Proportion of data above threshold
    method: str  # Fitting method used
    convergence: bool  # Whether optimization converged


class ExtremeValueAnalyzer:
    """
    Extreme Value Theory analyzer using Generalized Pareto Distribution.
    
    The GPD is used to model the distribution of exceedances over a threshold.
    CDF: F(x) = 1 - (1 + xi * x / sigma)^(-1/xi) for xi != 0
         F(x) = 1 - exp(-x / sigma) for xi = 0
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize EVT analyzer.
        
        Args:
            data: Array of loss observations
        """
        self.data = np.asarray(data)
        self.n = len(self.data)
        self.gpd_fit: Optional[GPDFitResult] = None
        logger.info(f"ExtremeValueAnalyzer initialized with {self.n} observations")
    
    def select_threshold(self, method: str = "percentile", 
                         percentile: float = 90.0) -> float:
        """
        Select threshold for POT analysis.
        
        Args:
            method: Selection method ('percentile', 'mean_residual', 'sqrt_n')
            percentile: Percentile for threshold (default 90th)
        
        Returns:
            Selected threshold value
        """
        if method == "percentile":
            threshold = np.percentile(self.data, percentile)
        elif method == "sqrt_n":
            # Use top sqrt(n) observations (Hill estimator rule of thumb)
            k = int(np.sqrt(self.n))
            sorted_data = np.sort(self.data)[::-1]
            threshold = sorted_data[k] if k < self.n else sorted_data[-1]
        elif method == "mean_residual":
            # Mean residual life plot approach - find stability region
            threshold = np.percentile(self.data, 90)  # Fallback to 90th
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        logger.info(f"Selected threshold: ${threshold:,.0f} (method={method})")
        return threshold
    
    def fit_gpd(self, threshold: Optional[float] = None,
                method: str = "mle") -> GPDFitResult:
        """
        Fit Generalized Pareto Distribution to exceedances.
        
        Args:
            threshold: Threshold value (auto-selected if None)
            method: Fitting method ('mle', 'mom', 'pwm')
        
        Returns:
            GPDFitResult with fitted parameters
        """
        if threshold is None:
            threshold = self.select_threshold()
        
        # Get exceedances
        exceedances = self.data[self.data > threshold] - threshold
        n_exceed = len(exceedances)
        
        if n_exceed < 10:
            logger.warning(f"Only {n_exceed} exceedances - results may be unreliable")
        
        if method == "mle":
            shape, scale, convergence = self._fit_gpd_mle(exceedances)
        elif method == "mom":
            shape, scale, convergence = self._fit_gpd_mom(exceedances)
        elif method == "pwm":
            shape, scale, convergence = self._fit_gpd_pwm(exceedances)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        self.gpd_fit = GPDFitResult(
            shape=shape,
            scale=scale,
            threshold=threshold,
            n_exceedances=n_exceed,
            n_total=self.n,
            exceedance_rate=n_exceed / self.n,
            method=method,
            convergence=convergence
        )
        
        logger.info(f"GPD fit: shape(ξ)={shape:.4f}, scale(σ)=${scale:,.0f}")
        logger.info(f"  Exceedances: {n_exceed}/{self.n} ({100*n_exceed/self.n:.1f}%)")
        
        return self.gpd_fit
    
    def _fit_gpd_mle(self, exceedances: np.ndarray) -> Tuple[float, float, bool]:
        """Maximum Likelihood Estimation for GPD."""
        def neg_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            n = len(exceedances)
            if abs(xi) < 1e-10:  # Exponential case
                return n * np.log(sigma) + np.sum(exceedances) / sigma
            else:
                z = 1 + xi * exceedances / sigma
                if np.any(z <= 0):
                    return 1e10
                return n * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(z))
        
        # Initial estimates using method of moments
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        xi_init = 0.5 * ((mean_exc**2 / var_exc) - 1)
        sigma_init = mean_exc * (1 - xi_init)
        
        result = minimize(neg_log_likelihood, [xi_init, sigma_init],
                         method='Nelder-Mead', options={'maxiter': 1000})

        return result.x[0], result.x[1], result.success

    def _fit_gpd_mom(self, exceedances: np.ndarray) -> Tuple[float, float, bool]:
        """Method of Moments estimation for GPD."""
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)

        # MoM estimators
        xi = 0.5 * ((mean_exc**2 / var_exc) - 1)
        sigma = mean_exc * (1 - xi)

        return xi, sigma, True

    def _fit_gpd_pwm(self, exceedances: np.ndarray) -> Tuple[float, float, bool]:
        """Probability Weighted Moments estimation for GPD."""
        n = len(exceedances)
        sorted_exc = np.sort(exceedances)

        # PWM estimators
        b0 = np.mean(sorted_exc)
        b1 = np.sum([(i / (n - 1)) * sorted_exc[i] for i in range(n)]) / n

        xi = 2 - b0 / (b0 - 2 * b1)
        sigma = 2 * b0 * b1 / (b0 - 2 * b1)

        return xi, sigma, True

    def gpd_var(self, confidence: float = 0.95) -> float:
        """
        Calculate GPD-based Value-at-Risk.

        VaR_p = u + (sigma/xi) * [(n/Nu * (1-p))^(-xi) - 1]

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            VaR estimate
        """
        if self.gpd_fit is None:
            self.fit_gpd()

        xi = self.gpd_fit.shape
        sigma = self.gpd_fit.scale
        u = self.gpd_fit.threshold
        zeta_u = self.gpd_fit.exceedance_rate

        p = confidence

        if abs(xi) < 1e-10:
            # Exponential case
            var = u + sigma * np.log(zeta_u / (1 - p))
        else:
            var = u + (sigma / xi) * ((zeta_u / (1 - p))**xi - 1)

        return var

    def gpd_tvar(self, confidence: float = 0.95) -> float:
        """
        Calculate GPD-based Tail Value-at-Risk (Expected Shortfall).

        TVaR_p = VaR_p / (1-xi) + (sigma - xi*u) / (1-xi)

        Args:
            confidence: Confidence level

        Returns:
            TVaR estimate
        """
        if self.gpd_fit is None:
            self.fit_gpd()

        xi = self.gpd_fit.shape
        sigma = self.gpd_fit.scale
        var = self.gpd_var(confidence)

        if xi >= 1:
            logger.warning("TVaR undefined for xi >= 1 (infinite mean)")
            return np.inf

        tvar = var / (1 - xi) + (sigma - xi * self.gpd_fit.threshold) / (1 - xi)

        return tvar

    def gpd_return_level(self, return_period: float) -> float:
        """
        Calculate loss level for a given return period.

        Args:
            return_period: Return period in years (e.g., 100 for 1-in-100 year)

        Returns:
            Expected loss level
        """
        if self.gpd_fit is None:
            self.fit_gpd()

        xi = self.gpd_fit.shape
        sigma = self.gpd_fit.scale
        u = self.gpd_fit.threshold
        zeta_u = self.gpd_fit.exceedance_rate

        # Probability of exceedance for return period
        p_exceed = 1 / return_period

        if abs(xi) < 1e-10:
            return_level = u + sigma * np.log(zeta_u / p_exceed)
        else:
            return_level = u + (sigma / xi) * ((zeta_u / p_exceed)**xi - 1)

        return return_level

    def calculate_all_evt_metrics(self) -> Dict:
        """
        Calculate comprehensive EVT-based risk metrics.

        Returns:
            Dictionary with all EVT metrics
        """
        if self.gpd_fit is None:
            self.fit_gpd()

        # Calculate confidence intervals for different levels
        confidence_levels = [0.90, 0.95, 0.99, 0.995]
        var_estimates = {f"var_{int(c*100)}": self.gpd_var(c) for c in confidence_levels}
        tvar_estimates = {f"tvar_{int(c*100)}": self.gpd_tvar(c) for c in confidence_levels}

        # Calculate return period levels
        return_periods = RISK_METRICS["return_periods"]
        return_levels = {f"{rp}_year": self.gpd_return_level(rp) for rp in return_periods}

        # Compare with empirical estimates
        empirical_var_95 = np.percentile(self.data, 95)
        empirical_tvar_95 = np.mean(self.data[self.data >= empirical_var_95])

        return {
            'gpd_parameters': {
                'shape_xi': self.gpd_fit.shape,
                'scale_sigma': self.gpd_fit.scale,
                'threshold': self.gpd_fit.threshold,
                'exceedance_rate': self.gpd_fit.exceedance_rate,
                'tail_type': self._interpret_shape(self.gpd_fit.shape)
            },
            'var_estimates': var_estimates,
            'tvar_estimates': tvar_estimates,
            'return_period_levels': return_levels,
            'comparison': {
                'empirical_var_95': empirical_var_95,
                'gpd_var_95': var_estimates['var_95'],
                'var_difference_pct': (var_estimates['var_95'] - empirical_var_95) / empirical_var_95 * 100,
                'empirical_tvar_95': empirical_tvar_95,
                'gpd_tvar_95': tvar_estimates['tvar_95'],
                'tvar_difference_pct': (tvar_estimates['tvar_95'] - empirical_tvar_95) / empirical_tvar_95 * 100
            },
            'fit_quality': {
                'method': self.gpd_fit.method,
                'convergence': self.gpd_fit.convergence,
                'n_exceedances': self.gpd_fit.n_exceedances,
                'n_total': self.gpd_fit.n_total
            }
        }

    @staticmethod
    def _interpret_shape(xi: float) -> str:
        """Interpret the shape parameter."""
        if xi > 0.3:
            return "Heavy tail (Fréchet-type) - very high extreme risk"
        elif xi > 0:
            return "Heavy tail - elevated extreme risk"
        elif abs(xi) < 0.05:
            return "Exponential tail - moderate extreme risk"
        else:
            return "Bounded tail (Weibull-type) - limited extreme risk"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def analyze_tail_risk(losses: np.ndarray, threshold_percentile: float = 90.0) -> Dict:
    """
    Perform complete EVT tail risk analysis.

    Args:
        losses: Array of loss observations
        threshold_percentile: Percentile for threshold selection

    Returns:
        Dictionary with comprehensive EVT analysis
    """
    analyzer = ExtremeValueAnalyzer(losses)
    threshold = analyzer.select_threshold(percentile=threshold_percentile)
    analyzer.fit_gpd(threshold=threshold, method="mle")

    return analyzer.calculate_all_evt_metrics()


def compare_risk_methods(losses: np.ndarray) -> Dict:
    """
    Compare empirical vs EVT-based risk metrics.

    Args:
        losses: Array of loss observations

    Returns:
        Comparison of methods for different return periods
    """
    analyzer = ExtremeValueAnalyzer(losses)
    analyzer.fit_gpd()

    return_periods = [10, 25, 50, 100, 250, 500, 1000]
    comparison = []

    for rp in return_periods:
        percentile = (1 - 1/rp) * 100
        empirical = np.percentile(losses, percentile) if percentile <= 100 else np.nan
        gpd = analyzer.gpd_return_level(rp)

        comparison.append({
            'return_period': rp,
            'empirical': empirical,
            'gpd': gpd,
            'difference_pct': ((gpd - empirical) / empirical * 100) if not np.isnan(empirical) else np.nan
        })

    return {
        'comparison': comparison,
        'gpd_parameters': {
            'shape': analyzer.gpd_fit.shape,
            'scale': analyzer.gpd_fit.scale,
            'threshold': analyzer.gpd_fit.threshold
        },
        'recommendation': _get_recommendation(analyzer.gpd_fit.shape)
    }


def _get_recommendation(shape: float) -> str:
    """Get recommendation based on tail behavior."""
    if shape > 0.2:
        return ("Heavy-tailed distribution detected. EVT/GPD estimates should be used "
                "for return periods beyond 50 years. Empirical estimates will underestimate risk.")
    elif shape > 0:
        return ("Moderately heavy tail. EVT improves accuracy for 100+ year return periods.")
    else:
        return ("Light or bounded tail. Empirical estimates may be adequate for most use cases.")


if __name__ == "__main__":
    # Demo with simulated catastrophe losses
    np.random.seed(42)

    # Simulate heavy-tailed losses (typical for CAT modeling)
    from scipy.stats import lognorm, pareto

    n_samples = 10000
    base_losses = lognorm.rvs(s=2, scale=np.exp(14), size=n_samples)

    print("=" * 70)
    print("Extreme Value Theory (EVT) Analysis Demo")
    print("=" * 70)

    # Analyze
    results = analyze_tail_risk(base_losses)

    print(f"\nGPD Parameters:")
    print(f"  Shape (ξ): {results['gpd_parameters']['shape_xi']:.4f}")
    print(f"  Scale (σ): ${results['gpd_parameters']['scale_sigma']:,.0f}")
    print(f"  Threshold: ${results['gpd_parameters']['threshold']:,.0f}")
    print(f"  Tail Type: {results['gpd_parameters']['tail_type']}")

    print(f"\nVaR Estimates:")
    for level, value in results['var_estimates'].items():
        print(f"  {level}: ${value:,.0f}")

    print(f"\nTVaR Estimates:")
    for level, value in results['tvar_estimates'].items():
        print(f"  {level}: ${value:,.0f}")

    print(f"\nReturn Period Levels:")
    for period, value in results['return_period_levels'].items():
        print(f"  {period}: ${value:,.0f}")

    print(f"\nComparison (Empirical vs GPD at 95%):")
    print(f"  Empirical VaR: ${results['comparison']['empirical_var_95']:,.0f}")
    print(f"  GPD VaR: ${results['comparison']['gpd_var_95']:,.0f}")
    print(f"  Difference: {results['comparison']['var_difference_pct']:.1f}%")

