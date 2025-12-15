"""
Peril Correlation Module for CATIA

Models dependencies between different catastrophe perils using copulas.
Captures the fact that some perils are correlated (e.g., hurricanes cause floods)
while others are independent or even negatively correlated.

Key concepts:
- Copulas: Separate marginal distributions from dependency structure
- Gaussian Copula: Models linear correlations (symmetric tail dependence)
- Clayton Copula: Lower tail dependence (joint extreme lows)
- Gumbel Copula: Upper tail dependence (joint extreme highs - common in CAT)
- t-Copula: Heavy tails with symmetric tail dependence

Physical relationships modeled:
- Hurricane → Flood: Strong positive correlation (storm surge, rainfall)
- Drought → Wildfire: Strong positive correlation (dry conditions)
- Earthquake → Tsunami: Conditional (offshore quakes only)
- Hurricane ↔ Earthquake: Near zero (independent processes)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln
import warnings

from catia.config import LOGGING_CONFIG, PERIL_CONFIG

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


# Default correlation matrix based on physical relationships
# Rows/columns: hurricane, flood, wildfire, earthquake
DEFAULT_PERIL_CORRELATIONS = {
    ('hurricane', 'hurricane'): 1.0,
    ('hurricane', 'flood'): 0.65,      # Hurricanes cause floods
    ('hurricane', 'wildfire'): -0.15,  # Wet conditions reduce fire risk
    ('hurricane', 'earthquake'): 0.0,  # Independent processes
    
    ('flood', 'hurricane'): 0.65,
    ('flood', 'flood'): 1.0,
    ('flood', 'wildfire'): -0.20,      # Wet vs dry conditions
    ('flood', 'earthquake'): 0.05,     # Slight: dam failures after quakes
    
    ('wildfire', 'hurricane'): -0.15,
    ('wildfire', 'flood'): -0.20,
    ('wildfire', 'wildfire'): 1.0,
    ('wildfire', 'earthquake'): 0.10,  # Fires from ruptured gas lines
    
    ('earthquake', 'hurricane'): 0.0,
    ('earthquake', 'flood'): 0.05,
    ('earthquake', 'wildfire'): 0.10,
    ('earthquake', 'earthquake'): 1.0,
}


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    perils: List[str]
    correlation_matrix: np.ndarray
    copula_type: str
    tail_dependence: Dict[str, float]
    samples: np.ndarray  # Correlated uniform samples


class CopulaBase:
    """Base class for copula implementations."""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def sample(self, n: int) -> np.ndarray:
        """Generate n samples from the copula."""
        raise NotImplementedError
    
    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Compute copula density."""
        raise NotImplementedError


class GaussianCopula(CopulaBase):
    """
    Gaussian (Normal) Copula.
    
    Models linear correlation between variables. Has no tail dependence,
    meaning extreme events are asymptotically independent.
    """
    
    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialize Gaussian copula.
        
        Args:
            correlation_matrix: Correlation matrix (must be positive definite)
        """
        super().__init__(correlation_matrix.shape[0])
        self.corr = correlation_matrix
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(correlation_matrix)
        if np.any(eigvals <= 0):
            logger.warning("Correlation matrix not positive definite, applying correction")
            self.corr = self._nearest_positive_definite(correlation_matrix)
        
        self.cholesky = np.linalg.cholesky(self.corr)
        logger.info(f"GaussianCopula initialized with {self.dim} dimensions")
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix."""
        B = (A + A.T) / 2
        eigval, eigvec = np.linalg.eigh(B)
        eigval = np.maximum(eigval, 1e-8)
        return eigvec @ np.diag(eigval) @ eigvec.T
    
    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate samples from Gaussian copula.
        
        Args:
            n: Number of samples
            random_state: Random seed
        
        Returns:
            Array of shape (n, dim) with uniform marginals
        """
        rng = np.random.default_rng(random_state)
        z = rng.standard_normal((n, self.dim))
        correlated_z = z @ self.cholesky.T
        # Transform to uniform using normal CDF
        u = stats.norm.cdf(correlated_z)
        return u
    
    @property
    def tail_dependence(self) -> Dict[str, float]:
        """Gaussian copula has no tail dependence."""
        return {'lower': 0.0, 'upper': 0.0}


class TCopula(CopulaBase):
    """
    Student-t Copula.

    Has symmetric tail dependence - extreme events tend to occur together.
    More appropriate for catastrophe modeling than Gaussian.
    """

    def __init__(self, correlation_matrix: np.ndarray, df: float = 4.0):
        """
        Initialize t-copula.

        Args:
            correlation_matrix: Correlation matrix
            df: Degrees of freedom (lower = heavier tails)
        """
        super().__init__(correlation_matrix.shape[0])
        self.corr = correlation_matrix
        self.df = df

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(correlation_matrix)
        if np.any(eigvals <= 0):
            self.corr = GaussianCopula._nearest_positive_definite(
                GaussianCopula(correlation_matrix), correlation_matrix)

        self.cholesky = np.linalg.cholesky(self.corr)
        logger.info(f"TCopula initialized: dim={self.dim}, df={df}")

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from t-copula."""
        rng = np.random.default_rng(random_state)

        # Generate multivariate t
        z = rng.standard_normal((n, self.dim))
        correlated_z = z @ self.cholesky.T

        # Scale by chi-squared
        s = rng.chisquare(self.df, n)
        t_samples = correlated_z / np.sqrt(s / self.df)[:, np.newaxis]

        # Transform to uniform using t CDF
        u = stats.t.cdf(t_samples, self.df)
        return u

    @property
    def tail_dependence(self) -> Dict[str, float]:
        """Calculate tail dependence coefficient for t-copula."""
        # Average correlation for tail dependence calculation
        avg_rho = (np.sum(self.corr) - self.dim) / (self.dim * (self.dim - 1))

        # Tail dependence formula for t-copula
        t_val = np.sqrt((self.df + 1) * (1 - avg_rho) / (1 + avg_rho))
        lambda_tail = 2 * stats.t.cdf(-t_val, self.df + 1)

        return {'lower': lambda_tail, 'upper': lambda_tail}


class ClaytonCopula(CopulaBase):
    """
    Clayton Copula (Archimedean).

    Has lower tail dependence - models joint extreme lows.
    Less common for catastrophe losses (we care about high losses).
    """

    def __init__(self, theta: float = 2.0):
        """
        Initialize Clayton copula (bivariate).

        Args:
            theta: Dependence parameter (theta > 0, higher = more dependence)
        """
        super().__init__(2)
        if theta <= 0:
            raise ValueError("Clayton theta must be > 0")
        self.theta = theta
        logger.info(f"ClaytonCopula initialized: theta={theta}")

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Clayton copula."""
        rng = np.random.default_rng(random_state)

        # Use conditional method
        u1 = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)

        # Inverse of conditional CDF
        u2 = ((w ** (-self.theta / (1 + self.theta)) - 1) *
              u1 ** (-self.theta) + 1) ** (-1 / self.theta)

        return np.column_stack([u1, u2])

    @property
    def tail_dependence(self) -> Dict[str, float]:
        """Clayton has lower tail dependence only."""
        lambda_lower = 2 ** (-1 / self.theta)
        return {'lower': lambda_lower, 'upper': 0.0}


class GumbelCopula(CopulaBase):
    """
    Gumbel Copula (Archimedean).

    Has upper tail dependence - models joint extreme highs.
    IDEAL for catastrophe modeling where we care about simultaneous large losses.
    """

    def __init__(self, theta: float = 2.0):
        """
        Initialize Gumbel copula (bivariate).

        Args:
            theta: Dependence parameter (theta >= 1, higher = more dependence)
        """
        super().__init__(2)
        if theta < 1:
            raise ValueError("Gumbel theta must be >= 1")
        self.theta = theta
        logger.info(f"GumbelCopula initialized: theta={theta}")

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from Gumbel copula using Marshall-Olkin method."""
        rng = np.random.default_rng(random_state)

        # Generate stable random variable with alpha = 1/theta
        alpha = 1 / self.theta

        # Approximate stable distribution sampling
        v = rng.uniform(0, np.pi, n)
        w = rng.exponential(1, n)

        s = (np.sin(alpha * v) / (np.cos(v) ** (1/alpha))) * \
            (np.cos(v - alpha * v) / w) ** ((1 - alpha) / alpha)

        # Generate uniforms and transform
        e1 = rng.exponential(1, n)
        e2 = rng.exponential(1, n)

        u1 = np.exp(-(e1 / s) ** (1 / self.theta))
        u2 = np.exp(-(e2 / s) ** (1 / self.theta))

        return np.column_stack([u1, u2])

    @property
    def tail_dependence(self) -> Dict[str, float]:
        """Gumbel has upper tail dependence only."""
        lambda_upper = 2 - 2 ** (1 / self.theta)
        return {'lower': 0.0, 'upper': lambda_upper}


# ============================================================================
# PERIL CORRELATION SIMULATOR
# ============================================================================

class PerilCorrelationSimulator:
    """
    Simulates correlated peril occurrences and losses.

    Uses copulas to model dependency structure between perils,
    then applies inverse CDF to get correlated losses from each
    peril's marginal distribution.
    """

    def __init__(self, perils: List[str],
                 copula_type: str = "t",
                 correlations: Optional[Dict] = None):
        """
        Initialize peril correlation simulator.

        Args:
            perils: List of peril types
            copula_type: 'gaussian', 't', 'gumbel', or 'clayton'
            correlations: Custom correlation dict (uses defaults if None)
        """
        self.perils = perils
        self.n_perils = len(perils)
        self.copula_type = copula_type

        # Build correlation matrix
        corr_dict = correlations or DEFAULT_PERIL_CORRELATIONS
        self.corr_matrix = self._build_correlation_matrix(corr_dict)

        # Initialize copula
        self.copula = self._create_copula()

        logger.info(f"PerilCorrelationSimulator initialized: {perils}")
        logger.info(f"Copula type: {copula_type}, tail dependence: {self.copula.tail_dependence}")

    def _build_correlation_matrix(self, corr_dict: Dict) -> np.ndarray:
        """Build correlation matrix from dictionary."""
        matrix = np.eye(self.n_perils)

        for i, p1 in enumerate(self.perils):
            for j, p2 in enumerate(self.perils):
                if i != j:
                    key = (p1, p2)
                    if key in corr_dict:
                        matrix[i, j] = corr_dict[key]
                    else:
                        # Default to low correlation if not specified
                        matrix[i, j] = 0.1

        return matrix

    def _create_copula(self) -> CopulaBase:
        """Create appropriate copula based on type."""
        if self.copula_type == "gaussian":
            return GaussianCopula(self.corr_matrix)
        elif self.copula_type == "t":
            return TCopula(self.corr_matrix, df=4)
        elif self.copula_type == "gumbel" and self.n_perils == 2:
            # Gumbel only bivariate - use average correlation
            avg_corr = self.corr_matrix[0, 1]
            theta = max(1.0, 1 / (1 - abs(avg_corr)))
            return GumbelCopula(theta)
        elif self.copula_type == "clayton" and self.n_perils == 2:
            avg_corr = self.corr_matrix[0, 1]
            theta = max(0.1, 2 * avg_corr / (1 - avg_corr)) if avg_corr < 1 else 10
            return ClaytonCopula(theta)
        else:
            # Default to t-copula for multivariate
            logger.warning(f"Using t-copula (requested {self.copula_type} not available for {self.n_perils}D)")
            return TCopula(self.corr_matrix, df=4)

    def generate_correlated_uniforms(self, n: int,
                                     random_state: Optional[int] = None
                                     ) -> np.ndarray:
        """
        Generate correlated uniform samples.

        Args:
            n: Number of samples
            random_state: Random seed

        Returns:
            Array of shape (n, n_perils) with correlated uniforms
        """
        return self.copula.sample(n, random_state)

    def simulate_correlated_losses(self, n: int,
                                   marginal_params: Dict[str, Dict],
                                   random_state: Optional[int] = None
                                   ) -> Dict[str, np.ndarray]:
        """
        Simulate correlated losses for each peril.

        Uses copula for dependency, then applies inverse CDF
        of each peril's loss distribution.

        Args:
            n: Number of simulations
            marginal_params: Dict with peril params (mean, std, distribution)
            random_state: Random seed

        Returns:
            Dict mapping peril name to loss array
        """
        # Generate correlated uniforms
        u = self.generate_correlated_uniforms(n, random_state)

        losses = {}
        for i, peril in enumerate(self.perils):
            params = marginal_params.get(peril, {})
            mean = params.get('mean', 10_000_000)
            std = params.get('std', 20_000_000)
            dist = params.get('distribution', 'lognormal')

            # Transform uniform to loss using inverse CDF
            if dist == 'lognormal':
                # Parameterize lognormal from mean/std
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                mu = np.log(mean) - sigma**2/2
                losses[peril] = stats.lognorm.ppf(u[:, i], s=sigma, scale=np.exp(mu))
            elif dist == 'pareto':
                alpha = params.get('alpha', 2.0)
                scale = mean * (alpha - 1) / alpha if alpha > 1 else mean
                losses[peril] = stats.pareto.ppf(u[:, i], b=alpha, scale=scale)
            else:
                # Default to lognormal
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                mu = np.log(mean) - sigma**2/2
                losses[peril] = stats.lognorm.ppf(u[:, i], s=sigma, scale=np.exp(mu))

        return losses

    def get_correlation_summary(self) -> Dict:
        """Get summary of correlation structure."""
        return {
            'perils': self.perils,
            'copula_type': self.copula_type,
            'correlation_matrix': self.corr_matrix.tolist(),
            'tail_dependence': self.copula.tail_dependence,
            'pairwise_correlations': {
                f"{self.perils[i]}-{self.perils[j]}": self.corr_matrix[i, j]
                for i in range(self.n_perils)
                for j in range(i+1, self.n_perils)
            }
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_correlated_perils(perils: List[str],
                               n_simulations: int = 10000,
                               copula_type: str = "t",
                               random_state: Optional[int] = None) -> Dict:
    """
    Convenience function to simulate correlated peril losses.

    Args:
        perils: List of peril types
        n_simulations: Number of simulations
        copula_type: Type of copula ('gaussian', 't', 'gumbel', 'clayton')
        random_state: Random seed

    Returns:
        Dictionary with correlated losses and correlation info
    """
    # Build marginal parameters from PERIL_CONFIG
    marginal_params = {}
    for peril in perils:
        if peril in PERIL_CONFIG:
            cfg = PERIL_CONFIG[peril]
            sev = cfg.get('severity_params', {'mu': 15, 'sigma': 2})
            # Convert lognormal mu/sigma to mean/std
            mu, sigma = sev.get('mu', 15), sev.get('sigma', 2)
            mean = np.exp(mu + sigma**2 / 2)
            std = mean * np.sqrt(np.exp(sigma**2) - 1)
            marginal_params[peril] = {
                'mean': mean,
                'std': std,
                'distribution': 'lognormal'
            }

    simulator = PerilCorrelationSimulator(perils, copula_type)
    losses = simulator.simulate_correlated_losses(
        n_simulations, marginal_params, random_state)

    # Calculate empirical correlations
    loss_matrix = np.column_stack([losses[p] for p in perils])
    empirical_corr = np.corrcoef(loss_matrix.T)

    return {
        'losses': losses,
        'correlation_summary': simulator.get_correlation_summary(),
        'empirical_correlation': empirical_corr.tolist(),
        'aggregate_loss': np.sum(loss_matrix, axis=1)
    }


def compare_independent_vs_correlated(perils: List[str],
                                      n_simulations: int = 10000
                                      ) -> Dict:
    """
    Compare risk metrics between independent and correlated simulations.

    Demonstrates why correlation matters for tail risk.
    """
    # Independent simulation (Gaussian with identity matrix)
    indep_sim = PerilCorrelationSimulator(
        perils, copula_type="gaussian",
        correlations={(p1, p2): 1.0 if p1 == p2 else 0.0
                      for p1 in perils for p2 in perils}
    )

    # Correlated simulation (t-copula with default correlations)
    corr_sim = PerilCorrelationSimulator(perils, copula_type="t")

    marginal_params = {}
    for peril in perils:
        if peril in PERIL_CONFIG:
            cfg = PERIL_CONFIG[peril]
            sev = cfg.get('severity_params', {'mu': 15, 'sigma': 2})
            mu, sigma = sev.get('mu', 15), sev.get('sigma', 2)
            mean = np.exp(mu + sigma**2 / 2)
            std = mean * np.sqrt(np.exp(sigma**2) - 1)
            marginal_params[peril] = {
                'mean': mean,
                'std': std,
                'distribution': 'lognormal'
            }

    # Simulate both
    indep_losses = indep_sim.simulate_correlated_losses(n_simulations, marginal_params, 42)
    corr_losses = corr_sim.simulate_correlated_losses(n_simulations, marginal_params, 42)

    # Aggregate
    indep_agg = sum(indep_losses.values())
    corr_agg = sum(corr_losses.values())

    # Risk metrics
    var_95_indep = np.percentile(indep_agg, 95)
    var_95_corr = np.percentile(corr_agg, 95)
    var_99_indep = np.percentile(indep_agg, 99)
    var_99_corr = np.percentile(corr_agg, 99)

    return {
        'independent': {
            'var_95': var_95_indep,
            'var_99': var_99_indep,
            'mean': np.mean(indep_agg),
            'max': np.max(indep_agg)
        },
        'correlated': {
            'var_95': var_95_corr,
            'var_99': var_99_corr,
            'mean': np.mean(corr_agg),
            'max': np.max(corr_agg),
            'tail_dependence': corr_sim.copula.tail_dependence
        },
        'impact': {
            'var_95_increase_pct': (var_95_corr - var_95_indep) / var_95_indep * 100,
            'var_99_increase_pct': (var_99_corr - var_99_indep) / var_99_indep * 100,
            'max_increase_pct': (np.max(corr_agg) - np.max(indep_agg)) / np.max(indep_agg) * 100
        }
    }

