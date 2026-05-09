"""
Tests for Peril Correlation module.

Tests copulas (Gaussian, t, Clayton, Gumbel) and PerilCorrelationSimulator.
"""

import pytest
import numpy as np

from catia.correlation import (
    GaussianCopula,
    TCopula,
    ClaytonCopula,
    GumbelCopula,
    PerilCorrelationSimulator,
    CorrelationResult,
    DEFAULT_PERIL_CORRELATIONS
)


class TestGaussianCopula:
    """Tests for GaussianCopula."""

    @pytest.fixture
    def correlation_matrix(self):
        """Create sample correlation matrix."""
        return np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

    def test_initialization(self, correlation_matrix):
        """Test Gaussian copula initialization."""
        copula = GaussianCopula(correlation_matrix)
        assert copula.dim == 3
        np.testing.assert_array_almost_equal(copula.corr, correlation_matrix)

    def test_sample_shape(self, correlation_matrix):
        """Test sample output shape."""
        copula = GaussianCopula(correlation_matrix)
        samples = copula.sample(100, random_state=42)
        
        assert samples.shape == (100, 3)

    def test_sample_uniform_marginals(self, correlation_matrix):
        """Test samples have uniform marginals."""
        copula = GaussianCopula(correlation_matrix)
        samples = copula.sample(5000, random_state=42)
        
        # Each marginal should be approximately uniform [0, 1]
        for d in range(3):
            assert samples[:, d].min() > 0
            assert samples[:, d].max() < 1
            # Mean should be close to 0.5
            assert np.mean(samples[:, d]) == pytest.approx(0.5, abs=0.05)

    def test_tail_dependence(self, correlation_matrix):
        """Test Gaussian copula has no tail dependence."""
        copula = GaussianCopula(correlation_matrix)
        td = copula.tail_dependence
        
        assert td['lower'] == 0.0
        assert td['upper'] == 0.0

    def test_reproducibility(self, correlation_matrix):
        """Test sampling is reproducible with same seed."""
        copula = GaussianCopula(correlation_matrix)
        samples1 = copula.sample(50, random_state=42)
        samples2 = copula.sample(50, random_state=42)
        
        np.testing.assert_array_almost_equal(samples1, samples2)


class TestTCopula:
    """Tests for TCopula."""

    @pytest.fixture
    def correlation_matrix(self):
        return np.array([[1.0, 0.6], [0.6, 1.0]])

    def test_initialization(self, correlation_matrix):
        """Test t-copula initialization."""
        copula = TCopula(correlation_matrix, df=4.0)
        assert copula.dim == 2
        assert copula.df == 4.0

    def test_sample_shape(self, correlation_matrix):
        """Test sample output shape."""
        copula = TCopula(correlation_matrix, df=5)
        samples = copula.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)

    def test_tail_dependence_positive(self, correlation_matrix):
        """Test t-copula has positive tail dependence."""
        copula = TCopula(correlation_matrix, df=4)
        td = copula.tail_dependence
        
        # t-copula has symmetric tail dependence
        assert td['lower'] > 0
        assert td['upper'] > 0
        assert td['lower'] == td['upper']

    def test_lower_df_higher_tail_dependence(self, correlation_matrix):
        """Test lower df gives higher tail dependence."""
        copula_low = TCopula(correlation_matrix, df=3)
        copula_high = TCopula(correlation_matrix, df=10)
        
        assert copula_low.tail_dependence['upper'] > copula_high.tail_dependence['upper']


class TestClaytonCopula:
    """Tests for Clayton Copula."""

    def test_initialization(self):
        """Test Clayton copula initialization."""
        copula = ClaytonCopula(theta=2.0)
        assert copula.dim == 2
        assert copula.theta == 2.0

    def test_sample_shape(self):
        """Test sample output shape."""
        copula = ClaytonCopula(theta=2.0)
        samples = copula.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)

    def test_lower_tail_dependence(self):
        """Test Clayton has lower tail dependence only."""
        copula = ClaytonCopula(theta=2.0)
        td = copula.tail_dependence
        
        assert td['lower'] > 0
        assert td['upper'] == 0.0


class TestGumbelCopula:
    """Tests for Gumbel Copula."""

    def test_initialization(self):
        """Test Gumbel copula initialization."""
        copula = GumbelCopula(theta=2.0)
        assert copula.dim == 2
        assert copula.theta == 2.0

    def test_sample_shape(self):
        """Test sample output shape."""
        copula = GumbelCopula(theta=2.0)
        samples = copula.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)

    def test_upper_tail_dependence(self):
        """Test Gumbel has upper tail dependence only."""
        copula = GumbelCopula(theta=2.0)
        td = copula.tail_dependence
        
        assert td['upper'] > 0
        assert td['lower'] == 0.0


class TestPerilCorrelationSimulator:
    """Tests for PerilCorrelationSimulator."""

    @pytest.fixture
    def simulator(self):
        """Create simulator with common perils."""
        return PerilCorrelationSimulator(
            perils=['hurricane', 'flood'],
            copula_type='t'
        )

    def test_initialization(self):
        """Test simulator initialization."""
        sim = PerilCorrelationSimulator(
            perils=['hurricane', 'flood', 'wildfire'],
            copula_type='gaussian'
        )
        assert sim.n_perils == 3
        assert sim.copula_type == 'gaussian'

    def test_correlation_matrix_from_defaults(self, simulator):
        """Test correlation matrix uses default correlations."""
        # Hurricane-flood correlation from defaults
        expected = DEFAULT_PERIL_CORRELATIONS[('hurricane', 'flood')]
        assert simulator.corr_matrix[0, 1] == expected

    def test_generate_correlated_uniforms(self, simulator):
        """Test generating correlated uniforms."""
        samples = simulator.generate_correlated_uniforms(100, random_state=42)
        
        assert samples.shape == (100, 2)
        assert samples.min() >= 0
        assert samples.max() <= 1

