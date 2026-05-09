"""
Tests for Uncertainty Quantification module.

Tests BootstrapAnalyzer, RiskMetricUncertainty, and confidence intervals.
"""

import pytest
import numpy as np

from catia.uncertainty import (
    BootstrapAnalyzer,
    RiskMetricUncertainty,
    ConfidenceInterval,
    UncertaintyResult
)


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_confidence_interval_creation(self):
        """Test ConfidenceInterval can be created."""
        ci = ConfidenceInterval(
            estimate=1000.0,
            lower=800.0,
            upper=1200.0,
            confidence=0.95,
            method="bootstrap_percentile"
        )
        assert ci.estimate == 1000.0
        assert ci.lower == 800.0
        assert ci.upper == 1200.0
        assert ci.confidence == 0.95

    def test_width_property(self):
        """Test width calculation."""
        ci = ConfidenceInterval(
            estimate=1000.0, lower=800.0, upper=1200.0,
            confidence=0.95, method="test"
        )
        assert ci.width == 400.0

    def test_relative_width_property(self):
        """Test relative width calculation."""
        ci = ConfidenceInterval(
            estimate=1000.0, lower=800.0, upper=1200.0,
            confidence=0.95, method="test"
        )
        assert ci.relative_width == 40.0  # (400 / 1000) * 100

    def test_relative_width_zero_estimate(self):
        """Test relative width when estimate is zero."""
        ci = ConfidenceInterval(
            estimate=0.0, lower=-10.0, upper=10.0,
            confidence=0.95, method="test"
        )
        assert ci.relative_width == np.inf


class TestBootstrapAnalyzer:
    """Tests for BootstrapAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample loss data."""
        np.random.seed(42)
        return np.random.exponential(1000, size=500)

    @pytest.fixture
    def analyzer(self, sample_data):
        """Create bootstrap analyzer."""
        return BootstrapAnalyzer(sample_data, n_bootstrap=100, random_state=42)

    def test_initialization(self, sample_data):
        """Test analyzer initializes correctly."""
        analyzer = BootstrapAnalyzer(sample_data, n_bootstrap=500)
        assert analyzer.n == len(sample_data)
        assert analyzer.n_bootstrap == 500

    def test_initialization_with_random_state(self, sample_data):
        """Test reproducibility with random state."""
        analyzer1 = BootstrapAnalyzer(sample_data, n_bootstrap=50, random_state=42)
        analyzer2 = BootstrapAnalyzer(sample_data, n_bootstrap=50, random_state=42)
        
        dist1 = analyzer1.compute_bootstrap_distribution(np.mean)
        dist2 = analyzer2.compute_bootstrap_distribution(np.mean)
        
        np.testing.assert_array_almost_equal(dist1, dist2)

    def test_compute_bootstrap_distribution(self, analyzer):
        """Test bootstrap distribution computation."""
        dist = analyzer.compute_bootstrap_distribution(np.mean)
        
        assert len(dist) == analyzer.n_bootstrap
        assert np.mean(dist) == pytest.approx(np.mean(analyzer.data), rel=0.1)

    def test_confidence_interval_percentile(self, analyzer):
        """Test percentile confidence interval."""
        ci = analyzer.confidence_interval(np.mean, confidence=0.95, method="percentile")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence == 0.95
        assert ci.method == "bootstrap_percentile"
        assert ci.lower < ci.estimate < ci.upper

    def test_confidence_interval_basic(self, analyzer):
        """Test basic bootstrap confidence interval."""
        ci = analyzer.confidence_interval(np.mean, confidence=0.95, method="basic")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "bootstrap_basic"
        assert ci.lower < ci.upper

    def test_confidence_interval_bca(self, analyzer):
        """Test BCa confidence interval."""
        ci = analyzer.confidence_interval(np.mean, confidence=0.95, method="bca")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == "bootstrap_bca"

    def test_invalid_method_raises_error(self, analyzer):
        """Test invalid CI method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.confidence_interval(np.mean, method="invalid")

    def test_custom_statistic(self, analyzer):
        """Test with custom statistic function."""
        def iqr(data):
            return np.percentile(data, 75) - np.percentile(data, 25)
        
        ci = analyzer.confidence_interval(iqr, confidence=0.90)
        assert ci.estimate > 0
        assert ci.confidence == 0.90


class TestRiskMetricUncertainty:
    """Tests for RiskMetricUncertainty."""

    @pytest.fixture
    def sample_losses(self):
        """Generate sample loss data."""
        np.random.seed(42)
        return np.random.exponential(10000, size=1000)

    @pytest.fixture
    def uncertainty(self, sample_losses):
        """Create uncertainty analyzer."""
        return RiskMetricUncertainty(sample_losses, n_bootstrap=100, random_state=42)

    def test_initialization(self, sample_losses):
        """Test initialization."""
        unc = RiskMetricUncertainty(sample_losses, n_bootstrap=200)
        assert len(unc.losses) == len(sample_losses)
        assert unc.n_bootstrap == 200

    def test_var_confidence_interval(self, uncertainty):
        """Test VaR confidence interval."""
        ci = uncertainty.var_confidence_interval(percentile=95.0, confidence=0.95)
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower < ci.estimate < ci.upper
        # VaR should be positive for positive losses
        assert ci.estimate > 0

    def test_tvar_confidence_interval(self, uncertainty):
        """Test TVaR confidence interval."""
        ci = uncertainty.tvar_confidence_interval(percentile=95.0, confidence=0.95)
        
        assert isinstance(ci, ConfidenceInterval)
        # TVaR >= VaR by definition
        var_ci = uncertainty.var_confidence_interval(percentile=95.0)
        assert ci.estimate >= var_ci.estimate * 0.99  # Allow small numerical error

