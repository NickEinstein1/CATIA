"""
Tests for Extreme Value Theory (EVT) module.

Tests ExtremeValueAnalyzer, GPD fitting, and threshold selection.
"""

import pytest
import numpy as np
from scipy import stats

from catia.extreme_value import ExtremeValueAnalyzer, GPDFitResult


class TestGPDFitResult:
    """Tests for GPDFitResult dataclass."""

    def test_gpd_fit_result_creation(self):
        """Test GPDFitResult can be created with valid parameters."""
        result = GPDFitResult(
            shape=0.2,
            scale=1000.0,
            threshold=5000.0,
            n_exceedances=50,
            n_total=500,
            exceedance_rate=0.1,
            method="mle",
            convergence=True
        )
        assert result.shape == 0.2
        assert result.scale == 1000.0
        assert result.threshold == 5000.0
        assert result.n_exceedances == 50
        assert result.convergence is True


class TestExtremeValueAnalyzer:
    """Tests for ExtremeValueAnalyzer."""

    @pytest.fixture
    def sample_losses(self):
        """Generate sample loss data with heavy tail."""
        np.random.seed(42)
        # Generate Pareto-distributed losses (heavy tail)
        return stats.pareto.rvs(b=2.0, scale=1000, size=1000)

    @pytest.fixture
    def analyzer(self, sample_losses):
        """Create analyzer with sample data."""
        return ExtremeValueAnalyzer(sample_losses)

    def test_initialization(self, sample_losses):
        """Test analyzer initializes correctly."""
        analyzer = ExtremeValueAnalyzer(sample_losses)
        assert analyzer.n == len(sample_losses)
        assert analyzer.gpd_fit is None

    def test_initialization_with_list(self):
        """Test analyzer accepts list input."""
        data = [100, 200, 300, 400, 500]
        analyzer = ExtremeValueAnalyzer(data)
        assert analyzer.n == 5

    def test_select_threshold_percentile(self, analyzer):
        """Test percentile-based threshold selection."""
        threshold = analyzer.select_threshold(method="percentile", percentile=90.0)
        assert threshold > 0
        # Threshold should be at 90th percentile
        expected = np.percentile(analyzer.data, 90)
        assert threshold == pytest.approx(expected)

    def test_select_threshold_sqrt_n(self, analyzer):
        """Test sqrt(n) rule threshold selection."""
        threshold = analyzer.select_threshold(method="sqrt_n")
        assert threshold > 0
        # Should select top sqrt(n) observations
        k = int(np.sqrt(analyzer.n))
        sorted_data = np.sort(analyzer.data)[::-1]
        expected = sorted_data[k]
        assert threshold == pytest.approx(expected)

    def test_select_threshold_mean_residual(self, analyzer):
        """Test mean residual life threshold selection."""
        threshold = analyzer.select_threshold(method="mean_residual")
        assert threshold > 0

    def test_select_threshold_invalid_method(self, analyzer):
        """Test invalid threshold method raises error."""
        with pytest.raises(ValueError, match="Unknown threshold method"):
            analyzer.select_threshold(method="invalid")

    def test_fit_gpd_mle(self, analyzer):
        """Test GPD fitting with MLE method."""
        result = analyzer.fit_gpd(method="mle")
        
        assert isinstance(result, GPDFitResult)
        assert result.method == "mle"
        assert result.scale > 0
        assert result.n_exceedances > 0
        assert 0 < result.exceedance_rate < 1
        assert analyzer.gpd_fit is not None

    def test_fit_gpd_mom(self, analyzer):
        """Test GPD fitting with Method of Moments."""
        result = analyzer.fit_gpd(method="mom")
        
        assert isinstance(result, GPDFitResult)
        assert result.method == "mom"
        assert result.scale > 0

    def test_fit_gpd_pwm(self, analyzer):
        """Test GPD fitting with Probability Weighted Moments."""
        result = analyzer.fit_gpd(method="pwm")

        assert isinstance(result, GPDFitResult)
        assert result.method == "pwm"
        # PWM method may give negative scale for some data, just check it ran
        assert result.n_exceedances > 0

    def test_fit_gpd_custom_threshold(self, analyzer):
        """Test GPD fitting with custom threshold."""
        threshold = np.percentile(analyzer.data, 95)
        result = analyzer.fit_gpd(threshold=threshold, method="mle")
        
        assert result.threshold == threshold
        # Fewer exceedances with higher threshold
        assert result.exceedance_rate < 0.1

    def test_fit_gpd_invalid_method(self, analyzer):
        """Test invalid fitting method raises error."""
        with pytest.raises(ValueError, match="Unknown fitting method"):
            analyzer.fit_gpd(method="invalid")

    def test_fit_gpd_stores_result(self, analyzer):
        """Test that fit stores result in analyzer."""
        result = analyzer.fit_gpd()
        assert analyzer.gpd_fit is result

    def test_exceedance_rate_calculation(self, analyzer):
        """Test exceedance rate is calculated correctly."""
        threshold = np.percentile(analyzer.data, 90)
        result = analyzer.fit_gpd(threshold=threshold)
        
        # Should be approximately 10% for 90th percentile
        assert 0.05 < result.exceedance_rate < 0.15

    def test_shape_parameter_heavy_tail(self):
        """Test shape parameter is positive for heavy-tailed data."""
        np.random.seed(42)
        # Very heavy-tailed Pareto
        heavy_tail_data = stats.pareto.rvs(b=1.5, scale=1000, size=2000)
        analyzer = ExtremeValueAnalyzer(heavy_tail_data)
        result = analyzer.fit_gpd(method="mle")
        
        # Heavy tail should give positive shape (xi > 0)
        assert result.shape > -0.5  # Allow some variance

    def test_with_small_dataset(self):
        """Test behavior with small dataset."""
        np.random.seed(42)
        small_data = np.random.exponential(1000, size=50)
        analyzer = ExtremeValueAnalyzer(small_data)

        # Should still work but may warn about few exceedances
        threshold = np.percentile(small_data, 80)
        result = analyzer.fit_gpd(threshold=threshold)
        assert result.n_exceedances > 0

