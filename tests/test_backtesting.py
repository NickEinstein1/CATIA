"""
Tests for Backtesting module.

Tests RollingWindowBacktester, metrics calculation, and calibration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression

from catia.backtesting import (
    RollingWindowBacktester,
    BacktestResult,
    BacktestSummary,
    CalibrationResult,
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_calibration
)


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_creation(self):
        """Test BacktestResult can be created."""
        result = BacktestResult(
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 1, 31),
            n_samples=100,
            predictions=np.array([1.0, 2.0, 3.0]),
            actuals=np.array([1.1, 2.1, 3.1]),
            metrics={'rmse': 0.1}
        )
        assert result.n_samples == 100
        assert result.metrics['rmse'] == 0.1


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_creation(self):
        """Test CalibrationResult can be created."""
        result = CalibrationResult(
            predicted_probs=np.array([0.1, 0.5, 0.9]),
            observed_freqs=np.array([0.15, 0.45, 0.85]),
            bin_counts=np.array([10, 20, 15]),
            calibration_error=0.05,
            max_calibration_error=0.1
        )
        assert result.calibration_error == 0.05


class TestRegressionMetrics:
    """Tests for regression metrics calculation."""

    def test_calculate_regression_metrics(self):
        """Test basic regression metrics."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert 'bias' in metrics
        
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['r2'] > 0.9  # Should be high for near-perfect predictions

    def test_tail_metrics(self):
        """Test tail-specific metrics."""
        np.random.seed(42)
        y_true = np.random.exponential(1000, 100)
        y_pred = y_true * 1.1  # 10% over-prediction
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert 'tail_rmse' in metrics
        assert 'tail_bias' in metrics


class TestClassificationMetrics:
    """Tests for classification metrics calculation."""

    def test_calculate_classification_metrics(self):
        """Test basic classification metrics."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 'brier' in metrics

    def test_without_probabilities(self):
        """Test metrics without probability predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'auc' not in metrics  # No probabilities provided


class TestCalculateCalibration:
    """Tests for calibration calculation."""

    def test_calculate_calibration(self):
        """Test calibration metrics."""
        np.random.seed(42)
        # Well-calibrated probabilities
        y_prob = np.random.uniform(0, 1, 100)
        y_true = (np.random.uniform(0, 1, 100) < y_prob).astype(int)
        
        result = calculate_calibration(y_true, y_prob, n_bins=10)
        
        assert isinstance(result, CalibrationResult)
        assert 0 <= result.calibration_error <= 1
        assert 0 <= result.max_calibration_error <= 1


class TestRollingWindowBacktester:
    """Tests for RollingWindowBacktester."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time series data for backtesting."""
        np.random.seed(42)
        n = 500
        
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        X = np.random.randn(n, 5)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n) * 0.5
        
        return X, y, dates.values

    def test_initialization(self):
        """Test backtester initialization."""
        bt = RollingWindowBacktester(
            model_class=LinearRegression,
            train_window=180,
            test_window=30,
            step_size=30
        )
        assert bt.train_window == 180
        assert bt.test_window == 30
        assert bt.step_size == 30

    def test_run_regression(self, time_series_data):
        """Test running backtest for regression."""
        X, y, dates = time_series_data
        
        bt = RollingWindowBacktester(
            model_class=LinearRegression,
            train_window=180,
            test_window=30,
            step_size=60,
            min_train_samples=50
        )
        
        summary = bt.run(X, y, dates, task='regression')
        
        assert isinstance(summary, BacktestSummary)
        assert summary.total_windows > 0
        assert len(summary.window_results) == summary.total_windows
        assert 'rmse' in summary.aggregate_metrics

    def test_degradation_detection(self, time_series_data):
        """Test degradation detection in backtest."""
        X, y, dates = time_series_data
        
        bt = RollingWindowBacktester(
            model_class=LinearRegression,
            train_window=180,
            test_window=30,
            step_size=60
        )
        
        summary = bt.run(X, y, dates, task='regression')
        
        # Summary should have degradation information
        assert hasattr(summary, 'degradation_detected')
        assert hasattr(summary, 'degradation_windows')

