"""
Backtesting Module for CATIA

Provides historical validation framework to ensure model predictions
match observed reality over time.

Key features:
- Rolling window backtests: Walk-forward validation
- Performance metrics: RMSE, MAE, calibration, coverage
- Prediction vs actual: Compare predicted losses to observed
- Degradation detection: Identify when model performance declines
- Validation reports: Comprehensive backtesting summaries

Use cases in catastrophe modeling:
- Model validation (regulatory requirements)
- Performance monitoring (detect model drift)
- Recalibration triggers (when to retrain)
- Confidence in predictions (historical accuracy)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

from catia.config import LOGGING_CONFIG

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest window."""
    window_start: datetime
    window_end: datetime
    n_samples: int
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: Dict[str, float]


@dataclass
class BacktestSummary:
    """Summary of all backtest results."""
    total_windows: int
    total_samples: int
    window_results: List[BacktestResult]
    aggregate_metrics: Dict[str, float]
    metrics_over_time: pd.DataFrame
    degradation_detected: bool
    degradation_windows: List[int]


@dataclass
class CalibrationResult:
    """Calibration analysis results."""
    predicted_probs: np.ndarray
    observed_freqs: np.ndarray
    bin_counts: np.ndarray
    calibration_error: float  # Expected Calibration Error
    max_calibration_error: float  # Maximum Calibration Error


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Relative metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Bias
    metrics['bias'] = np.mean(y_pred - y_true)
    metrics['bias_pct'] = metrics['bias'] / (np.mean(y_true) + 1e-10) * 100
    
    # Tail metrics (for catastrophe modeling)
    p90_idx = y_true >= np.percentile(y_true, 90)
    if p90_idx.sum() > 0:
        metrics['tail_rmse'] = np.sqrt(mean_squared_error(y_true[p90_idx], y_pred[p90_idx]))
        metrics['tail_bias'] = np.mean(y_pred[p90_idx] - y_true[p90_idx])
    
    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate classification performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Probability metrics
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.5
        metrics['brier'] = brier_score_loss(y_true, y_prob)
    
    return metrics


def calculate_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> CalibrationResult:
    """
    Calculate probability calibration metrics.
    
    Well-calibrated models should have predicted probabilities
    that match observed frequencies.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    predicted_probs = np.zeros(n_bins)
    observed_freqs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            predicted_probs[i] = y_prob[mask].mean()
            observed_freqs[i] = y_true[mask].mean()
    
    # Expected Calibration Error (weighted by bin size)
    weights = bin_counts / bin_counts.sum()
    calibration_gaps = np.abs(predicted_probs - observed_freqs)
    ece = np.sum(weights * calibration_gaps)
    mce = calibration_gaps.max()
    
    return CalibrationResult(
        predicted_probs=predicted_probs,
        observed_freqs=observed_freqs,
        bin_counts=bin_counts,
        calibration_error=ece,
        max_calibration_error=mce
    )


def calculate_coverage(y_true: np.ndarray, y_lower: np.ndarray,
                       y_upper: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction interval coverage.

    Args:
        y_true: Actual values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval

    Returns:
        Coverage metrics
    """
    covered = (y_true >= y_lower) & (y_true <= y_upper)

    return {
        'coverage': covered.mean(),
        'avg_interval_width': np.mean(y_upper - y_lower),
        'median_interval_width': np.median(y_upper - y_lower)
    }


# ============================================================================
# ROLLING WINDOW BACKTESTER
# ============================================================================

class RollingWindowBacktester:
    """
    Walk-forward backtesting with rolling windows.

    Simulates real-world model deployment by training on past data
    and validating on future data.
    """

    def __init__(self, model_class: type, model_params: Dict = None,
                 train_window: int = 365, test_window: int = 30,
                 step_size: int = 30, min_train_samples: int = 100):
        """
        Initialize backtester.

        Args:
            model_class: Class of model to backtest
            model_params: Parameters for model initialization
            train_window: Training window size (days)
            test_window: Test window size (days)
            step_size: Step between windows (days)
            min_train_samples: Minimum training samples required
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples

        self.results: List[BacktestResult] = []

        logger.info(f"RollingWindowBacktester initialized: train={train_window}d, "
                    f"test={test_window}d, step={step_size}d")

    def run(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
            task: str = 'regression') -> BacktestSummary:
        """
        Run rolling window backtest.

        Args:
            X: Feature matrix
            y: Target values
            dates: Date for each sample
            task: 'regression' or 'classification'

        Returns:
            BacktestSummary with all results
        """
        self.results = []
        dates = pd.to_datetime(dates)

        # Determine window boundaries
        min_date = dates.min()
        max_date = dates.max()

        current_start = min_date + timedelta(days=self.train_window)

        window_idx = 0
        while current_start + timedelta(days=self.test_window) <= max_date:
            train_end = current_start
            train_start = train_end - timedelta(days=self.train_window)
            test_end = current_start + timedelta(days=self.test_window)

            # Get training data
            train_mask = (dates >= train_start) & (dates < train_end)
            test_mask = (dates >= current_start) & (dates < test_end)

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            if len(X_train) >= self.min_train_samples and len(X_test) > 0:
                # Train model
                model = self.model_class(**self.model_params)
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate metrics
                if task == 'regression':
                    metrics = calculate_regression_metrics(y_test, y_pred)
                else:
                    y_prob = None
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

                result = BacktestResult(
                    window_start=current_start,
                    window_end=test_end,
                    n_samples=len(X_test),
                    predictions=y_pred,
                    actuals=y_test,
                    metrics=metrics
                )
                self.results.append(result)

                logger.debug(f"Window {window_idx}: {current_start.date()} - {test_end.date()}, "
                             f"n={len(X_test)}, metrics={metrics}")

            current_start += timedelta(days=self.step_size)
            window_idx += 1

        return self._create_summary(task)

    def _create_summary(self, task: str) -> BacktestSummary:
        """Create summary from all window results."""
        if not self.results:
            return BacktestSummary(
                total_windows=0, total_samples=0, window_results=[],
                aggregate_metrics={}, metrics_over_time=pd.DataFrame(),
                degradation_detected=False, degradation_windows=[]
            )

        # Aggregate predictions and actuals
        all_predictions = np.concatenate([r.predictions for r in self.results])
        all_actuals = np.concatenate([r.actuals for r in self.results])

        # Aggregate metrics
        if task == 'regression':
            aggregate_metrics = calculate_regression_metrics(all_actuals, all_predictions)
        else:
            aggregate_metrics = calculate_classification_metrics(all_actuals, all_predictions)

        # Metrics over time
        metrics_data = []
        for r in self.results:
            row = {'window_start': r.window_start, 'window_end': r.window_end,
                   'n_samples': r.n_samples}
            row.update(r.metrics)
            metrics_data.append(row)
        metrics_df = pd.DataFrame(metrics_data)

        # Detect degradation
        degradation_detected, degradation_windows = self._detect_degradation(task)

        return BacktestSummary(
            total_windows=len(self.results),
            total_samples=sum(r.n_samples for r in self.results),
            window_results=self.results,
            aggregate_metrics=aggregate_metrics,
            metrics_over_time=metrics_df,
            degradation_detected=degradation_detected,
            degradation_windows=degradation_windows
        )

    def _detect_degradation(self, task: str, threshold: float = 0.2) -> Tuple[bool, List[int]]:
        """
        Detect model performance degradation over time.

        Uses comparison of recent vs historical performance.
        """
        if len(self.results) < 4:
            return False, []

        # Get primary metric
        metric_name = 'rmse' if task == 'regression' else 'accuracy'
        metrics = [r.metrics.get(metric_name, 0) for r in self.results]

        # Compare first half vs second half
        mid = len(metrics) // 2
        early_mean = np.mean(metrics[:mid])
        late_mean = np.mean(metrics[mid:])

        # For RMSE lower is better, for accuracy higher is better
        if task == 'regression':
            degradation = (late_mean - early_mean) / (early_mean + 1e-10)
        else:
            degradation = (early_mean - late_mean) / (early_mean + 1e-10)

        degradation_detected = degradation > threshold

        # Find specific windows with degradation
        degradation_windows = []
        if degradation_detected:
            baseline = early_mean
            for i, m in enumerate(metrics):
                if task == 'regression':
                    if m > baseline * (1 + threshold):
                        degradation_windows.append(i)
                else:
                    if m < baseline * (1 - threshold):
                        degradation_windows.append(i)

        return degradation_detected, degradation_windows


# ============================================================================
# PREDICTION VS ACTUAL ANALYZER
# ============================================================================

class PredictionActualAnalyzer:
    """Analyze predicted vs actual losses over time."""

    def __init__(self):
        self.comparison_data = []

    def add_comparison(self, date: datetime, predicted: float, actual: float,
                       predicted_lower: float = None, predicted_upper: float = None):
        """Add a prediction-actual pair."""
        self.comparison_data.append({
            'date': date,
            'predicted': predicted,
            'actual': actual,
            'predicted_lower': predicted_lower,
            'predicted_upper': predicted_upper,
            'error': actual - predicted,
            'abs_error': abs(actual - predicted),
            'pct_error': (actual - predicted) / (actual + 1e-10) * 100
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of prediction accuracy."""
        if not self.comparison_data:
            return {}

        df = pd.DataFrame(self.comparison_data)

        predicted = df['predicted'].values
        actual = df['actual'].values

        summary = {
            'n_comparisons': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'metrics': calculate_regression_metrics(actual, predicted),
            'mean_error': df['error'].mean(),
            'std_error': df['error'].std(),
            'mean_abs_error': df['abs_error'].mean(),
            'mean_pct_error': df['pct_error'].mean(),
            'correlation': np.corrcoef(predicted, actual)[0, 1]
        }

        # Coverage if intervals provided
        if df['predicted_lower'].notna().all():
            coverage = calculate_coverage(
                actual,
                df['predicted_lower'].values,
                df['predicted_upper'].values
            )
            summary['coverage'] = coverage

        return summary

    def get_dataframe(self) -> pd.DataFrame:
        """Get comparison data as DataFrame."""
        return pd.DataFrame(self.comparison_data)


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_backtest_report(summary: BacktestSummary, title: str = "Backtest Report") -> str:
    """Format backtest summary as readable report."""
    lines = [
        "=" * 70,
        title,
        "=" * 70,
        "",
        f"Total Windows: {summary.total_windows}",
        f"Total Samples: {summary.total_samples:,}",
        ""
    ]

    if summary.total_windows == 0:
        lines.append("No backtest results available.")
        return "\n".join(lines)

    # Aggregate metrics
    lines.append("AGGREGATE METRICS:")
    lines.append("-" * 40)
    for metric, value in summary.aggregate_metrics.items():
        if isinstance(value, float):
            if 'pct' in metric or 'mape' in metric:
                lines.append(f"  {metric:20s}: {value:8.2f}%")
            elif abs(value) > 1000:
                lines.append(f"  {metric:20s}: ${value:,.0f}")
            else:
                lines.append(f"  {metric:20s}: {value:8.4f}")

    lines.append("")

    # Degradation warning
    if summary.degradation_detected:
        lines.append("⚠️  DEGRADATION DETECTED!")
        lines.append(f"    Affected windows: {summary.degradation_windows}")
        lines.append("    Consider model retraining.")
    else:
        lines.append("✅ No significant performance degradation detected.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_backtest(model_class: type, X: np.ndarray, y: np.ndarray,
                 dates: np.ndarray, task: str = 'regression',
                 **kwargs) -> BacktestSummary:
    """
    Convenience function to run backtest.

    Args:
        model_class: Model class to backtest
        X: Features
        y: Targets
        dates: Dates for each sample
        task: 'regression' or 'classification'
        **kwargs: Additional arguments for RollingWindowBacktester

    Returns:
        BacktestSummary
    """
    backtester = RollingWindowBacktester(model_class, **kwargs)
    return backtester.run(X, y, dates, task)
