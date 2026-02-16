"""
Property-based tests for CATIA risk metrics.

Assert mathematical invariants that must always hold:
- VaR <= TVaR (by definition)
- Return period losses are monotonic (longer return period => higher loss)
- Non-negative losses and sensible ordering of percentiles
"""

import numpy as np
import pytest

from catia.financial_impact import (
    FinancialImpactSimulator,
    run_multi_peril_analysis,
)
from catia.config import SIMULATION_CONFIG, RISK_METRICS


# -----------------------------------------------------------------------------
# Properties on any loss array
# -----------------------------------------------------------------------------

def test_var_leq_tvar_any_losses():
    """Property: For any loss distribution, VaR(p) <= TVaR(p)."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        losses = np.abs(rng.exponential(scale=1e6, size=1000))
        sim = FinancialImpactSimulator(0.5, {"mu": 14, "sigma": 2})
        metrics = sim.calculate_var_tvar(losses)
        assert metrics["var"] <= metrics["tvar"], "VaR must be <= TVaR"


def test_return_periods_monotonic_any_losses():
    """Property: Return period losses must be non-decreasing in return period."""
    rng = np.random.default_rng(123)
    losses = np.abs(rng.lognormal(mean=14, sigma=2, size=2000))
    sim = FinancialImpactSimulator(0.5, {"mu": 14, "sigma": 2})
    rp = sim.calculate_return_periods(losses)
    periods = RISK_METRICS["return_periods"]
    values = [rp[f"{p}_year"] for p in periods]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"Return period {periods[i]}y ({values[i]}) must be <= {periods[i+1]}y ({values[i+1]})"
        )


def test_losses_non_negative_after_simulation():
    """Property: Simulated annual losses are non-negative."""
    sim = FinancialImpactSimulator(0.5, {"mu": 14, "sigma": 2})
    losses = sim.simulate_annual_losses(num_years=500)
    assert np.all(losses >= 0), "All simulated losses must be >= 0"
    assert np.mean(losses) >= 0, "Mean loss must be >= 0"


def test_percentiles_ordered():
    """Property: 50th <= 75th <= 90th <= 95th <= 99th percentile."""
    rng = np.random.default_rng(99)
    losses = np.abs(rng.lognormal(14, 2, size=1000))
    sim = FinancialImpactSimulator(0.5, {"mu": 14, "sigma": 2})
    agg = sim.calculate_aggregate_metrics(losses)
    pct = agg["percentiles"]
    order = ["50th", "75th", "90th", "95th", "99th"]
    for i in range(len(order) - 1):
        assert pct[order[i]] <= pct[order[i + 1]], (
            f"Percentile {order[i]} <= {order[i+1]}"
        )


# -----------------------------------------------------------------------------
# Properties on multi-peril output (use reduced iterations for speed)
# -----------------------------------------------------------------------------

@pytest.fixture
def reduced_mc():
    """Temporarily reduce Monte Carlo iterations for property tests."""
    old = SIMULATION_CONFIG["monte_carlo_iterations"]
    SIMULATION_CONFIG["monte_carlo_iterations"] = 800
    yield
    SIMULATION_CONFIG["monte_carlo_iterations"] = old


def test_multi_peril_var_leq_tvar(reduced_mc):
    """Property: Multi-peril run produces VaR <= TVaR."""
    results = run_multi_peril_analysis(
        perils=["hurricane"],
        include_evt=False,
        include_uncertainty=False,
        include_correlation=False,
    )
    agg = results["aggregate_metrics"]
    var_95 = agg["risk_metrics"]["var"]
    tvar_95 = agg["risk_metrics"]["tvar"]
    assert var_95 <= tvar_95, "Multi-peril VaR(95) <= TVaR(95)"


def test_multi_peril_return_periods_monotonic(reduced_mc):
    """Property: Multi-peril return periods are monotonic."""
    results = run_multi_peril_analysis(
        perils=["hurricane", "flood"],
        include_evt=False,
        include_uncertainty=False,
    )
    rp = results["aggregate_metrics"]["return_periods"]
    periods = [10, 25, 50, 100, 250, 500, 1000]
    values = [rp[f"{p}_year"] for p in periods]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"RP {periods[i]}y <= {periods[i+1]}y"
        )


def test_aggregate_contributions_sum_consistent(reduced_mc):
    """Property: Per-peril mean losses are non-negative; aggregate stats exist."""
    results = run_multi_peril_analysis(
        perils=["hurricane", "flood"],
        include_evt=False,
        include_uncertainty=False,
    )
    for c in results["contributions"]:
        assert c["mean_loss"] >= 0, f"Peril {c['peril_name']} mean_loss >= 0"
        assert 0 <= c["contribution_pct"] <= 100, "Contribution % in [0,100]"
    agg = results["aggregate_metrics"]
    assert agg["descriptive_stats"]["mean"] >= 0
    assert agg["risk_metrics"]["var"] >= 0
    assert agg["risk_metrics"]["tvar"] >= 0
