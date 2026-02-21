"""Tests for stress scenario application (Solvency-II-style)."""

import pytest
from catia.scenario_analysis import apply_stress_scenarios, PREDEFINED_SCENARIOS


def test_apply_stress_scenarios_baseline_only():
    baseline = {"mean": 100.0, "var_95": 200.0, "tvar_95": 250.0, "return_periods": {"10_year": 50, "100_year": 300}}
    out = apply_stress_scenarios(baseline, scenario_ids=["baseline"])
    assert "baseline" in out["scenarios"]
    assert out["scenarios"]["baseline"]["mean"] == 100.0
    assert out["scenarios"]["baseline"]["var_95"] == 200.0


def test_apply_stress_scenarios_severity_shock():
    baseline = {"mean": 100.0, "var_95": 200.0, "tvar_95": 250.0, "return_periods": {"10_year": 50, "100_year": 300}}
    out = apply_stress_scenarios(baseline, scenario_ids=["solvency_severity_shock"])
    s = out["scenarios"]["solvency_severity_shock"]
    assert s["mean"] == pytest.approx(100.0 * 1.3)
    assert s["var_95"] == pytest.approx(200.0 * 1.3)
    assert s["tvar_95"] == pytest.approx(250.0 * 1.3)
    assert s["return_periods"]["10_year"] == pytest.approx(50 * 1.3)
    assert "Severity" in s["name"] or "severity" in s["name"].lower()


def test_apply_stress_scenarios_combined_shock():
    baseline = {"mean": 100.0, "var_95": 200.0, "tvar_95": 250.0, "return_periods": {}}
    out = apply_stress_scenarios(baseline, scenario_ids=["solvency_combined_shock"])
    s = out["scenarios"]["solvency_combined_shock"]
    # frequency 1.5, severity 1.3 -> mean *= 1.5 * 1.3
    assert s["mean"] == pytest.approx(100.0 * 1.5 * 1.3)
    assert s["var_95"] == pytest.approx(200.0 * 1.3)


def test_apply_stress_scenarios_all_default():
    baseline = {"mean": 10.0, "var_95": 20.0, "tvar_95": 25.0, "return_periods": {"100_year": 100.0}}
    out = apply_stress_scenarios(baseline)
    assert out["baseline"] == baseline
    assert len(out["scenarios"]) >= 5
    for sid, data in out["scenarios"].items():
        assert "name" in data and "mean" in data and "var_95" in data
