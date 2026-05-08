"""Tests for CATIA terminal agent (NL routing + bridge)."""

from catia.agent_bridge import ActuarialScience, RiskAnalysis
from catia.agent_repl import interpret_natural_language
from catia.config import SIMULATION_CONFIG
from catia.run_spec import merge_cli_run_spec


def test_interpret_natural_language_simulate():
    verb, argv = interpret_natural_language("run a monte carlo for hurricane on gulf coast")
    assert verb == "simulate"
    assert "--perils" in argv
    assert "hurricane" in argv


def test_interpret_natural_language_risk():
    verb, argv = interpret_natural_language("train the risk model for flood")
    assert verb == "risk"
    assert "flood" in argv


def test_interpret_natural_language_full():
    verb, argv = interpret_natural_language("full pipeline for the east coast")
    assert verb == "run"
    assert "--region" in argv
    assert "US_East_Coast" in argv


def test_interpret_natural_language_tips():
    for phrase in ("tips", "hint", "more tips", "show tips", "give me tips"):
        verb, argv = interpret_natural_language(phrase)
        assert verb == "tips"
        assert argv == []


def test_risk_analysis_bridge_mock():
    ra = RiskAnalysis()
    out = ra.run("US_Gulf_Coast", use_mock_data=True, perils=["hurricane"])
    assert out.model_summary.get("train_status") == "ok"
    assert len(out.data["climate"]) > 0


def test_merge_cli_run_spec_overrides_mock():
    s = merge_cli_run_spec(region="US_East_Coast", no_mock_data=True)
    assert s.region == "US_East_Coast"
    assert s.use_mock_data is False


def test_actuarial_bridge_small_mc(monkeypatch):
    old = SIMULATION_CONFIG["monte_carlo_iterations"]
    SIMULATION_CONFIG["monte_carlo_iterations"] = 400
    try:
        ac = ActuarialScience()
        res = ac.multi_peril(["hurricane"], include_uncertainty=False, num_iterations=400)
        assert "mean" in res.aggregate_metrics["descriptive_stats"]
    finally:
        SIMULATION_CONFIG["monte_carlo_iterations"] = old
