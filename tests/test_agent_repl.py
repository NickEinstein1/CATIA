"""Tests for CATIA terminal agent (NL shortcuts + bridge)."""

from catia.agent_bridge import ActuarialScience, RiskAnalysis
from catia.agent_repl import (
    _repl_take_artifact_names,
    _repl_take_perils,
    interpret_natural_language,
)
from catia.config import SIMULATION_CONFIG
from catia.run_spec import merge_cli_run_spec


def test_interpret_natural_language_plain_text_does_not_run_pipeline():
    for phrase in (
        "run a monte carlo for hurricane on gulf coast",
        "train the risk model for flood",
        "full pipeline for the east coast",
        "hurricane variation on the gulf coast",
        "compute var for hurricane gulf",
    ):
        verb, argv = interpret_natural_language(phrase)
        assert verb == "repl_suggest_slash"
        assert argv == []


def test_interpret_natural_language_tips():
    for phrase in ("tips", "hint", "more tips", "show tips", "give me tips"):
        verb, argv = interpret_natural_language(phrase)
        assert verb == "tips"
        assert argv == []


def test_interpret_natural_language_dashboard():
    verb, argv = interpret_natural_language("start the dashboard")
    assert verb == "dashboard"
    assert argv == []


def test_interpret_natural_language_help():
    for phrase in ("help", "?", "hi", "hello", "please help"):
        verb, argv = interpret_natural_language(phrase)
        assert verb == "help"
        assert argv == []


def test_repl_take_artifact_names():
    names, j = _repl_take_artifact_names(
        ["--artifacts", "report", "dashboard", "--region", "x"], 0, "--artifacts"
    )
    assert names == ["report", "dashboard"]
    assert j == 3


def test_repl_take_perils_stops_at_next_flag():
    perils, j = _repl_take_perils(["-p", "hurricane", "flood", "--scenario", "x"], 0, "-p")
    assert perils == ["hurricane", "flood"]
    assert j == 3


def test_repl_take_perils_long_form():
    perils, j = _repl_take_perils(["--perils", "hurricane"], 0, "--perils")
    assert perils == ["hurricane"]
    assert j == 2


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
