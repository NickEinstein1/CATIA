"""Exercise every ``catia-agent`` REPL slash command via :func:`dispatch_command`."""

from __future__ import annotations

import asyncio
import shlex
from unittest.mock import patch

from rich.console import Console

from catia.agent_repl import AgentSession, dispatch_command


def _minimal_pipeline_result() -> dict:
    return {
        "risk_metrics": {
            "descriptive_stats": {"mean": 1_234_567.0},
            "risk_metrics": {"var": 2_000_000.0, "tvar": 3_500_000.0},
        },
        "mitigation_summary": {"total_risk_reduction": 0.25},
    }


async def _dispatch(
    line: str, session: AgentSession | None = None
) -> tuple[bool, Console, AgentSession]:
    console = Console(record=True, width=120, legacy_windows=False)
    sess = session or AgentSession()
    keep_going = await dispatch_command(console, sess, line)
    return keep_going, console, sess


def _run(line: str, session: AgentSession | None = None) -> tuple[bool, str, AgentSession]:
    ok, console, sess = asyncio.run(_dispatch(line, session))
    return ok, console.export_text(clear=False), sess


def test_repl_empty_line_noop():
    ok, out, _ = _run("   ")
    assert ok is True
    assert out == ""


def test_repl_help():
    ok, out, _ = _run("/help")
    assert ok is True
    assert "/run" in out
    assert "/dashboard" in out


def test_repl_tips():
    ok, out, _ = _run("/tips")
    assert ok is True
    assert "Tip" in out or "tips" in out.lower()


def test_repl_exit_aliases():
    for cmd in ("/exit", "/quit", "/q"):
        ok, out, _ = _run(cmd)
        assert ok is False
        assert "oodbye" in out


def test_repl_unknown_command():
    ok, out, _ = _run("/not-a-real-command")
    assert ok is True
    assert "Unknown command" in out


def test_repl_json_empty_then_populated():
    ok, out, _ = _run("/json")
    assert ok is True
    assert "No result yet" in out

    session = AgentSession()
    session.last_result = {"ok": True}
    session.last_label = "unit"
    ok, out2, _ = _run("/json", session)
    assert ok is True
    assert "ok" in out2
    assert "true" in out2.lower()


@patch("catia.agent_repl.run_catia_analysis", return_value=_minimal_pipeline_result())
def test_repl_run_default_artifacts(mock_pipe):
    ok, out, sess = _run("/run -p hurricane")
    assert ok is True
    mock_pipe.assert_called_once()
    call_kw = mock_pipe.call_args.kwargs
    assert call_kw.get("artifacts") is not None
    assert "dashboard" not in call_kw["artifacts"]
    assert sess.last_result is not None


@patch("catia.agent_repl.run_catia_analysis", return_value=_minimal_pipeline_result())
def test_repl_run_full_artifacts(mock_pipe):
    _run("/run --full -p hurricane")
    call_kw = mock_pipe.call_args.kwargs
    assert call_kw.get("artifacts") is None


@patch("catia.agent_repl.run_catia_analysis", return_value=_minimal_pipeline_result())
def test_repl_run_explicit_artifacts(mock_pipe):
    _run("/run --artifacts report dashboard -p hurricane")
    names = mock_pipe.call_args.kwargs["artifacts"]
    assert names == ["report", "dashboard"]


@patch("catia.agent_repl.run_catia_analysis", return_value=_minimal_pipeline_result())
def test_repl_spec_uses_file(mock_pipe, tmp_path):
    yml = tmp_path / "t.yaml"
    yml.write_text(
        "region: US_Gulf_Coast\n"
        "perils: [hurricane]\n"
        "use_mock_data: true\n"
        "artifacts: [report]\n",
        encoding="utf-8",
    )
    line = "/spec " + shlex.quote(str(yml.resolve()))
    ok, out, _ = _run(line)
    assert ok is True
    mock_pipe.assert_called_once()
    assert "Pipeline complete" in out or "Mean" in out


@patch("catia.agent_repl.RiskAnalysis")
def test_repl_risk(mock_ra_class):
    mock_ra = mock_ra_class.return_value
    mock_ra.run.return_value = type(
        "R",
        (),
        {
            "region": "US_Gulf_Coast",
            "perils": ["hurricane"],
            "use_mock_data": True,
            "data": {"climate": [1], "historical_events": [1]},
            "model_summary": {
                "probability_model": "Dummy",
                "severity_model": "Dummy",
            },
        },
    )()

    ok, out, sess = _run("/risk -p hurricane")
    assert ok is True
    mock_ra.run.assert_called_once()
    assert sess.last_result is not None
    assert "Risk" in out or "hurricane" in out


@patch("catia.agent_repl.ActuarialScience")
def test_repl_simulate(mock_ac_class):
    mock_ac = mock_ac_class.return_value
    mock_ac.multi_peril.return_value = type(
        "A",
        (),
        {
            "perils": ["hurricane", "flood"],
            "aggregate_metrics": {
                "descriptive_stats": {"mean": 9e6},
                "risk_metrics": {"var": 1e7, "tvar": 2e7},
            },
        },
    )()

    ok, out, sess = _run("/simulate -p hurricane -p flood")
    assert ok is True
    mock_ac.multi_peril.assert_called_once()
    assert sess.last_result is not None


@patch("catia.dashboard.run_dashboard")
def test_repl_dashboard_starts_background(mock_dash):
    ok, out, sess = _run("/dashboard --port 9876")
    assert ok is True
    assert "9876" in out or "http" in out
    assert sess.dashboard_thread is not None
    sess.dashboard_thread.join(timeout=2)


@patch("catia.dashboard.run_dashboard")
def test_repl_dashboard_second_invocation_when_alive(mock_dash):
    class FakeThread:
        def is_alive(self) -> bool:
            return True

    sess = AgentSession()
    sess.dashboard_thread = FakeThread()
    sess.dashboard_host = "127.0.0.1"
    sess.dashboard_port = 8050
    ok, out, _ = _run("/dashboard", sess)
    assert ok is True
    assert "already running" in out
    mock_dash.assert_not_called()


def test_repl_natural_language_routes_to_help():
    ok, out, _ = _run("help")
    assert ok is True
    assert "/run" in out


def test_repl_natural_language_routes_to_tips():
    ok, out, _ = _run("show tips")
    assert ok is True


@patch("catia.dashboard.run_dashboard")
def test_repl_natural_language_dashboard(mock_dash):
    ok, _, sess = _run("open dashboard")
    assert ok is True
    sess.dashboard_thread.join(timeout=2)


def test_repl_plain_text_does_not_invoke_pipeline():
    with patch("catia.agent_repl.run_catia_analysis") as mock_pipe:
        ok, out, _ = _run("full pipeline for east with hurricane and flood")
        assert ok is True
        mock_pipe.assert_not_called()
        assert "/run" in out
        ok2, out2, _ = _run("just typing random hurricane words")
        assert ok2 is True
        mock_pipe.assert_not_called()
        assert "/run" in out2 or "deterministic" in out2.lower()
