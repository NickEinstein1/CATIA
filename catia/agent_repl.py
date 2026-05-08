"""
Terminal agent-style interface for CATIA (Click + Rich).

Async REPL: structured ``/commands`` and lightweight natural-language routing.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from catia.agent_bridge import ActuarialResult, ActuarialScience, RiskAnalysis, RiskAnalysisResult
from catia.config import DEFAULT_PERILS, PERIL_CONFIG, SIMULATION_CONFIG
from catia.pipeline import run_catia_analysis
from catia.run_spec import load_run_spec


def _require_perils(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if t in PERIL_CONFIG and t not in out:
            out.append(t)
    return out


def _parse_region(text: str) -> Optional[str]:
    low = text.lower()
    mapping = [
        ("gulf", "US_Gulf_Coast"),
        ("east coast", "US_East_Coast"),
        ("west coast", "US_West_Coast"),
        ("midwest", "US_Midwest"),
    ]
    for key, rid in mapping:
        if key in low:
            return rid
    for rid in (
        "US_Gulf_Coast",
        "US_East_Coast",
        "US_West_Coast",
        "US_Midwest",
        "Caribbean",
        "Europe",
    ):
        if rid.lower().replace("_", " ") in low or rid in text:
            return rid
    return None


def interpret_natural_language(text: str) -> Tuple[str, List[str]]:
    """
    Map free text to a pseudo-command and args for the dispatcher.

    Returns (verb, argv) where argv excludes the verb.
    """
    t = text.strip()
    low = t.lower()

    region = _parse_region(t) or "US_Gulf_Coast"
    perils = _require_perils(low.split())
    if not perils:
        perils = list(DEFAULT_PERILS)

    actuarial_kw = any(
        k in low
        for k in (
            "simulate",
            "simulation",
            "monte carlo",
            "montecarlo",
            "var",
            "tvar",
            "loss",
            "actuarial",
        )
    )
    risk_kw = any(
        k in low
        for k in ("train", "model", "ml", "feature", "risk model", "predict")
    )
    full_kw = any(
        k in low
        for k in ("full pipeline", "full analysis", "everything", "end to end", "end-to-end")
    )

    if full_kw or (not actuarial_kw and not risk_kw and "help" not in low):
        args = ["--region", region, "--perils", *perils]
        return "run", args

    if actuarial_kw and not risk_kw:
        args = ["--perils", *perils]
        if "baseline" in low:
            args.extend(["--scenario", "baseline"])
        if "stress" in low or "high stress" in low:
            args.extend(["--scenario", "high_stress"])
        return "simulate", args

    if risk_kw and not actuarial_kw:
        args = ["--region", region, "--perils", *perils]
        return "risk", args

    if "help" in low or low in ("?", "hi", "hello"):
        return "help", []

    args = ["--region", region, "--perils", *perils]
    return "run", args


def _print_error(console: Console, title: str, exc: BaseException) -> None:
    msg = Text(str(exc), style="bold red")
    console.print(
        Panel(
            msg,
            title=f"[red bold]{title}[/red bold]",
            border_style="red",
            subtitle="Session continues — fix inputs or try /help",
        )
    )


def _json_panel(console: Console, data: Any, title: str) -> None:
    try:
        payload = json.dumps(data, indent=2, default=str)
    except TypeError:
        payload = repr(data)
    syntax = Syntax(payload, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))


def _summary_panel(
    console: Console, title: str, lines: List[str]
) -> None:
    body = Group(*[Text(line) for line in lines])
    console.print(Panel(body, title=title, border_style="cyan"))


async def _run_blocking(
    console: Console,
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run blocking CATIA work in a thread with a Rich ``Live`` spinner."""

    spinner = Spinner("dots", text=label, style="cyan")
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        return await asyncio.to_thread(fn, *args, **kwargs)


class AgentSession:
    """Holds last result for /json and optional state."""

    def __init__(self) -> None:
        self.last_result: Any = None
        self.last_label: str = "last"


async def dispatch_command(
    console: Console,
    session: AgentSession,
    line: str,
) -> bool:
    """
    Handle one REPL line. Returns False if the REPL should exit.
    """
    stripped = line.strip()
    if not stripped:
        return True

    if stripped.startswith("/"):
        parts = shlex.split(stripped)
        verb = parts[0].lower()
        argv = parts[1:]
    else:
        verb, argv = interpret_natural_language(stripped)
        verb = "/" + verb

    try:
        if verb in ("/exit", "/quit", "/q"):
            console.print(Panel(Text("Goodbye.", style="dim"), border_style="magenta"))
            return False

        if verb == "/help":
            lines = [
                "Structured commands (prefix /):",
                "  /run [--region R] [--perils P ...] [--mock|--real] [--scenario S] [--iterations N] [--output-dir D]",
                "  /risk [--region R] [--perils P ...] [--mock|--real]  — data + train (RiskAnalysis → data_acquisition + risk_prediction)",
                "  /simulate [--perils P ...] [--scenario S] [--iterations N] [--no-uncertainty]  — actuarial MC (ActuarialScience → financial_impact)",
                "  /spec PATH — run full pipeline from YAML/JSON RunSpec file",
                "  /json — print last result as highlighted JSON",
                "  /exit — leave the session",
                "",
                "Natural language: mention perils, 'simulate', 'train model', 'full pipeline', or regions (e.g. gulf coast).",
            ]
            _summary_panel(console, "CATIA agent — help", lines)
            return True

        if verb == "/json":
            if session.last_result is None:
                console.print(
                    Panel(
                        Text("No result yet. Run /run or /risk first.", style="yellow"),
                        title="Output",
                    )
                )
                return True
            _json_panel(console, session.last_result, title=session.last_label)
            return True

        if verb == "/spec":
            if not argv:
                raise click.UsageError("/spec requires a path to a YAML/JSON RunSpec")
            path = argv[0]
            spec = load_run_spec(path)

            raw = await _run_blocking(
                console,
                "Full pipeline (RunSpec)…",
                run_catia_analysis,
                **spec.to_kwargs(),
            )
            session.last_result = raw
            session.last_label = f"run_spec:{path}"
            mean = raw["risk_metrics"]["descriptive_stats"]["mean"]
            var = raw["risk_metrics"]["risk_metrics"]["var"]
            _summary_panel(
                console,
                "Pipeline complete",
                [
                    f"RunSpec: {path}",
                    f"Mean annual loss: ${mean:,.0f}",
                    f"VaR (95%): ${var:,.0f}",
                    "Use /json for full structured output.",
                ],
            )
            return True

        if verb == "/risk":
            region = "US_Gulf_Coast"
            perils: List[str] = list(DEFAULT_PERILS)
            use_mock = True
            i = 0
            while i < len(argv):
                a = argv[i]
                if a == "--region" and i + 1 < len(argv):
                    region = argv[i + 1]
                    i += 2
                    continue
                if a == "--perils" and i + 1 < len(argv):
                    perils = []
                    i += 1
                    while i < len(argv) and not argv[i].startswith("-"):
                        perils.append(argv[i])
                        i += 1
                    continue
                if a == "--real":
                    use_mock = False
                    i += 1
                    continue
                if a == "--mock":
                    use_mock = True
                    i += 1
                    continue
                raise click.UsageError(f"Unknown /risk argument: {a}")
            ra = RiskAnalysis()

            result: RiskAnalysisResult = await _run_blocking(
                console,
                "Risk analysis: fetch data + train model…",
                ra.run,
                region,
                use_mock_data=use_mock,
                perils=perils,
            )
            session.last_result = {
                "region": result.region,
                "perils": result.perils,
                "use_mock_data": result.use_mock_data,
                "data_summary": {
                    "climate_rows": len(result.data["climate"]),
                    "events": len(result.data["historical_events"]),
                },
                "model_summary": result.model_summary,
            }
            session.last_label = "risk_analysis"
            _summary_panel(
                console,
                "RiskAnalysis",
                [
                    f"Region: {result.region}",
                    f"Perils: {', '.join(result.perils)}",
                    f"Mock data: {result.use_mock_data}",
                    f"Model: {result.model_summary}",
                ],
            )
            return True

        if verb == "/simulate":
            perils: List[str] = list(DEFAULT_PERILS)
            scenario_id: Optional[str] = None
            num_iterations: Optional[int] = None
            include_uncertainty = True
            i = 0
            while i < len(argv):
                a = argv[i]
                if a == "--perils" and i + 1 < len(argv):
                    perils = []
                    i += 1
                    while i < len(argv) and not argv[i].startswith("-"):
                        perils.append(argv[i])
                        i += 1
                    continue
                if a == "--scenario" and i + 1 < len(argv):
                    scenario_id = argv[i + 1]
                    i += 2
                    continue
                if a == "--iterations" and i + 1 < len(argv):
                    num_iterations = int(argv[i + 1])
                    i += 2
                    continue
                if a == "--no-uncertainty":
                    include_uncertainty = False
                    i += 1
                    continue
                raise click.UsageError(f"Unknown /simulate argument: {a}")
            ac = ActuarialScience()

            out: ActuarialResult = await _run_blocking(
                console,
                "ActuarialScience: Monte Carlo multi-peril…",
                ac.multi_peril,
                perils,
                include_uncertainty=include_uncertainty,
                scenario_id=scenario_id,
                num_iterations=num_iterations,
            )
            session.last_result = {
                "perils": out.perils,
                "aggregate_descriptive": out.aggregate_metrics.get("descriptive_stats"),
                "aggregate_risk_metrics": out.aggregate_metrics.get("risk_metrics"),
                "scenario_id": scenario_id,
            }
            session.last_label = "actuarial_multi_peril"
            m = out.aggregate_metrics["descriptive_stats"]["mean"]
            v = out.aggregate_metrics["risk_metrics"]["var"]
            _summary_panel(
                console,
                "ActuarialScience",
                [
                    f"Perils: {', '.join(out.perils)}",
                    f"Mean annual loss: ${m:,.0f}",
                    f"VaR (95%): ${v:,.0f}",
                    "Use /json for more metrics.",
                ],
            )
            return True

        if verb == "/run":
            region = "US_Gulf_Coast"
            perils: List[str] = list(DEFAULT_PERILS)
            use_mock = True
            scenario_id: Optional[str] = None
            iterations: Optional[int] = None
            output_dir: Optional[str] = None
            i = 0
            while i < len(argv):
                a = argv[i]
                if a == "--region" and i + 1 < len(argv):
                    region = argv[i + 1]
                    i += 2
                    continue
                if a == "--perils" and i + 1 < len(argv):
                    perils = []
                    i += 1
                    while i < len(argv) and not argv[i].startswith("-"):
                        perils.append(argv[i])
                        i += 1
                    continue
                if a == "--real":
                    use_mock = False
                    i += 1
                    continue
                if a == "--mock":
                    use_mock = True
                    i += 1
                    continue
                if a == "--scenario" and i + 1 < len(argv):
                    scenario_id = argv[i + 1]
                    i += 2
                    continue
                if a == "--iterations" and i + 1 < len(argv):
                    iterations = int(argv[i + 1])
                    i += 2
                    continue
                if a == "--output-dir" and i + 1 < len(argv):
                    output_dir = argv[i + 1]
                    i += 2
                    continue
                raise click.UsageError(f"Unknown /run argument: {a}")

            raw = await _run_blocking(
                console,
                "Full CATIA pipeline (data → ML → actuarial → mitigation → reports)…",
                run_catia_analysis,
                region,
                use_mock,
                perils,
                scenario_id=scenario_id,
                monte_carlo_iterations=iterations,
                output_dir=output_dir,
                artifacts=None,
            )
            session.last_result = raw
            session.last_label = "full_pipeline"
            mean = raw["risk_metrics"]["descriptive_stats"]["mean"]
            var = raw["risk_metrics"]["risk_metrics"]["var"]
            _summary_panel(
                console,
                "Full pipeline",
                [
                    f"Region: {region} | Perils: {', '.join(perils)}",
                    f"Mean annual loss: ${mean:,.0f}",
                    f"VaR (95%): ${var:,.0f}",
                    f"(Simulation iterations this run: {iterations or SIMULATION_CONFIG['monte_carlo_iterations']})",
                    "Use /json for full structured output.",
                ],
            )
            return True

        console.print(Panel(Text(f"Unknown command: {verb}", style="yellow"), title="Parser"))
        return True

    except Exception as e:
        _print_error(console, "CATIA / actuarial error", e)
        return True


async def async_repl() -> None:
    console = Console()
    session = AgentSession()
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]CATIA agent[/bold cyan] — "
                "type [bold]/help[/bold] or describe what you want (e.g. "
                "[italic]simulate hurricane flood gulf coast[/italic])."
            ),
            border_style="cyan",
            title="Welcome",
        )
    )
    loop = True
    while loop:
        try:
            line = await asyncio.to_thread(
                input,
                click.style("catia› ", fg="green", bold=True),
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted — use /exit to quit.[/dim]")
            continue
        loop = await dispatch_command(console, session, line)


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CATIA terminal agent (async REPL)."""
    if ctx.invoked_subcommand is None:
        asyncio.run(async_repl())


@cli.command("repl")
def repl_cmd() -> None:
    """Start the interactive agent session (same as bare ``catia-agent``)."""
    asyncio.run(async_repl())


def main() -> None:
    """Console script entry: same as ``click`` group."""
    cli()


if __name__ == "__main__":
    main()
