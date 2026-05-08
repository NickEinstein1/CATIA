"""
Terminal agent-style interface for CATIA (Click + Rich).

Async REPL: structured ``/commands`` and lightweight natural-language routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from catia.agent_bridge import ActuarialResult, ActuarialScience, RiskAnalysis, RiskAnalysisResult
from catia.config import DEFAULT_PERILS, SIMULATION_CONFIG
from catia.pipeline import run_catia_analysis
from catia.run_spec import KNOWN_ARTIFACTS, load_run_spec, merge_cli_run_spec

# Default /run (and default RunSpec example) skips static HTML charts; add dashboard via
# ``/run --full``, ``/run --artifacts dashboard …``, or ``catia-agent dashboard``.
REPL_RUN_ARTIFACTS: Tuple[str, ...] = (
    "report",
    "assumption_register",
    "compliance",
    "enhancements",
)

# One-line hints (plain text); shown in /tips and rotated occasionally before the prompt.
AGENT_TIPS: Tuple[str, ...] = (
    "Default runs use mock/synthetic data — fine for demos; pair --real with NOAA_API_TOKEN for climate pulls.",
    "After /run or /risk, type /json to inspect results; catia_report.json includes metadata.transparency.",
    "Shell: catia-agent run --explain (or CATIA_EXPLAIN=1) prints what the pipeline will do before it runs.",
    "In the REPL, /run skips static Plotly HTML by default; use /run --full or /run --artifacts dashboard … to emit charts.",
    "Regions are coarse labels (e.g. US_Gulf_Coast), not city polygons — extend the stack for asset-level work.",
    "/simulate skips mitigation and reports; use /run for the full pipeline to outputs/.",
    "Analyses need slash commands (/run, /simulate, /risk) — plain English won’t start the pipeline (only tips/help/dashboard shortcuts).",
    "Stuck? python -m catia.agent_repl avoids PATH issues on Windows if catia-agent is not found.",
    "Dashboard & API: catia-agent dashboard and catia-agent api mirror catia --dashboard and catia --api.",
    "SHAP feature importance needs CATIA_USE_SHAP=1 and runs in the full /run path when SHAP is installed.",
    "/spec accepts YAML or JSON — keep a RunSpec in version control for repeatable analyses.",
    "Higher Monte Carlo iterations (--iterations or spec) mean smoother tails but slower runs; start low, then scale up.",
    "Artifacts list in RunSpec (report, dashboard, shap) controls what files land under output_dir — skip what you do not need.",
    "PYTHONWARNINGS=default surfaces deprecation noise; use logging level DEBUG only when tracing pipeline steps.",
    "pytest tests/ and examples/… configs are the fastest check after changing peril or region logic.",
)

REPL_PROMPT_MARKUP = "[bold cyan]catia[/bold cyan] [dim]>[/dim] "


def _quick_tips_rich() -> Text:
    """Three high-value tips on the welcome screen."""
    return Text.from_markup(
        "[bold yellow]Quick tips[/bold yellow]\n"
        "[yellow]•[/yellow] [dim]Use[/dim] [green]/json[/green] [dim]after a run to explore output;[/dim] "
        "[green]/tips[/green] [dim]lists all hints.[/dim]\n"
        "[yellow]•[/yellow] [dim]Mock data is default — intentional for transparency; see[/dim] "
        "[cyan]notebooks/docs/transparency.md[/cyan][dim].[/dim]\n"
        "[yellow]•[/yellow] [dim]Shell one-shot:[/dim] [magenta]catia-agent run -r US_Gulf_Coast -p hurricane --explain[/magenta]"
    )


def _print_all_tips(console: Console) -> None:
    table = Table(title="CATIA agent — tips", show_header=True, header_style="bold cyan", border_style="cyan")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Tip", style="white")
    for i, tip in enumerate(AGENT_TIPS, start=1):
        table.add_row(str(i), tip)
    console.print(Panel(table, subtitle="[dim]Rotate automatically every few prompts — or ask for[/dim] [green]tips[/green] [dim]in plain English.[/dim]"))


def _maybe_print_rotating_tip(console: Console, session: "AgentSession") -> None:
    """Print a gentle hint every few prompts so the session stays discoverable."""
    if session.prompt_count <= 1:
        return
    if (session.prompt_count - 1) % 5 != 0:
        return
    idx = ((session.prompt_count - 1) // 5 - 1) % len(AGENT_TIPS)
    console.print(
        Text.from_markup(
            f"[bold yellow]Tip[/bold yellow] [dim]—[/dim] {AGENT_TIPS[idx]}"
        )
    )

EXAMPLE_PROMPTS_INTRO = (
    "[bold yellow]Example prompts[/bold yellow] [dim](copy or type your own)[/dim]"
)


def _example_prompts_block() -> Text:
    return Text.from_markup(
        f"{EXAMPLE_PROMPTS_INTRO}\n"
        "[green]/run[/green] [dim]-r US_Gulf_Coast -p hurricane -p flood[/dim]  [dim](no static HTML charts)[/dim]\n"
        "[green]/run[/green] [dim]--full[/dim]  [dim]— same + Plotly bundle in outputs/[/dim]\n"
        "[green]/simulate[/green] [dim]-p hurricane -p flood[/dim]  "
        "[dim]— Monte Carlo (no fuzzy plain-text routing)[/dim]\n"
        "[green]/risk[/green] [dim]--region US_East_Coast --perils hurricane[/dim]\n"
        "[green]/spec[/green] [dim](default:[/dim] examples/runs/baseline.yaml[dim])[/dim] "
        "[yellow][PATH][/yellow]\n"
        "[green]/json[/green] [dim]— pretty-print last result[/dim]\n"
        "[green]/tips[/green]  [dim]— all hints (or type[/dim] [magenta]tips[/magenta] [dim]in plain English)[/dim]\n"
        "[green]/help[/green]  [dim]— full command list[/dim]\n"
        "[green]/exit[/green]"
    )


def _print_repl_welcome(console: Console) -> None:
    header = Text.from_markup(
        "[bold cyan]CATIA agent[/bold cyan] — interactive multi-peril modeling.\n"
        "[dim]Slash commands[/dim] [bright_white]/run[/bright_white], "
        "[bright_white]/risk[/bright_white], [bright_white]/simulate[/bright_white] … "
        "[dim]Type[/dim] [yellow]/[/yellow] [dim]commands for models; plain shortcuts: tips, help, dashboard.[/dim]"
    )
    body = Group(header, Text(""), _example_prompts_block(), Text(""), _quick_tips_rich())
    console.print(
        Panel(
            body,
            title="[bold white on blue] Welcome [/]",
            subtitle="[dim]Colors:[/dim] Windows Terminal / VS Code terminal. "
            "[yellow]RICH_FORCE_COLOR=1[/yellow] [dim]if needed.[/dim]",
            border_style="cyan",
        )
    )


async def _prompt_line(console: Console) -> str:
    """Read one line with a Rich-colored prompt (works on ANSI-capable terminals)."""

    def _read() -> str:
        return str(
            Prompt.ask(
                REPL_PROMPT_MARKUP,
                console=console,
                show_default=False,
            )
        ).strip()

    return await asyncio.to_thread(_read)


def _repl_take_perils(argv: List[str], start: int, flag: str) -> Tuple[List[str], int]:
    """Parse tokens after ``--perils`` or ``-p`` until the next option. Returns ``(perils, next_index)``."""
    i = start + 1
    if i >= len(argv) or argv[i].startswith("-"):
        raise click.UsageError(f"{flag} requires at least one peril name")
    out: List[str] = []
    while i < len(argv) and not argv[i].startswith("-"):
        out.append(argv[i])
        i += 1
    return out, i


def _repl_take_artifact_names(argv: List[str], start: int, flag: str) -> Tuple[List[str], int]:
    """Parse artifact keys after ``--artifacts`` until the next option."""
    i = start + 1
    if i >= len(argv) or argv[i].startswith("-"):
        raise click.UsageError(f"{flag} requires at least one artifact name")
    out: List[str] = []
    while i < len(argv) and not argv[i].startswith("-"):
        name = argv[i]
        if name not in KNOWN_ARTIFACTS:
            raise click.UsageError(
                f"Unknown artifact {name!r}. Valid: {', '.join(sorted(KNOWN_ARTIFACTS))}"
            )
        out.append(name)
        i += 1
    return out, i


def interpret_natural_language(text: str) -> Tuple[str, List[str]]:
    """
    Map a **non-slash** line to at most a handful of conversational commands.

    Pipeline work (**/run**, **/risk**, **/simulate**, …) is intentionally **not**
    inferred from free text — users must type slash commands for deterministic parsing.
    """
    t = text.strip()
    low = t.lower()
    toks = low.split()

    if "tips" in toks or (len(toks) == 1 and toks[0] in ("tip", "hint", "hints")):
        return "tips", []
    if "tip" in toks and any(w in toks for w in ("show", "list", "more", "give")):
        return "tips", []

    if any(
        phrase in low
        for phrase in (
            "dashboard",
            "dash board",
            "start dashboard",
            "open dashboard",
            "show dashboard",
        )
    ):
        return "dashboard", []

    if "help" in low or low in ("?", "hi", "hello"):
        return "help", []

    return "repl_suggest_slash", []


def _print_non_slash_hint(console: Console, line: str) -> None:
    """Tell the user analyses require ``/`` commands (no fuzzy NL routing)."""
    shown = line if len(line) <= 120 else line[:117] + "…"
    console.print(
        Panel(
            Text.from_markup(
                "This REPL only runs models on [bold]/commands[/bold] so input stays "
                "deterministic (no guessing intent from free text).\n\n"
                f"You typed: [yellow]{shown}[/yellow]\n\n"
                "Try [green]/help[/green] for all commands, or e.g. "
                "[green]/run[/green] [dim]-p hurricane[/dim], "
                "[green]/simulate[/green] [dim]-p flood[/dim], "
                "[green]/risk[/green] [dim]-p hurricane[/dim].\n"
                "[dim]Plain shortcuts still work: “tips”, “help”, “open dashboard”.[/dim]"
            ),
            title="[bold]Use a / command[/bold]",
            border_style="yellow",
        )
    )


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


def _summary_table(
    console: Console,
    title: str,
    rows: List[Tuple[str, str]],
    *,
    hint: Optional[str] = None,
) -> None:
    """Print a key/value summary as a bordered table inside a panel."""
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        box=box.ROUNDED,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("Metric", style="dim", no_wrap=False, ratio=1)
    table.add_column("Value", style="bold default", no_wrap=False, ratio=2)
    for label, value in rows:
        table.add_row(label, value)
    parts: List[Any] = [table]
    if hint:
        parts.extend([Text(""), Text(hint, style="dim italic")])
    console.print(
        Panel(
            Group(*parts),
            title=f"[bold bright_cyan]{title}[/bold bright_cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


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
        self.prompt_count: int = 0
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_host: str = "127.0.0.1"
        self.dashboard_port: int = 8050


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
        if verb == "repl_suggest_slash":
            _print_non_slash_hint(console, stripped)
            return True
        verb = "/" + verb

    try:
        if verb in ("/exit", "/quit", "/q"):
            console.print(Panel(Text("Goodbye.", style="dim"), border_style="magenta"))
            return False

        if verb == "/tips":
            _print_all_tips(console)
            return True

        if verb == "/help":
            help_text = Text.from_markup(
                "[bold cyan]Structured[/bold cyan] [dim](prefix [green]/[/green]):[/dim]\n"
                "  [green]/run[/green]  [dim][-r|--region R] [-p|--perils P …] [--real] [--scenario S] "
                "[--iterations N] [-o|--output-dir D] [--full | --artifacts A …][/dim]\n"
                "      [dim](REPL default skips static Plotly HTML under outputs/;[/dim] [green]--full[/green] "
                "[dim]enables all artifacts.)[/dim]\n"
                "  [green]/risk[/green]   [dim][-r|--region R] [-p|--perils P …] [--real|--mock][/dim]\n"
                "  [green]/simulate[/green]  [dim][-p|--perils P …] [--scenario S] [--iterations N] "
                "[--no-uncertainty][/dim]\n"
                "  [green]/spec[/green] [yellow][PATH][/yellow]  [dim]— YAML/JSON [RunSpec]; default:[/dim] "
                "[dim]examples/runs/baseline.yaml[/dim]\n"
                "  [green]/dashboard[/green]  [dim][--host H] [--port P] [-v][/dim]  [dim]— Dash UI in background[/dim]\n"
                "  [green]/json[/green]  [green]/tips[/green]  [green]/help[/green]  [green]/exit[/green]\n\n"
                "[bold cyan]Shell[/bold cyan] [dim](outside this REPL):[/dim]\n"
                "  [magenta]catia-agent run …[/magenta]   [magenta]catia-agent api …[/magenta]   "
                "[magenta]catia-agent dashboard …[/magenta]\n\n"
        "[bold cyan]Plain text[/bold cyan] [dim](no [green]/[/green]): only[/dim] "
        "[yellow]tips[/yellow][dim],[/dim] [yellow]help[/yellow][dim],[/dim] [yellow]dashboard[/yellow] "
        "[dim]shortcuts — everything else must use[/dim] [green]/run[/green][dim],[/dim] "
        "[green]/risk[/green][dim],[/dim] [green]/simulate[/green][dim], etc.[/dim]"
            )
            console.print(
                Panel(help_text, title="[bold]CATIA agent — help[/bold]", border_style="cyan")
            )
            return True

        if verb == "/dashboard":
            try:
                from catia.dashboard import run_dashboard
            except ImportError as e:
                raise click.ClickException(
                    f"Dash required for dashboard: {e}. Install with: pip install dash"
                ) from e

            host = "127.0.0.1"
            port = 8050
            verbose = False
            i = 0
            while i < len(argv):
                a = argv[i]
                if a == "--host" and i + 1 < len(argv):
                    host = argv[i + 1]
                    i += 2
                    continue
                if a == "--port" and i + 1 < len(argv):
                    port = int(argv[i + 1])
                    i += 2
                    continue
                if a in ("-v", "--verbose"):
                    verbose = True
                    i += 1
                    continue
                raise click.UsageError(f"Unknown /dashboard argument: {a}")

            if session.dashboard_thread is not None and session.dashboard_thread.is_alive():
                url = f"http://{session.dashboard_host}:{session.dashboard_port}"
                console.print(
                    Panel(
                        Text(
                            f"Dashboard thread already running — {url}\n"
                            "Stop the REPL or use another terminal: "
                            "catia-agent dashboard --port <other>",
                            style="yellow",
                        ),
                        title="Dashboard",
                        border_style="cyan",
                    )
                )
                return True

            session.dashboard_host = host
            session.dashboard_port = port

            def _run_dash() -> None:
                run_dashboard(host=host, port=port, debug=verbose)

            th = threading.Thread(
                target=_run_dash,
                name="catia-dashboard",
                daemon=True,
            )
            session.dashboard_thread = th
            th.start()
            await asyncio.sleep(0.25)
            url = f"http://{host}:{port}"
            console.print(
                Panel(
                    Text.from_markup(
                        "[bold]Dash server[/bold] started in a background thread.\n"
                        f"Open [link={url}]{url}[/link] in a browser.\n"
                        "[dim]API figures (if used) expect catia-agent api on :8000 unless you reconfigure.[/dim]"
                    ),
                    title="Dashboard",
                    border_style="cyan",
                )
            )
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
            if argv:
                path = argv[0]
            else:
                default_spec = Path("examples/runs/baseline.yaml")
                if default_spec.is_file():
                    path = str(default_spec)
                else:
                    raise click.UsageError(
                        "/spec needs a RunSpec path, e.g. examples/runs/baseline.yaml "
                        f"(missing default: {default_spec})"
                    )
            spec = load_run_spec(path)

            raw = await _run_blocking(
                console,
                "Full pipeline (RunSpec)…",
                run_catia_analysis,
                **spec.to_kwargs(),
            )
            session.last_result = raw
            session.last_label = f"run_spec:{path}"
            rm = raw["risk_metrics"]["risk_metrics"]
            ds = raw["risk_metrics"]["descriptive_stats"]
            rows = [
                ("RunSpec", path),
                ("Mean annual loss", f"${ds['mean']:,.0f}"),
                ("VaR (95%)", f"${rm['var']:,.0f}"),
                ("TVaR (95%)", f"${rm['tvar']:,.0f}"),
            ]
            _summary_table(
                console,
                "Pipeline complete",
                rows,
                hint="Use /json for full structured output.",
            )
            return True

        if verb == "/risk":
            region = "US_Gulf_Coast"
            perils: List[str] = list(DEFAULT_PERILS)
            use_mock = True
            i = 0
            while i < len(argv):
                a = argv[i]
                if a in ("--region", "-r") and i + 1 < len(argv):
                    region = argv[i + 1]
                    i += 2
                    continue
                if a in ("--perils", "-p"):
                    perils, i = _repl_take_perils(argv, i, a)
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
            ms = result.model_summary
            rows = [
                ("Region", result.region),
                ("Perils", ", ".join(result.perils)),
                ("Data", "Mock" if result.use_mock_data else "Live"),
                ("Climate rows", f"{len(result.data['climate']):,}"),
                ("Historical events", f"{len(result.data['historical_events']):,}"),
                (
                    "Probability model",
                    str(ms.get("probability_model") or "—"),
                ),
                ("Severity model", str(ms.get("severity_model") or "—")),
            ]
            _summary_table(
                console,
                "Risk analysis",
                rows,
                hint="Use /json for full structured output.",
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
                if a in ("--perils", "-p"):
                    perils, i = _repl_take_perils(argv, i, a)
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
            arm = out.aggregate_metrics["risk_metrics"]
            ads = out.aggregate_metrics["descriptive_stats"]
            scen = scenario_id or "—"
            rows = [
                ("Perils", ", ".join(out.perils)),
                ("Scenario", scen),
                ("Mean annual loss", f"${ads['mean']:,.0f}"),
                ("VaR (95%)", f"${arm['var']:,.0f}"),
                ("TVaR (95%)", f"${arm['tvar']:,.0f}"),
            ]
            _summary_table(
                console,
                "Actuarial simulation",
                rows,
                hint="Use /json for more metrics.",
            )
            return True

        if verb == "/run":
            region = "US_Gulf_Coast"
            perils: List[str] = list(DEFAULT_PERILS)
            use_mock = True
            scenario_id: Optional[str] = None
            iterations: Optional[int] = None
            output_dir: Optional[str] = None
            run_artifacts: Optional[List[str]] = list(REPL_RUN_ARTIFACTS)
            i = 0
            while i < len(argv):
                a = argv[i]
                if a in ("--region", "-r") and i + 1 < len(argv):
                    region = argv[i + 1]
                    i += 2
                    continue
                if a in ("--perils", "-p"):
                    perils, i = _repl_take_perils(argv, i, a)
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
                if a in ("--output-dir", "-o") and i + 1 < len(argv):
                    output_dir = argv[i + 1]
                    i += 2
                    continue
                if a == "--full":
                    run_artifacts = None
                    i += 1
                    continue
                if a == "--artifacts":
                    names, i = _repl_take_artifact_names(argv, i, a)
                    run_artifacts = names
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
                artifacts=run_artifacts,
            )
            session.last_result = raw
            session.last_label = "full_pipeline"
            rm = raw["risk_metrics"]["risk_metrics"]
            ds = raw["risk_metrics"]["descriptive_stats"]
            mc_iters = iterations or SIMULATION_CONFIG["monte_carlo_iterations"]
            artifact_note = (
                "all (incl. static HTML)"
                if run_artifacts is None
                else ", ".join(run_artifacts)
            )
            rows = [
                ("Region", region),
                ("Perils", ", ".join(perils)),
                ("Data", "Mock" if use_mock else "Live"),
                ("Artifacts", artifact_note),
                ("Monte Carlo iterations", f"{mc_iters:,}"),
                ("Mean annual loss", f"${ds['mean']:,.0f}"),
                ("VaR (95%)", f"${rm['var']:,.0f}"),
                ("TVaR (95%)", f"${rm['tvar']:,.0f}"),
            ]
            hint = "Use /json for full structured output."
            if run_artifacts is not None:
                hint += (
                    " For Plotly HTML charts under outputs/, use /run --full "
                    "or /run --artifacts dashboard (and any other artifact keys you need)."
                )
            _summary_table(
                console,
                "Full pipeline",
                rows,
                hint=hint,
            )
            return True

        console.print(Panel(Text(f"Unknown command: {verb}", style="yellow"), title="Parser"))
        return True

    except Exception as e:
        _print_error(console, "CATIA / actuarial error", e)
        return True


async def async_repl() -> None:
    console = Console(highlight=True)
    session = AgentSession()
    _print_repl_welcome(console)
    loop = True
    while loop:
        session.prompt_count += 1
        _maybe_print_rotating_tip(console, session)
        try:
            line = await _prompt_line(console)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim red]Interrupted[/dim red] — use [green]/exit[/green] to quit.")
            continue
        loop = await dispatch_command(console, session, line)


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CATIA tools: REPL (default), or ``run`` / ``api`` / ``dashboard`` / ``repl`` like ``catia``."""
    if ctx.invoked_subcommand is None:
        asyncio.run(async_repl())


@cli.command("repl")
def repl_cmd() -> None:
    """Start the interactive agent session (same as bare ``catia-agent``)."""
    asyncio.run(async_repl())


@cli.command("dashboard")
@click.option(
    "--host",
    "dashboard_host",
    default="127.0.0.1",
    show_default=True,
    help="Bind address for the Dash server",
)
@click.option(
    "--port",
    "dashboard_port",
    default=8050,
    type=int,
    show_default=True,
    help="Port for the Dash server",
)
@click.option("-v", "--verbose", is_flag=True, help="Dash debug / verbose logging")
def dashboard_cmd(dashboard_host: str, dashboard_port: int, verbose: bool) -> None:
    """Start the Dash system dashboard (same as ``catia --dashboard``)."""
    try:
        from catia.dashboard import run_dashboard
    except ImportError as e:
        raise click.ClickException(
            f"Dash required for dashboard: {e}. Install dash: pip install dash"
        ) from e

    run_dashboard(host=dashboard_host, port=dashboard_port, debug=verbose)


@cli.command("run")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="YAML/JSON RunSpec (optional)",
)
@click.option("-r", "--region", default=None, help="Region override")
@click.option(
    "-p",
    "--peril",
    "--perils",
    "perils",
    multiple=True,
    type=click.Choice(["hurricane", "flood", "wildfire", "earthquake", "drought"]),
    help="Peril (repeat); same as ``catia -p`` / ``catia --perils``",
)
@click.option(
    "--no-mock-data",
    is_flag=True,
    help="Use real APIs where implemented (overrides config file)",
)
@click.option("-o", "--output-dir", default=None, help="Output directory")
@click.option("--scenario", "scenario_id", default=None, help="Climate scenario id")
@click.option("--iterations", type=int, default=None, help="Monte Carlo iterations")
@click.option("--seed", type=int, default=None, help="Random seed override")
@click.option(
    "--artifacts",
    multiple=True,
    type=click.Choice(sorted(KNOWN_ARTIFACTS)),
    help="Output artifact keys (repeatable); default all",
)
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Log transparency manifest before run (same as ``catia --explain``)",
)
def run_cmd(
    config_path: Optional[str],
    region: Optional[str],
    perils: Tuple[str, ...],
    no_mock_data: bool,
    output_dir: Optional[str],
    scenario_id: Optional[str],
    iterations: Optional[int],
    seed: Optional[int],
    artifacts: Tuple[str, ...],
    verbose: bool,
    explain: bool,
) -> None:
    """One-shot full pipeline (same behavior as ``catia`` without ``--api``/``--dashboard``)."""
    from catia.cli import _mc_iterations_warn_threshold, setup_logging

    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    spec = merge_cli_run_spec(
        config_path=config_path,
        region=region,
        perils=list(perils) if perils else None,
        no_mock_data=no_mock_data,
        output_dir=output_dir,
        scenario_id=scenario_id,
        monte_carlo_iterations=iterations,
        random_seed=seed,
        artifacts=list(artifacts) if artifacts else None,
        explain=(True if explain else None),
    )
    kw = spec.to_kwargs()
    logger.info("Running CATIA analysis (catia-agent run)...")
    for key in (
        "region",
        "perils",
        "use_mock_data",
        "scenario_id",
        "monte_carlo_iterations",
        "random_seed",
        "output_dir",
        "artifacts",
        "explain",
    ):
        logger.info("  %s: %s", key, kw.get(key))

    mc = kw.get("monte_carlo_iterations")
    thr = _mc_iterations_warn_threshold()
    if mc is not None and mc > thr:
        logger.warning(
            "monte_carlo_iterations=%s exceeds warn threshold %s "
            "(raise CATIA_MC_WARN to silence); run may take a long time.",
            mc,
            thr,
        )

    try:
        results = run_catia_analysis(**kw)
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=verbose)
        raise SystemExit(1) from e

    print(f"\n{'='*60}")
    print("CATIA Analysis Complete")
    print(f"{'='*60}")
    print(
        f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}"
    )
    print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
    print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")
    print(f"{'='*60}")


@cli.command("api")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="API bind address (use 0.0.0.0 for all interfaces; default loopback)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="API port",
)
def api_cmd(host: str, port: int) -> None:
    """Start the FastAPI server (same as ``catia --api``)."""
    try:
        import uvicorn
        from catia.api.app import app
    except ImportError as e:
        raise click.ClickException(
            f"uvicorn required for API: {e}. pip install uvicorn"
        ) from e

    logging.getLogger(__name__).info(
        "Starting CATIA API server on %s:%s", host, port
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Console script entry: same as ``click`` group."""
    cli()


if __name__ == "__main__":
    main()
