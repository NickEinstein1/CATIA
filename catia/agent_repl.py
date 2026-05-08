"""
Terminal agent-style interface for CATIA (Click + Rich).

Async REPL: structured ``/commands`` and lightweight natural-language routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from catia.agent_bridge import ActuarialResult, ActuarialScience, RiskAnalysis, RiskAnalysisResult
from catia.config import DEFAULT_PERILS, PERIL_CONFIG, SIMULATION_CONFIG
from catia.pipeline import run_catia_analysis
from catia.run_spec import KNOWN_ARTIFACTS, load_run_spec, merge_cli_run_spec

# One-line hints (plain text); shown in /tips and rotated occasionally before the prompt.
AGENT_TIPS: Tuple[str, ...] = (
    "Default runs use mock/synthetic data — fine for demos; pair --real with NOAA_API_TOKEN for climate pulls.",
    "After /run or /risk, type /json to inspect results; catia_report.json includes metadata.transparency.",
    "Shell: catia-agent run --explain (or CATIA_EXPLAIN=1) prints what the pipeline will do before it runs.",
    "Faster iteration: catia-agent run -p hurricane only, or use examples/runs/minimal_report.yaml with artifacts: [report].",
    "Regions are coarse labels (e.g. US_Gulf_Coast), not city polygons — extend the stack for asset-level work.",
    "/simulate skips mitigation and reports; use /run for the full pipeline to outputs/.",
    "Natural language: include peril names (hurricane, flood) and words like simulate, train model, or gulf coast.",
    "Stuck? python -m catia.agent_repl avoids PATH issues on Windows if catia-agent is not found.",
    "Dashboard & API: catia-agent dashboard and catia-agent api mirror catia --dashboard and catia --api.",
    "SHAP feature importance needs CATIA_USE_SHAP=1 and runs in the full /run path when SHAP is installed.",
    "/spec accepts YAML or JSON — keep a RunSpec in version control for repeatable analyses.",
    "Higher Monte Carlo iterations (--iterations or spec) mean smoother tails but slower runs; start low, then scale up.",
    "Artifacts list in RunSpec (report, dashboard, shap) controls what files land under output_dir — skip what you do not need.",
    "PYTHONWARNINGS=default surfaces deprecation noise; use logging level DEBUG only when tracing pipeline steps.",
    "pytest tests/ and examples/… configs are the fastest check after changing peril or region logic.",
)

# Rich markup label for Prompt.ask inside the async REPL (defined once for reuse).
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
        "[green]/run[/green] [dim]-r US_Gulf_Coast -p hurricane -p flood[/dim]\n"
        "[green]/run[/green] [dim]--scenario high_stress --iterations 3000[/dim]\n"
        "[magenta]simulate hurricane flood for the gulf coast[/magenta]  "
        "[dim]— natural language → actuarial MC[/dim]\n"
        "[magenta]train the risk model for flood[/magenta]  "
        "[dim]— data + ML only[/dim]\n"
        "[green]/risk[/green] [dim]--region US_East_Coast --perils hurricane[/dim]\n"
        "[green]/spec[/green] [dim]examples/runs/baseline.yaml[/dim]\n"
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
        "[dim]or plain English.[/dim]"
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


def interpret_natural_language(text: str) -> Tuple[str, List[str]]:
    """
    Map free text to a pseudo-command and args for the dispatcher.

    Returns (verb, argv) where argv excludes the verb.
    """
    t = text.strip()
    low = t.lower()
    toks = low.split()

    if "tips" in toks or (len(toks) == 1 and toks[0] in ("tip", "hint", "hints")):
        return "tips", []
    if "tip" in toks and any(w in toks for w in ("show", "list", "more", "give")):
        return "tips", []

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
        self.prompt_count: int = 0


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

        if verb == "/tips":
            _print_all_tips(console)
            return True

        if verb == "/help":
            help_text = Text.from_markup(
                "[bold cyan]Structured[/bold cyan] [dim](prefix [green]/[/green]):[/dim]\n"
                "  [green]/run[/green]  [dim][-r|--region R] [-p|--perils P …] [--real] [--scenario S] "
                "[--iterations N] [-o|--output-dir D][/dim]\n"
                "  [green]/risk[/green]   [dim][-r|--region R] [-p|--perils P …] [--real|--mock][/dim]\n"
                "  [green]/simulate[/green]  [dim][-p|--perils P …] [--scenario S] [--iterations N] "
                "[--no-uncertainty][/dim]\n"
                "  [green]/spec[/green] [yellow]PATH[/yellow]  [dim]— YAML/JSON [RunSpec][/dim]\n"
                "  [green]/json[/green]  [green]/tips[/green]  [green]/help[/green]  [green]/exit[/green]\n\n"
                "[bold cyan]Shell[/bold cyan] [dim](outside this REPL):[/dim]\n"
                "  [magenta]catia-agent run …[/magenta]   [magenta]catia-agent api …[/magenta]   "
                "[magenta]catia-agent dashboard …[/magenta]\n\n"
                "[bold cyan]Natural language[/bold cyan] [dim](no [green]/[/green]):[/dim] name "
                "[yellow]perils[/yellow], [yellow]gulf coast[/yellow], "
                "[yellow]simulate[/yellow], [yellow]train model[/yellow], [yellow]full pipeline[/yellow], "
                "[yellow]tips[/yellow]."
            )
            console.print(
                Panel(help_text, title="[bold]CATIA agent — help[/bold]", border_style="cyan")
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
