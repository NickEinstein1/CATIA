"""
CATIA Command Line Interface

Entry point for the CATIA package when installed via pip.
Usage: catia [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from catia import __version__
from catia.config import LOGGING_CONFIG
from catia.pipeline import run_catia_analysis
from catia.run_spec import KNOWN_ARTIFACTS, RunSpec, load_run_spec


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else LOGGING_CONFIG["level"]
    logging.basicConfig(level=level, format=LOGGING_CONFIG["format"])


def _merge_run_spec(args: argparse.Namespace) -> RunSpec:
    """Build a RunSpec from optional --config and CLI overrides."""
    base = load_run_spec(args.config) if args.config else RunSpec()
    updates: Dict[str, Any] = {}

    if args.region is not None:
        updates["region"] = args.region
    if args.perils is not None:
        updates["perils"] = list(args.perils)
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.scenario is not None:
        updates["scenario_id"] = args.scenario
    if args.iterations is not None:
        updates["monte_carlo_iterations"] = args.iterations
    if args.seed is not None:
        updates["random_seed"] = args.seed
    if args.artifacts is not None:
        updates["artifacts"] = list(args.artifacts)

    use_mock = base.use_mock_data
    if args.no_mock_data:
        use_mock = False
    updates["use_mock_data"] = use_mock

    return base.model_copy(update=updates)


def main():
    """Main CLI entry point."""
    art_list = ", ".join(sorted(KNOWN_ARTIFACTS))
    parser = argparse.ArgumentParser(
        prog="catia",
        description="CATIA: Catastrophe AI System for Climate Risk Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  catia --region US_Gulf_Coast --perils hurricane flood
  catia --config examples/runs/baseline.yaml -v
  catia --config examples/runs/minimal_report.yaml --output-dir ./my_run
  catia --scenario high_stress --iterations 5000 --seed 42
  catia --api --port 8000
  catia --dashboard --dashboard-port 8050
  catia --version

Artifact keys for --artifacts (default: all): {art_list}
        """,
    )

    parser.add_argument(
        "--version", "-V", action="version", version=f"CATIA v{__version__}"
    )

    parser.add_argument(
        "--region",
        "-r",
        default=None,
        help="Geographic region (default: US_Gulf_Coast, or value from --config)",
    )

    parser.add_argument(
        "--perils",
        "-p",
        nargs="+",
        default=None,
        choices=["hurricane", "flood", "wildfire", "earthquake", "drought"],
        help="Perils to analyze (default: package default, or --config)",
    )

    parser.add_argument(
        "--no-mock-data",
        action="store_true",
        help="Use real API data where implemented (requires keys); overrides config file",
    )

    parser.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="FILE",
        help="Run specification YAML or JSON (see examples/runs/)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Directory for outputs (overrides config file and OUTPUT_CONFIG default)",
    )

    parser.add_argument(
        "--scenario",
        default=None,
        metavar="ID",
        help="Climate scenario id (e.g. baseline, RCP4.5_mid, high_stress)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Monte Carlo iterations for this run (temporary override)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed override for this run (temporary override)",
    )

    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=None,
        choices=sorted(KNOWN_ARTIFACTS),
        metavar="NAME",
        help="Which output artifacts to write (default: all)",
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the FastAPI server instead of running analysis",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the Dash system dashboard (browse metrics, charts, assumptions)",
    )

    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8050,
        help="Dashboard port when using --dashboard (default 8050)",
    )

    parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind address (default 127.0.0.1)",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.api:
        try:
            import uvicorn
            from catia.api.app import app

            logger.info("Starting CATIA API server on %s:%s", args.host, args.port)
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)
        return

    if args.dashboard:
        try:
            from catia.dashboard import run_dashboard

            logger.info(
                "Starting CATIA dashboard on %s:%s",
                args.dashboard_host,
                args.dashboard_port,
            )
            run_dashboard(
                host=args.dashboard_host,
                port=args.dashboard_port,
                debug=args.verbose,
            )
        except ImportError as e:
            logger.error("Dash not available: %s. pip install dash", e)
            sys.exit(1)
        return

    try:
        spec = _merge_run_spec(args)
        kw = spec.to_kwargs()
        logger.info("Running CATIA analysis...")
        for k in (
            "region",
            "perils",
            "use_mock_data",
            "scenario_id",
            "monte_carlo_iterations",
            "random_seed",
            "output_dir",
            "artifacts",
        ):
            logger.info("  %s: %s", k, kw.get(k))

        results = run_catia_analysis(**kw)

        print(f"\n{'='*60}")
        print("CATIA Analysis Complete")
        print(f"{'='*60}")
        print(
            f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}"
        )
        print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
        print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
