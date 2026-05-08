"""
Structured disclosure for CATIA runs (open-source transparency).

Build a machine-readable manifest for ``catia_report.json`` and optional log lines
so users know **data source**, **steps**, **parameters**, and **limits**.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_pipeline_manifest(
    *,
    region: str,
    use_mock_data: bool,
    perils: List[str],
    scenario_id: Optional[str],
    output_dir: str,
    artifacts: Optional[List[str]],
    monte_carlo_iterations: int,
    random_seed: Optional[int],
    severity_distribution: str,
    catia_version: str,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable description of what the full pipeline is doing.

    Keep wording honest: mock vs API paths, coarse regions, no hidden steps.
    """
    if use_mock_data:
        data_source = (
            "Mock / synthetic: ``DataAcquisition`` generates climate, socioeconomic, "
            "and historical-event tables in-process (no live IBTrACS/HURDAT hook in "
            "the default path). Suitable for demos and CI; not a vendor cat-model replacement."
        )
    else:
        data_source = (
            "Live APIs attempted where implemented: NOAA climate (requires "
            "``NOAA_API_TOKEN``), World Bank socioeconomic; failures fall back to mock. "
            "Historical cat-event tables may still be synthetic — verify in "
            "``catia.data_acquisition``."
        )

    steps: List[str] = [
        "Data: ``fetch_all_data`` → climate DataFrame, socioeconomic DataFrame, per-peril event history",
        "Risk (ML): ``train_risk_model`` → ``RiskPredictor`` (frequency/severity targets from features)",
        "Actuarial: ``run_multi_peril_analysis`` → multi-peril Monte Carlo (config severity family, optional EVT & bootstrap uncertainty)",
        "Mitigation: ``generate_mitigation_recommendations`` from simulated baseline loss",
        "Visualization: ``create_dashboard`` static Plotly HTML bundle when ``dashboard`` artifact is enabled",
        "Reports: ``catia_report.json``, optional assumption register, compliance HTML, Phase-1 sensitivity exports per artifact filter",
    ]

    artifact_note: str
    if artifacts is None:
        artifact_note = (
            "All artifact types enabled: report, assumption_register, compliance, "
            "dashboard, enhancements"
        )
    else:
        artifact_note = f"Artifact subset only: {', '.join(artifacts)}"

    return {
        "catia_version": catia_version,
        "region": region,
        "perils": list(perils),
        "climate_scenario_id": scenario_id,
        "use_mock_data": use_mock_data,
        "data_source_summary": data_source,
        "monte_carlo_iterations": int(monte_carlo_iterations),
        "random_seed": random_seed,
        "severity_distribution_config": severity_distribution,
        "output_directory": output_dir,
        "artifacts_policy": artifact_note,
        "pipeline_steps_plain": steps,
        "limitations": [
            "Regions are named buckets (e.g. US_Gulf_Coast), not town- or asset-level geographies unless you extend the stack.",
            "Outputs are research/ops support tooling unless you validate against your own data and governance standards.",
        ],
        "read_more": "notebooks/docs/transparency.md",
        "key_modules": {
            "data": "catia.data_acquisition",
            "ml_risk": "catia.risk_prediction",
            "actuarial": "catia.financial_impact",
            "mitigation": "catia.mitigation",
            "orchestration": "catia.pipeline",
        },
    }


def log_pipeline_manifest(manifest: Dict[str, Any], log: Optional[logging.Logger] = None) -> None:
    """Emit a clear pre-run summary at INFO (use with ``--explain`` or ``explain=True``)."""
    lg = log or logger
    lg.info("=" * 72)
    lg.info("CATIA transparency — this run will:")
    for i, line in enumerate(manifest["pipeline_steps_plain"], 1):
        lg.info("  %s. %s", i, line)
    lg.info("Data: %s", manifest["data_source_summary"])
    lg.info(
        "Parameters: region=%s perils=%s scenario=%s MC_iter=%s seed=%s severity=%s",
        manifest["region"],
        ",".join(manifest["perils"]),
        manifest["climate_scenario_id"],
        manifest["monte_carlo_iterations"],
        manifest["random_seed"],
        manifest["severity_distribution_config"],
    )
    lg.info("Outputs: %s → %s", manifest["artifacts_policy"], manifest["output_directory"])
    lg.info("=" * 72)
