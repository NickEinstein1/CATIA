"""
Assumption register: documented list of assumptions with version and rationale.

Every frequency, severity, correlation, and vulnerability assumption is recorded
so results are auditable and reproducible. Export to JSON for reports and compliance.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from catia import __version__

logger = logging.getLogger(__name__)


def _get_peril_config() -> Dict[str, Any]:
    try:
        from catia.config import PERIL_CONFIG
        return dict(PERIL_CONFIG)
    except ImportError:
        return {}


def _get_simulation_config() -> Dict[str, Any]:
    try:
        from catia.config import SIMULATION_CONFIG
        return dict(SIMULATION_CONFIG)
    except ImportError:
        return {}


def _get_risk_metrics_config() -> Dict[str, Any]:
    try:
        from catia.config import RISK_METRICS
        return dict(RISK_METRICS)
    except ImportError:
        return {}


def _get_intensity_distribution() -> Dict[str, Any]:
    try:
        from catia.config import INTENSITY_DISTRIBUTION
        return dict(INTENSITY_DISTRIBUTION)
    except ImportError:
        return {}


def build_assumption_register(
    *,
    include_perils: bool = True,
    include_simulation: bool = True,
    include_risk_metrics: bool = True,
    include_intensity: bool = True,
    rationales: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build the assumption register from current config.

    Rationales can be passed to document why assumptions were chosen (e.g. for compliance).
    """
    rationales = rationales or {}
    register = {
        "schema_version": "1.0",
        "catia_version": __version__,
        "generated_at": datetime.now().isoformat(),
        "sections": [],
    }

    if include_perils:
        perils = _get_peril_config()
        register["sections"].append({
            "id": "peril_parameters",
            "name": "Peril frequency and severity parameters",
            "rationale": rationales.get("peril_parameters", "Default parameters from PERIL_CONFIG; tune per region and data."),
            "assumptions": {
                p: {
                    "frequency_base": c.get("frequency_base"),
                    "severity_params": c.get("severity_params"),
                    "regions": c.get("regions", []),
                    "magnitude_scale": c.get("magnitude_scale"),
                }
                for p, c in perils.items()
            },
        })

    if include_simulation:
        sim = _get_simulation_config()
        register["sections"].append({
            "id": "simulation",
            "name": "Monte Carlo simulation",
            "rationale": rationales.get("simulation", "Poisson frequency; configurable severity distribution and iterations."),
            "assumptions": {
                "frequency_distribution": sim.get("frequency_distribution"),
                "severity_distribution": sim.get("severity_distribution"),
                "monte_carlo_iterations": sim.get("monte_carlo_iterations"),
                "random_seed": sim.get("random_seed"),
                "spliced_threshold_percentile": sim.get("spliced_threshold_percentile"),
            },
        })

    if include_risk_metrics:
        rm = _get_risk_metrics_config()
        register["sections"].append({
            "id": "risk_metrics",
            "name": "Risk metrics definition",
            "rationale": rationales.get("risk_metrics", "Standard VaR/TVaR and return periods for capital and reporting."),
            "assumptions": dict(rm),
        })

    if include_intensity:
        try:
            intensity = _get_intensity_distribution()
            if intensity:
                register["sections"].append({
                    "id": "intensity_distribution",
                    "name": "Hazard intensity distribution (exposure-based loss)",
                    "rationale": rationales.get("intensity_distribution", "Weibull shape/scale per peril for sampling event intensity."),
                    "assumptions": intensity,
                })
        except Exception:
            pass

    return register


def write_assumption_register(
    path: str | Path,
    rationales: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build the assumption register and write to JSON.

    Returns the register dict. Creates parent dirs if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    register = build_assumption_register(rationales=rationales, **kwargs)
    with open(path, "w") as f:
        json.dump(register, f, indent=2, default=str)
    logger.info("Assumption register written to %s", path)
    return register
