"""
Climate scenario layer: apply forward-looking scenario adjustments to peril parameters.

Maps scenario IDs (e.g. RCP4.5, SSP2, high_stress) to frequency and severity
multipliers so the same pipeline can run baseline vs. scenario-based risk.
"""

import logging
import math
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_scenarios() -> Dict[str, Any]:
    try:
        from catia.config import CLIMATE_SCENARIOS
        return dict(CLIMATE_SCENARIOS)
    except ImportError:
        return {}


def get_scenario_info(scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return scenario metadata. If scenario_id is None, return all scenarios.
    """
    scenarios = _get_scenarios()
    if scenario_id is None:
        return {"scenarios": scenarios, "default": "baseline"}
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario_id: {scenario_id}. Known: {list(scenarios.keys())}")
    return {
        "scenario_id": scenario_id,
        "description": scenarios[scenario_id].get("description", ""),
        "peril_adjustments": scenarios[scenario_id].get("peril_adjustments", {}),
    }


def get_scenario_adjustments(scenario_id: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    Return per-peril adjustments for a scenario: frequency_multiplier, severity_multiplier.

    Returns empty dict if scenario_id is None or 'baseline'. Otherwise returns
    { peril: { "frequency_multiplier": float, "severity_multiplier": float } }.
    """
    if not scenario_id or scenario_id == "baseline":
        return {}
    info = get_scenario_info(scenario_id)
    adj = info.get("peril_adjustments", {})
    return {
        p: {
            "frequency_multiplier": v.get("frequency_multiplier", 1.0),
            "severity_multiplier": v.get("severity_multiplier", 1.0),
        }
        for p, v in adj.items()
    }


def apply_scenario_to_peril_config(
    peril: str,
    frequency_base: float,
    severity_params: Dict[str, Any],
    scenario_id: Optional[str],
) -> tuple[float, Dict[str, Any]]:
    """
    Apply scenario adjustments to a single peril's frequency and severity params.

    Severity multiplier is applied by scaling the lognormal scale (exp(mu)) so
    expected severity scales; for other distributions we scale 'scale' if present.

    Returns (adjusted_frequency_base, adjusted_severity_params).
    """
    adjustments = get_scenario_adjustments(scenario_id)
    if not adjustments or peril not in adjustments:
        return frequency_base, severity_params

    mult_f = adjustments[peril].get("frequency_multiplier", 1.0)
    mult_s = adjustments[peril].get("severity_multiplier", 1.0)
    new_freq = frequency_base * mult_f

    sev = deepcopy(severity_params)
    if mult_s == 1.0:
        return new_freq, sev
    # Lognormal: E[X] = exp(mu + sigma^2/2). Scale E[X] by mult_s => new_mu = mu + log(mult_s)
    if "mu" in sev or "body_mu" in sev:
        sev["mu"] = sev.get("mu", 15) + math.log(mult_s)
        if "body_mu" in sev:
            sev["body_mu"] = sev["body_mu"] + math.log(mult_s)
    elif "scale" in sev:
        sev["scale"] = sev["scale"] * mult_s
    return new_freq, sev
