"""
Calibrate peril frequency/severity from acquired (live) historical events.

Closes the gap where Monte Carlo used only static PERIL_CONFIG while
data acquisition pulled real climate/catalogs that never reached the simulator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from catia.config import PERIL_CONFIG

logger = logging.getLogger(__name__)


def _years_span(events: pd.DataFrame) -> float:
    if events is None or events.empty or "year" not in events.columns:
        return 0.0
    years = pd.to_numeric(events["year"], errors="coerce").dropna()
    if years.empty:
        return 0.0
    return float(max(1.0, years.max() - years.min() + 1))


def _fit_lognormal(losses: np.ndarray) -> Dict[str, float]:
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses) & (losses > 0)]
    if len(losses) < 3:
        return {}
    log_x = np.log(losses)
    return {"mu": float(np.mean(log_x)), "sigma": float(max(0.35, np.std(log_x, ddof=1)))}


def calibrate_peril_params(
    historical_events: pd.DataFrame,
    perils: List[str],
    *,
    prior_weight: float = 0.35,
    min_events: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Estimate per-peril ``frequency_base`` and ``severity_params`` from events.

    Shrinks toward ``PERIL_CONFIG`` when sample size is small so thin catalogs
    do not explode Monte Carlo rates.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if historical_events is None or historical_events.empty:
        for p in perils:
            cfg = PERIL_CONFIG.get(p, {})
            out[p] = {
                "frequency_base": float(cfg.get("frequency_base", 0.5)),
                "severity_params": dict(cfg.get("severity_params", {"mu": 15, "sigma": 2})),
                "calibration": "prior_only",
                "n_events": 0,
            }
        return out

    df = historical_events.copy()
    if "event_type" not in df.columns:
        df["event_type"] = perils[0] if len(perils) == 1 else "unknown"

    for peril in perils:
        cfg = PERIL_CONFIG.get(peril, {})
        prior_freq = float(cfg.get("frequency_base", 0.5))
        prior_sev = dict(cfg.get("severity_params", {"mu": 15, "sigma": 2}))
        subset = df[df["event_type"].astype(str) == peril]
        n = int(len(subset))
        span = _years_span(subset) if n else _years_span(df)
        if n < min_events or span <= 0:
            out[peril] = {
                "frequency_base": prior_freq,
                "severity_params": prior_sev,
                "calibration": "prior_insufficient_sample",
                "n_events": n,
                "years_span": span,
            }
            continue

        emp_freq = n / span
        # Shrink frequency toward prior
        w = min(1.0, n / 40.0) * (1.0 - prior_weight)
        freq = (1.0 - w) * prior_freq + w * emp_freq
        freq = float(np.clip(freq, 0.05, 8.0))

        losses = None
        if "loss_usd" in subset.columns:
            losses = pd.to_numeric(subset["loss_usd"], errors="coerce").to_numpy()
        fitted = _fit_lognormal(losses) if losses is not None else {}
        if fitted:
            sev = {
                "mu": (1.0 - w) * float(prior_sev.get("mu", 15)) + w * fitted["mu"],
                "sigma": (1.0 - w) * float(prior_sev.get("sigma", 2)) + w * fitted["sigma"],
            }
            cal = "empirical_events"
        else:
            sev = prior_sev
            cal = "frequency_only"

        out[peril] = {
            "frequency_base": freq,
            "severity_params": sev,
            "calibration": cal,
            "n_events": n,
            "years_span": span,
            "empirical_frequency": float(emp_freq),
        }
        logger.info(
            "Calibrated %s: freq=%.3f (emp=%.3f, n=%s, years=%.0f) mode=%s",
            peril,
            freq,
            emp_freq,
            n,
            span,
            cal,
        )
    return out
