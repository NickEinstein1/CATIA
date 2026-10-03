"""Tests for hazard calibration from acquired events."""

from __future__ import annotations

import pandas as pd

from catia.hazard_calibration import calibrate_peril_params
from catia.financial_impact import MultiPerilSimulator


def test_calibrate_uses_empirical_frequency():
    events = pd.DataFrame(
        {
            "year": [2000 + i // 2 for i in range(20)],
            "month": [6] * 20,
            "event_type": ["hurricane"] * 20,
            "loss_usd": [1e6 * (i + 1) for i in range(20)],
            "magnitude": [2.0] * 20,
        }
    )
    cal = calibrate_peril_params(events, ["hurricane"], min_events=5)
    assert cal["hurricane"]["n_events"] == 20
    assert cal["hurricane"]["calibration"] == "empirical_events"
    assert cal["hurricane"]["frequency_base"] > 0


def test_calibrate_falls_back_on_thin_sample():
    events = pd.DataFrame(
        {
            "year": [2020],
            "month": [8],
            "event_type": ["flood"],
            "loss_usd": [500000],
        }
    )
    cal = calibrate_peril_params(events, ["flood"], min_events=5)
    assert cal["flood"]["calibration"] == "prior_insufficient_sample"


def test_simulator_honors_overrides():
    overrides = {
        "hurricane": {
            "frequency_base": 2.5,
            "severity_params": {"mu": 14.0, "sigma": 1.5},
        }
    }
    sim = MultiPerilSimulator(
        ["hurricane"],
        use_correlation=False,
        peril_overrides=overrides,
    )
    assert sim.simulators["hurricane"].event_frequency == 2.5
