"""Tests for configurable live alert rules."""

from __future__ import annotations

from catia.live_alert_rules import LiveAlertHit, evaluate_live_rules


def test_evaluate_live_rules_min_score():
    events = [
        {
            "lat": 29.0,
            "lon": -90.0,
            "title": "Big shake",
            "catia_peril": "earthquake",
            "catia_score": 80.0,
        }
    ]
    rules = [{"id": "t", "label": "High", "min_score": 75.0}]
    hits = evaluate_live_rules(events, rules)
    assert len(hits) == 1
    assert isinstance(hits[0], LiveAlertHit)
    assert hits[0].rule_id == "t"
