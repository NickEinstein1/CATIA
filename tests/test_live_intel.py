"""Tests for live intel scoring and enrichment."""

from __future__ import annotations

from catia.live_intel import enrich_and_rank_events, infer_catia_peril, score_event


def test_infer_catia_peril_earthquake():
    assert infer_catia_peril({"category": "earthquake", "category_label": "Earthquake"}) == "earthquake"


def test_infer_catia_peril_storm_maps_hurricane():
    assert infer_catia_peril({"category": "severe_storms", "category_label": "Severe Storms"}) == "hurricane"


def test_score_event_earthquake_magnitude():
    ev = {
        "category": "earthquake",
        "severity_label": "M 7.2",
        "time_iso": "2099-01-01 00:00 UTC",
        "lat": 0.0,
        "lon": 0.0,
    }
    sc, comps = score_event(ev, focal_region=None)
    assert sc >= 50.0
    assert comps["severity"] > 0.8


def test_enrich_and_rank_filters_peril():
    events = [
        {
            "id": "a",
            "lat": 10.0,
            "lon": 10.0,
            "category": "earthquake",
            "category_label": "Earthquake",
            "severity_label": "M 6.0",
            "time_iso": "2099-01-01 00:00 UTC",
            "source": "USGS",
        },
        {
            "id": "b",
            "lat": 11.0,
            "lon": 11.0,
            "category": "wildfires",
            "category_label": "Wildfires",
            "time_iso": "2099-01-01 00:00 UTC",
            "source": "EONET",
        },
    ]
    out = enrich_and_rank_events(events, peril_filter="earthquake")
    assert len(out) == 1
    assert out[0]["catia_peril"] == "earthquake"
