"""Tests for live→book, reinsurance layers, and portfolio export pack."""

from __future__ import annotations

import zipfile
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

from catia.live_portfolio import live_event_to_one_storm
from catia.portfolio_accumulation import analyze_portfolio
from catia.portfolio_export import build_portfolio_export_zip
from catia.reinsurance import apply_layers_to_loss, apply_layers_to_losses, parse_layers

SAMPLE_CSV = """id,lat,lon,tiv,flood_zone
nola-1,29.95,-90.07,5000000,AE
miami-1,25.76,-80.19,8000000,X
"""


def test_reinsurance_xl_recovery():
    out = apply_layers_to_loss(
        3_000_000,
        [{"attachment": 1_000_000, "limit": 5_000_000, "share": 1.0, "name": "XL"}],
    )
    assert out["gross_loss"] == 3_000_000
    assert out["recovered"] == 2_000_000
    assert out["net_loss"] == 1_000_000


def test_reinsurance_share():
    out = apply_layers_to_loss(
        3_000_000,
        [{"attachment": 1_000_000, "limit": 5_000_000, "share": 0.5}],
    )
    assert out["recovered"] == 1_000_000
    assert out["net_loss"] == 2_000_000


def test_reinsurance_vectorized_net_mean_below_gross():
    layers = parse_layers([{"attachment": 500_000, "limit": 2_000_000, "share": 1.0}])
    gross = np.array([0.0, 600_000.0, 3_000_000.0])
    applied = apply_layers_to_losses(gross, layers)
    assert applied["mean_net"] < applied["mean_gross"]
    assert float(applied["net_losses"][0]) == 0.0
    assert float(applied["net_losses"][1]) == 500_000.0


def test_live_event_to_one_storm_earthquake():
    event = {
        "id": "usgs-1",
        "lat": 29.9,
        "lon": -90.1,
        "title": "M 6.2 - Gulf",
        "category": "earthquake",
        "severity_label": "M 6.2",
        "source": "USGS",
    }
    spec = live_event_to_one_storm(event)
    assert spec["peril"] == "earthquake"
    assert spec["mode"] == "radius"
    assert abs(spec["intensity"] - 6.2) < 1e-6
    assert spec["center_lat"] == 29.9
    assert spec["intensity_source"] == "event_magnitude"


def test_live_event_hurricane_default_intensity():
    event = {
        "id": "eonet-1",
        "lat": 25.0,
        "lon": -80.0,
        "title": "Severe Storms",
        "category": "severeStorms",
        "source": "EONET",
        "catia_score": 80,
    }
    spec = live_event_to_one_storm(event)
    assert spec["peril"] == "hurricane"
    assert spec["radius_km"] >= 100
    assert spec["intensity"] > 0


def test_analyze_portfolio_with_xl_and_live():
    event = {
        "id": "eq-nola",
        "lat": 29.95,
        "lon": -90.07,
        "title": "M 5.8 Near NOLA",
        "category": "earthquake",
        "severity_label": "M 5.8",
        "source": "USGS",
    }
    result = analyze_portfolio(
        csv_text=SAMPLE_CSV,
        num_iterations=150,
        include_fema=False,
        live_event=event,
        reinsurance_layers=[
            {"attachment": 500_000, "limit": 10_000_000, "share": 1.0, "name": "Cat XL"}
        ],
    )
    assert result["one_storm"]["live_event_id"] == "eq-nola"
    assert result["one_storm"]["hit_count"] >= 1
    assert "reinsurance" in result["one_storm"]
    assert result["one_storm"]["reinsurance"]["net_loss"] <= result["one_storm"]["total_loss"]
    assert result["book"]["aggregate_net"] is not None
    assert result["reinsurance_layers"]


def test_export_pack_zip_contents():
    result = analyze_portfolio(
        csv_text=SAMPLE_CSV,
        num_iterations=100,
        include_fema=False,
        one_storm={
            "peril": "hurricane",
            "intensity": 130,
            "mode": "region",
            "region_id": "US_Gulf_Coast",
        },
        reinsurance_layers=[{"attachment": 1e6, "limit": 5e6, "share": 1.0}],
    )
    data, name = build_portfolio_export_zip(result)
    assert name.endswith(".zip")
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = set(zf.namelist())
    assert "portfolio_report.json" in names
    assert "summary.csv" in names
    assert "portfolio_pack.html" in names
    assert "one_storm.csv" in names


def test_api_live_hit_and_export():
    from catia.api.app import app

    client = TestClient(app)
    event = {
        "id": "api-eq",
        "lat": 29.95,
        "lon": -90.07,
        "title": "M 6.0",
        "category": "earthquake",
        "severity_label": "M 6.0",
        "source": "USGS",
    }
    hit = client.post(
        "/api/v1/portfolio/live-hit",
        json={
            "csv_text": SAMPLE_CSV,
            "event": event,
            "run_simulation": False,
            "reinsurance_layers": [
                {"attachment": 100000, "limit": 5000000, "share": 1.0}
            ],
        },
    )
    assert hit.status_code == 200, hit.text
    body = hit.json()
    assert body["one_storm"]["hit_count"] >= 1
    assert body["one_storm"]["reinsurance"]["net_loss"] <= body["one_storm"]["total_loss"]

    exp = client.post(
        "/api/v1/portfolio/export",
        json={
            "csv_text": SAMPLE_CSV,
            "num_iterations": 100,
            "live_event": event,
            "reinsurance_layers": [
                {"attachment": 100000, "limit": 5000000, "share": 1.0}
            ],
        },
    )
    assert exp.status_code == 200, exp.text
    assert exp.headers["content-type"].startswith("application/zip")
    assert zipfile.is_zipfile(BytesIO(exp.content))


def test_parse_layers_rejects_bad_share():
    with pytest.raises(ValueError):
        parse_layers([{"attachment": 0, "limit": 1, "share": 0}])
