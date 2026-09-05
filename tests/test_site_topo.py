"""Tests for DEM / floodplain topography providers and site hazard wiring."""

from __future__ import annotations

import pytest

from catia.site_hazards import build_hazard_assessment, resolve_topography, topography_stub
from catia.site_topo import (
    _classify_fema_zone,
    estimate_slope_percent,
    fetch_fema_flood_zone,
    fetch_open_meteo_elevation,
    fetch_site_terrain,
    fetch_usgs_elevation,
    topo_providers_enabled,
)


def test_classify_fema_ae_is_sfha():
    hit = _classify_fema_zone("AE", "", "T")
    assert hit["sfha"] is True
    assert hit["flood_risk_class"] == "sfha_high"
    assert hit["floodplain_hint"] == "elevated_flood_sensitivity"


def test_classify_fema_x_shaded_moderate():
    hit = _classify_fema_zone("X", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD", "F")
    assert hit["sfha"] is False
    assert hit["flood_risk_class"] == "moderate_0_2_pct"


def test_topo_stub_still_works():
    topo = topography_stub(29.95, -90.07, "US_Gulf_Coast")
    assert topo["data_quality"] == "heuristic_stub"


def test_resolve_topography_offline_stub(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "0")
    topo = resolve_topography(29.95, -90.07, "US_Gulf_Coast")
    assert topo["data_quality"] == "heuristic_stub"


def test_fetch_usgs_elevation_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("CATIA_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CATIA_SITE_TOPO", "1")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"value": "2.4", "resolution": 1}

    class _Sess:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("catia.site_topo._session", lambda: _Sess())
    hit = fetch_usgs_elevation(29.95, -90.07)
    assert hit is not None
    assert hit["elevation_m"] == 2.4
    assert hit["source"] == "usgs_epqs"
    # second call hits cache
    hit2 = fetch_usgs_elevation(29.95, -90.07)
    assert hit2["elevation_m"] == 2.4


def test_fetch_open_meteo_elevation_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("CATIA_CACHE_DIR", str(tmp_path))

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"elevation": [312.0]}

    class _Sess:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("catia.site_topo._session", lambda: _Sess())
    hit = fetch_open_meteo_elevation(48.85, 2.35)
    assert hit["elevation_m"] == 312.0
    assert hit["source"] == "open_meteo"


def test_fetch_fema_flood_zone_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("CATIA_CACHE_DIR", str(tmp_path))

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "features": [
                    {
                        "attributes": {
                            "FLD_ZONE": "AE",
                            "ZONE_SUBTY": "",
                            "SFHA_TF": "T",
                            "STATIC_BFE": 12.5,
                        },
                    }
                ]
            }

    class _Sess:
        def get(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("catia.site_topo._session", lambda: _Sess())
    hit = fetch_fema_flood_zone(29.95, -90.07)
    assert hit is not None
    assert hit["fld_zone"] == "AE"
    assert hit["sfha"] is True
    assert hit["source"] == "fema_nfhl"


def test_fetch_site_terrain_composes(monkeypatch, tmp_path):
    monkeypatch.setenv("CATIA_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CATIA_SITE_TOPO", "1")

    monkeypatch.setattr(
        "catia.site_topo.fetch_elevation",
        lambda lat, lon: {"elevation_m": 3.0, "source": "usgs_epqs"},
    )
    monkeypatch.setattr(
        "catia.site_topo.estimate_slope_percent",
        lambda lat, lon: {
            "slope_percent": 1.2,
            "slope_class": "flat_to_gentle",
            "samples": 5,
            "center_elevation_m": 3.0,
        },
    )
    monkeypatch.setattr(
        "catia.site_topo.fetch_fema_flood_zone",
        lambda lat, lon: {
            "fld_zone": "AE",
            "sfha": True,
            "flood_risk_class": "sfha_high",
            "floodplain_hint": "elevated_flood_sensitivity",
            "source": "fema_nfhl",
        },
    )
    pack = fetch_site_terrain(29.95, -90.07)
    assert pack["ok"] is True
    assert "usgs_epqs" in pack["providers_used"]
    assert "fema_nfhl" in pack["providers_used"]


def test_hazard_assessment_uses_sfha_boost(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "1")

    def _fake_resolve(lat, lon, region_id):
        return {
            "elevation_m": 2.0,
            "elevation_m_estimate": 2.0,
            "elevation_band": "lowland_coastal",
            "slope_percent": 1.0,
            "slope_class": "flat_to_gentle",
            "coastal_proximity_hint": True,
            "floodplain_hint": "elevated_flood_sensitivity",
            "flood_zone": "AE",
            "sfha": True,
            "flood_risk_class": "sfha_high",
            "wildfire_terrain_hint": False,
            "seismic_terrain_hint": False,
            "data_quality": "dem_floodplain",
            "providers": ["usgs_epqs", "fema_nfhl"],
            "notes": "test",
        }

    monkeypatch.setattr("catia.site_hazards.resolve_topography", _fake_resolve)
    out = build_hazard_assessment(
        lat=29.95,
        lon=-90.07,
        region_id="US_Gulf_Coast",
        property_type="buy_building",
    )
    flood = next(h for h in out["hazards"] if h["peril"] == "flood")
    assert flood["topography_factor"] >= 1.4
    assert "FEMA AE" in flood["notes"]
    assert out["topography"]["sfha"] is True


def test_estimate_slope_percent(monkeypatch, tmp_path):
    monkeypatch.setenv("CATIA_CACHE_DIR", str(tmp_path))

    def _elev(lat, lon):
        # Rising to the north
        base = 10.0 + (lat - 29.95) * 111320.0 * 0.05  # ~5% grade northward
        return {"elevation_m": base, "source": "usgs_epqs"}

    monkeypatch.setattr("catia.site_topo.fetch_elevation", _elev)
    slope = estimate_slope_percent(29.95, -90.07)
    assert slope is not None
    assert slope["slope_percent"] > 0
    assert slope["slope_class"] in (
        "flat_to_gentle",
        "gentle",
        "moderate",
        "steep_or_varied",
        "unknown",
    )


def test_topo_providers_enabled_flag(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "0")
    assert topo_providers_enabled() is False
    monkeypatch.setenv("CATIA_SITE_TOPO", "1")
    assert topo_providers_enabled() is True
