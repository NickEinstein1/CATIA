"""Unit tests for live feed parsers (no network)."""

from __future__ import annotations

from catia.live_catastrophe_feeds import _live_proxies, _parse_eonet_json, _parse_gdacs_geojson, _parse_usgs_geojson, _session


def test_parse_usgs_geojson_minimal():
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "abc",
                    "mag": 4.2,
                    "place": "Testville",
                    "time": 1_700_000_000_000,
                    "url": "https://earthquake.usgs.gov/",
                },
                "geometry": {"type": "Point", "coordinates": [-100.5, 35.2]},
            }
        ],
    }
    rows = _parse_usgs_geojson(data)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "USGS"
    assert r["category"] == "earthquake"
    assert r["lat"] == 35.2
    assert r["lon"] == -100.5
    assert "M 4.2" in r["severity_label"]
    assert r.get("geometry", {}).get("type") == "Point"
    assert r.get("provenance", {}).get("feed") == "usgs"


def test_session_bypasses_env_proxy_by_default(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:50700")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:50700")
    monkeypatch.delenv("CATIA_LIVE_USE_SYSTEM_PROXY", raising=False)
    sess = _session()
    assert sess.trust_env is False
    assert sess.proxies.get("https") is None
    assert sess.proxies.get("http") is None


def test_session_honors_system_proxy_when_enabled(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:50700")
    monkeypatch.setenv("CATIA_LIVE_USE_SYSTEM_PROXY", "1")
    sess = _session()
    assert sess.trust_env is True
    assert _live_proxies() is None


def test_parse_eonet_json_minimal():
    data = {
        "events": [
            {
                "id": "EONET1",
                "title": "Wildfire X",
                "link": "https://eonet.gsfc.nasa.gov/",
                "categories": [{"id": "wildfires", "title": "Wildfires"}],
                "geometry": [
                    {
                        "date": "2024-01-15T12:00:00Z",
                        "type": "Point",
                        "coordinates": [12.34, 56.78],
                    }
                ],
            }
        ]
    }
    rows = _parse_eonet_json(data)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "NASA EONET"
    assert "wildfire" in r["category"].lower() or "Wildfire" in r.get("category_label", "")


def test_parse_gdacs_geojson_minimal():
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [130.0, 30.0]},
                "properties": {
                    "eventtype": "EQ",
                    "eventid": "1",
                    "episodeid": "1",
                    "name": "Test quake",
                    "url": {"details": "https://www.gdacs.org/"},
                    "alertlevel": "Orange",
                    "fromdate": "2024-06-01T00:00:00+00:00",
                },
            }
        ],
    }
    rows = _parse_gdacs_geojson(data)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "GDACS"
    assert r["category"] == "earthquake"
    assert r["lat"] == 30.0
    assert r["lon"] == 130.0
