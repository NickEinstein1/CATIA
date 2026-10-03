"""
Live data connectors with retries and optional caching.

Primary sources (no silent mock inside connectors — callers decide fallback policy):
- Open-Meteo archive climate (free, no token)
- NOAA CDO when NOAA_API_TOKEN is set
- World Bank socioeconomic indicators
- USGS FDSN earthquake catalog for historical EQ events
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from catia.config import API_CONFIG
from catia.geo_regions import REGION_CENTROIDS

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = os.environ.get(
    "CATIA_OPEN_METEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)
USGS_FDSN = os.environ.get(
    "CATIA_USGS_FDSN_URL",
    "https://earthquake.usgs.gov/fdsnws/event/1/query",
)

REGION_TO_ISO: Dict[str, str] = {
    "US_Gulf_Coast": "USA",
    "US_East_Coast": "USA",
    "US_West_Coast": "USA",
    "US_Midwest": "USA",
    "US_Southwest": "USA",
    "Caribbean": "JAM",
    "Japan": "JPN",
    "Europe": "DEU",
    "Mediterranean": "ITA",
    "Australia": "AUS",
    "South_America": "BRA",
    "Chile": "CHL",
    "Africa": "ZAF",
    "Southeast_Asia": "IDN",
    "South_Asia": "IND",
    "Turkey": "TUR",
    "Indonesia": "IDN",
}


def _session_with_retries(timeout: int = 30, retries: int = 3) -> requests.Session:
    s = requests.Session()
    # Bypass dead IDE proxies by default (same policy as live feeds)
    if os.environ.get("CATIA_LIVE_USE_SYSTEM_PROXY", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        s.trust_env = False
        s.proxies.update({"http": None, "https": None})
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "CATIA/2.5 (catastrophe-risk; research)"})
    return s


def region_lat_lon(region: str) -> Tuple[float, float]:
    if region in REGION_CENTROIDS:
        return REGION_CENTROIDS[region]
    return 28.5, -90.0


class OpenMeteoClimateConnector:
    """Free historical daily climate via Open-Meteo archive API."""

    def __init__(self, timeout: int = 45):
        self.timeout = timeout

    def fetch_climate(self, region: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        lat, lon = region_lat_lon(region)
        # Cap range to keep payloads reasonable
        try:
            start = datetime.strptime(start_date[:10], "%Y-%m-%d")
            end = datetime.strptime(end_date[:10], "%Y-%m-%d")
        except ValueError:
            end = datetime.utcnow()
            start = end - timedelta(days=365 * 3)
        if (end - start).days > 366 * 4:
            start = end - timedelta(days=365 * 3)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "surface_pressure_mean",
                    "relative_humidity_2m_mean",
                ]
            ),
            "timezone": "UTC",
        }
        try:
            with _session_with_retries(self.timeout) as session:
                r = session.get(OPEN_METEO_ARCHIVE, params=params, timeout=self.timeout)
                r.raise_for_status()
            payload = r.json()
            daily = payload.get("daily") or {}
            dates = daily.get("time") or []
            if not dates:
                return None
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(dates),
                    "temperature": daily.get("temperature_2m_mean"),
                    "precipitation": daily.get("precipitation_sum"),
                    "wind_speed": daily.get("wind_speed_10m_max"),
                    "sea_level_pressure": daily.get("surface_pressure_mean"),
                    "humidity": daily.get("relative_humidity_2m_mean"),
                    "region": region,
                }
            )
            df["source"] = "open-meteo"
            logger.info("Open-Meteo climate: %s rows for %s", len(df), region)
            return df
        except Exception as e:
            logger.warning("Open-Meteo climate fetch failed: %s", e)
            return None


class NOAAConnector:
    """NOAA NCEI CDO climate when NOAA_API_TOKEN is set."""

    def __init__(self, api_token: Optional[str] = None):
        self.token = api_token or os.environ.get("NOAA_API_TOKEN", "")
        self.config = API_CONFIG.get("NOAA", {})
        self.base = self.config.get("base_url", "https://www.ncei.noaa.gov").rstrip("/")
        self.timeout = self.config.get("timeout", 30)

    def fetch_climate(self, region: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not self.token:
            return None
        try:
            url = f"{self.base}/cdo-web/api/v2/data"
            params = {
                "datasetid": "GHCND",
                "locationid": "FIPS:22",  # LA — Gulf proxy; map expands later
                "startdate": start_date[:10],
                "enddate": end_date[:10],
                "limit": 1000,
                "units": "metric",
            }
            headers = {"token": self.token}
            with _session_with_retries(self.timeout) as session:
                r = session.get(url, params=params, headers=headers, timeout=self.timeout)
                r.raise_for_status()
            data = r.json()
            results = data.get("results") or []
            if not results:
                return None
            df = pd.DataFrame(results)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df["region"] = region
            df["source"] = "noaa"
            return df
        except Exception as e:
            logger.warning("NOAA fetch failed: %s", e)
            return None


class WorldBankConnector:
    """World Bank socioeconomic indicators (public API)."""

    def __init__(self):
        self.config = API_CONFIG.get("WORLD_BANK", {})
        self.base = (self.config.get("base_url", "https://api.worldbank.org/v2") or "").rstrip("/")
        self.timeout = self.config.get("timeout", 30)

    def _indicator(self, session: requests.Session, iso: str, indicator: str) -> Optional[float]:
        url = f"{self.base}/country/{iso}/indicator/{indicator}"
        params = {"format": "json", "per_page": 20, "date": "2018:2024"}
        r = session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        for rec in data[1] or []:
            if rec.get("value") is not None:
                return float(rec["value"])
        return None

    def fetch_indicators(self, country_iso: str = "USA") -> Optional[pd.DataFrame]:
        try:
            with _session_with_retries(self.timeout) as session:
                pop = self._indicator(session, country_iso, "SP.POP.TOTL")
                gdp = self._indicator(session, country_iso, "NY.GDP.PCAP.CD")
                pov = self._indicator(session, country_iso, "SI.POV.NAHC")
            if pop is None and gdp is None:
                return None
            # Rough density: national pop / country land area proxy (km²) — USA ~9.8e6
            area = {
                "USA": 9_834_000,
                "JPN": 378_000,
                "DEU": 357_000,
                "BRA": 8_516_000,
                "AUS": 7_692_000,
                "IND": 3_287_000,
                "IDN": 1_905_000,
                "CHL": 756_000,
                "ZAF": 1_221_000,
                "ITA": 301_000,
                "TUR": 783_000,
                "JAM": 11_000,
            }.get(country_iso, 500_000)
            density = (pop / area) if pop else 35.0
            # Infrastructure index heuristic from GDP band (documented as derived)
            g = float(gdp or 25000)
            infra = float(np.clip(0.35 + (g / 120_000), 0.3, 0.95))
            out = pd.DataFrame(
                [
                    {
                        "region": country_iso,
                        "population": float(pop) if pop else float(density * area),
                        "population_density": density,
                        "gdp_per_capita": g,
                        "infrastructure_index": infra,
                        "poverty_rate": float(pov / 100.0) if pov and pov > 1 else float(pov or 0.12),
                        "source": "worldbank",
                    }
                ]
            )
            logger.info("World Bank socioeconomic for %s", country_iso)
            return out
        except Exception as e:
            logger.warning("World Bank fetch failed: %s", e)
            return None


class USGSHistoricalConnector:
    """USGS FDSN event catalog → historical earthquake events near a region."""

    def __init__(self, timeout: int = 40):
        self.timeout = timeout

    def fetch_events(
        self,
        region: str,
        *,
        start_year: int = 2000,
        min_magnitude: float = 4.5,
        maxradius_km: float = 600.0,
    ) -> Optional[pd.DataFrame]:
        lat, lon = region_lat_lon(region)
        params = {
            "format": "geojson",
            "starttime": f"{start_year}-01-01",
            "endtime": datetime.utcnow().strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": maxradius_km,
            "orderby": "time",
            "limit": 2000,
        }
        try:
            with _session_with_retries(self.timeout) as session:
                r = session.get(USGS_FDSN, params=params, timeout=self.timeout)
                r.raise_for_status()
            feats = (r.json() or {}).get("features") or []
            rows: List[Dict[str, Any]] = []
            for feat in feats:
                props = feat.get("properties") or {}
                mag = props.get("mag")
                tms = props.get("time")
                if mag is None or tms is None:
                    continue
                dt = datetime.utcfromtimestamp(float(tms) / 1000.0)
                # Loss proxy from magnitude (documented model, not mock RNG catalog)
                loss = float(10 ** (1.5 * float(mag) + 4.0))
                rows.append(
                    {
                        "year": dt.year,
                        "month": dt.month,
                        "event_type": "earthquake",
                        "region": region,
                        "magnitude": float(mag),
                        "loss_usd": loss,
                        "affected_population": int(min(5_000_000, 10 ** (float(mag) - 1))),
                        "peril_name": "Earthquake",
                        "source": "usgs_fdsn",
                        "event_id": props.get("ids") or props.get("code"),
                    }
                )
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).sort_values(["year", "month"])
            logger.info("USGS historical EQ: %s events for %s", len(df), region)
            return df
        except Exception as e:
            logger.warning("USGS historical fetch failed: %s", e)
            return None


def fetch_noaa_climate_cached(
    cache: Optional[Any],
    region: str,
    start_date: str,
    end_date: str,
    connector: Optional[NOAAConnector] = None,
) -> Optional[pd.DataFrame]:
    """Prefer Open-Meteo (always available), then NOAA when token set."""
    params = {"region": region, "start_date": start_date, "end_date": end_date, "v": 2}
    if cache:
        cached = cache.get("climate_live", params)
        if cached is not None:
            return cached

    om = OpenMeteoClimateConnector()
    df = om.fetch_climate(region, start_date, end_date)
    if df is None:
        conn = connector or NOAAConnector()
        df = conn.fetch_climate(region, start_date, end_date)
    if df is not None and cache is not None and not df.empty:
        cache.set("climate_live", params, df)
    return df


def fetch_worldbank_cached(
    cache: Optional[Any],
    region: str,
    connector: Optional[WorldBankConnector] = None,
) -> Optional[pd.DataFrame]:
    iso = REGION_TO_ISO.get(region, "USA")
    params = {"region": region, "country_iso": iso, "v": 2}
    if cache:
        cached = cache.get("worldbank_socio", params)
        if cached is not None:
            return cached
    conn = connector or WorldBankConnector()
    df = conn.fetch_indicators(iso)
    if df is not None:
        df = df.copy()
        df["region"] = region
        if cache is not None:
            cache.set("worldbank_socio", params, df)
    return df


def fetch_usgs_events_cached(
    cache: Optional[Any],
    region: str,
) -> Optional[pd.DataFrame]:
    params = {"region": region, "v": 1}
    if cache:
        cached = cache.get("usgs_eq_hist", params)
        if cached is not None:
            return cached
    df = USGSHistoricalConnector().fetch_events(region)
    if df is not None and cache is not None and not df.empty:
        cache.set("usgs_eq_hist", params, df)
    return df
