"""
Data Acquisition Module for CATIA.

Live-first by default: Open-Meteo climate, World Bank socioeconomic,
USGS historical earthquakes, and climate-extreme proxies for other perils.

Mock data is opt-in (``use_mock_data=True`` / ``CATIA_USE_MOCK_DATA=1``) for tests.
Silent mock substitution in live mode is disabled unless ``CATIA_ALLOW_MOCK_FALLBACK=1``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from catia.config import DATA_CONFIG, DEFAULT_PERILS, LOGGING_CONFIG, PERIL_CONFIG
from catia.data_policy import (
    DataUnavailableError,
    allow_mock_fallback,
    provenance_blob,
    use_mock_by_default,
)

try:
    from catia.data.cache import FileDataCache
    from catia.data.connectors import (
        fetch_noaa_climate_cached,
        fetch_usgs_events_cached,
        fetch_worldbank_cached,
    )

    _DATA_LAYER_AVAILABLE = True
except ImportError:
    FileDataCache = None
    fetch_noaa_climate_cached = fetch_worldbank_cached = fetch_usgs_events_cached = None
    _DATA_LAYER_AVAILABLE = False

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Fetch climate, socioeconomic, and historical hazard data with provenance."""

    def __init__(self, use_mock_data: Optional[bool] = None, cache: Optional[Any] = None):
        if use_mock_data is None:
            use_mock_data = use_mock_by_default()
        self.use_mock_data = bool(use_mock_data)
        self.session = self._create_session()
        self._cache = cache
        self._source_log: Dict[str, str] = {}
        self._notes: List[str] = []
        if cache is None and _DATA_LAYER_AVAILABLE and not self.use_mock_data:
            cache_dir = DATA_CONFIG.get("cache_dir", "data/cache")
            ttl = DATA_CONFIG.get("cache_ttl_seconds")
            self._cache = FileDataCache(cache_dir, ttl) if cache_dir else None
        logger.info(
            "DataAcquisition initialized (mock_data=%s, cache=%s)",
            self.use_mock_data,
            self._cache is not None,
        )

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _fail_or_mock(self, what: str, mock_fn):
        if allow_mock_fallback():
            self._notes.append(f"{what}: live failed — degraded mock (CATIA_ALLOW_MOCK_FALLBACK)")
            self._source_log[what] = "degraded_mock"
            return mock_fn()
        raise DataUnavailableError(
            f"Live {what} unavailable. Set CATIA_ALLOW_MOCK_FALLBACK=1 for degraded demos, "
            "or pass use_mock_data=True for explicit synthetic runs."
        )

    def fetch_climate_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.use_mock_data:
            self._source_log["climate"] = "mock"
            return self._generate_mock_climate_data(region, start_date, end_date)
        if _DATA_LAYER_AVAILABLE and fetch_noaa_climate_cached:
            try:
                df = fetch_noaa_climate_cached(self._cache, region, start_date, end_date)
                if df is not None and not df.empty:
                    src = str(df["source"].iloc[0]) if "source" in df.columns else "live_climate"
                    self._source_log["climate"] = src
                    return df
            except Exception as e:
                logger.warning("Live climate fetch error: %s", e)
        return self._fail_or_mock(
            "climate",
            lambda: self._generate_mock_climate_data(region, start_date, end_date),
        )

    def fetch_socioeconomic_data(self, region: str) -> pd.DataFrame:
        if self.use_mock_data:
            self._source_log["socioeconomic"] = "mock"
            return self._generate_mock_socioeconomic_data(region)
        if _DATA_LAYER_AVAILABLE and fetch_worldbank_cached:
            try:
                df = fetch_worldbank_cached(self._cache, region)
                if df is not None and not df.empty:
                    self._source_log["socioeconomic"] = "worldbank"
                    return df
            except Exception as e:
                logger.warning("Live socioeconomic fetch error: %s", e)
        return self._fail_or_mock(
            "socioeconomic",
            lambda: self._generate_mock_socioeconomic_data(region),
        )

    def fetch_historical_events(self, region: str, event_type: str) -> pd.DataFrame:
        if self.use_mock_data:
            self._source_log[f"events:{event_type}"] = "mock"
            return self._generate_mock_historical_events(region, event_type)

        if event_type == "earthquake" and _DATA_LAYER_AVAILABLE and fetch_usgs_events_cached:
            try:
                df = fetch_usgs_events_cached(self._cache, region)
                if df is not None and not df.empty:
                    self._source_log["events:earthquake"] = "usgs_fdsn"
                    return df
            except Exception as e:
                logger.warning("USGS historical error: %s", e)

        # Climate-extreme proxies for weather perils (grounded in live climate series)
        if event_type in ("hurricane", "flood", "wildfire", "drought"):
            try:
                climate = self.fetch_climate_data(region, "2015-01-01", "2024-12-31")
                derived = self._events_from_climate_extremes(region, event_type, climate)
                if derived is not None and not derived.empty:
                    self._source_log[f"events:{event_type}"] = "open-meteo_extremes"
                    self._notes.append(
                        f"{event_type}: events derived from live climate extremes "
                        "(not a vendor catastrophe catalog)."
                    )
                    return derived
            except DataUnavailableError:
                raise
            except Exception as e:
                logger.warning("Extreme-proxy events failed for %s: %s", event_type, e)

        return self._fail_or_mock(
            f"events:{event_type}",
            lambda: self._generate_mock_historical_events(region, event_type),
        )

    def _events_from_climate_extremes(
        self, region: str, event_type: str, climate: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        if climate is None or climate.empty or "date" not in climate.columns:
            return None
        df = climate.copy()
        df["date"] = pd.to_datetime(df["date"])
        peril_config = PERIL_CONFIG.get(event_type, {})
        severity_params = peril_config.get("severity_params", {"mu": 15, "sigma": 2})

        if event_type == "hurricane":
            series = pd.to_numeric(df.get("wind_speed"), errors="coerce")
            thr = float(series.quantile(0.97)) if series.notna().any() else 45.0
            mask = series >= thr
            mag = (series / 20.0).clip(1, 5)
        elif event_type == "flood":
            series = pd.to_numeric(df.get("precipitation"), errors="coerce")
            thr = float(series.quantile(0.98)) if series.notna().any() else 40.0
            mask = series >= thr
            mag = (series / 25.0).clip(1, 5)
        elif event_type == "wildfire":
            temp = pd.to_numeric(df.get("temperature"), errors="coerce")
            precip = pd.to_numeric(df.get("precipitation"), errors="coerce").fillna(0)
            mask = (temp >= temp.quantile(0.95)) & (precip <= precip.quantile(0.3))
            mag = ((temp - 20) / 8.0).clip(1, 5)
        else:  # drought
            precip = pd.to_numeric(df.get("precipitation"), errors="coerce").fillna(0)
            roll = precip.rolling(30, min_periods=10).mean()
            mask = roll <= roll.quantile(0.05)
            mag = (1.0 / (roll + 0.5)).clip(1, 5)

        hits = df.loc[mask.fillna(False)].copy()
        if hits.empty:
            return pd.DataFrame()
        # Cap density
        if len(hits) > 80:
            hits = hits.sample(80, random_state=42)
        rows = []
        for _, row in hits.iterrows():
            m = float(mag.loc[row.name]) if row.name in mag.index else 2.0
            rows.append(
                {
                    "year": int(row["date"].year),
                    "month": int(row["date"].month),
                    "event_type": event_type,
                    "region": region,
                    "magnitude": m,
                    "loss_usd": float(np.exp(severity_params["mu"] + severity_params["sigma"] * (m / 5.0))),
                    "affected_population": int(10_000 * m),
                    "peril_name": peril_config.get("name", event_type.title()),
                    "source": "open-meteo_extremes",
                }
            )
        return pd.DataFrame(rows).sort_values(["year", "month"])

    def _generate_mock_climate_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq="D")
        n = len(dates)
        rng = np.random.default_rng(abs(hash(region)) % (2**32))
        data = {
            "date": dates,
            "temperature": rng.normal(20, 5, n),
            "precipitation": np.abs(rng.normal(5, 10, n)),
            "wind_speed": np.abs(rng.normal(10, 5, n)),
            "sea_level_pressure": rng.normal(1013, 5, n),
            "humidity": np.clip(rng.normal(65, 15, n), 0, 100),
            "region": region,
            "source": "mock",
        }
        return pd.DataFrame(data)

    def _generate_mock_socioeconomic_data(self, region: str) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(region)) % (2**32))
        return pd.DataFrame(
            {
                "region": [region],
                "population": [float(rng.uniform(2_000_000, 25_000_000))],
                "population_density": [float(rng.uniform(50, 500))],
                "gdp_per_capita": [float(rng.uniform(5000, 50000))],
                "infrastructure_index": [float(rng.uniform(0.3, 0.9))],
                "poverty_rate": [float(rng.uniform(0.05, 0.3))],
                "source": ["mock"],
            }
        )

    def _generate_mock_historical_events(self, region: str, event_type: str) -> pd.DataFrame:
        peril_config = PERIL_CONFIG.get(event_type, {})
        base_freq = peril_config.get("frequency_base", 0.5)
        years_of_history = 24
        expected_events = int(base_freq * years_of_history)
        rng = np.random.default_rng(abs(hash((region, event_type))) % (2**32))
        n_events = int(rng.integers(max(3, expected_events - 5), expected_events + 10))
        seasonality = peril_config.get("seasonality", list(range(1, 13)))
        years = rng.choice(range(2000, 2024), n_events)
        months = rng.choice(seasonality, n_events)
        severity_params = peril_config.get("severity_params", {"mu": 15, "sigma": 2})
        if event_type == "earthquake":
            magnitude = rng.uniform(4, 9, n_events)
        else:
            magnitude = rng.uniform(1, 5, n_events)
        return pd.DataFrame(
            {
                "year": years,
                "month": months,
                "event_type": event_type,
                "region": region,
                "magnitude": magnitude,
                "loss_usd": rng.lognormal(severity_params["mu"], severity_params["sigma"], n_events),
                "affected_population": rng.integers(1000, 1000000, n_events),
                "peril_name": peril_config.get("name", event_type.title()),
                "source": "mock",
            }
        ).sort_values(["year", "month"])

    def fetch_multi_peril_events(self, region: str, perils: List[str] = None) -> pd.DataFrame:
        perils = perils or DEFAULT_PERILS
        all_events = []
        for peril in perils:
            if peril in PERIL_CONFIG:
                all_events.append(self.fetch_historical_events(region, peril))
            else:
                logger.warning("Unknown peril type: %s", peril)
        if all_events:
            combined = pd.concat(all_events, ignore_index=True)
            return combined.sort_values(["year", "month"]).reset_index(drop=True)
        return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame, data_type: str = "climate") -> Tuple[pd.DataFrame, Dict]:
        report = {
            "original_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "outliers_removed": 0,
            "data_type": data_type,
        }
        if DATA_CONFIG["data_validation"]["check_missing_values"]:
            # Don't drop source column-only NaNs aggressively on event frames
            essential = [c for c in df.columns if c not in ("source", "event_id", "peril_name")]
            df = df.dropna(subset=[c for c in essential if c in df.columns])
            report["rows_after_missing_removal"] = len(df)
        if DATA_CONFIG["data_validation"]["check_outliers"] and len(df) > 2:
            threshold = DATA_CONFIG["data_validation"]["outlier_threshold"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ("year", "month", "magnitude"):
                    continue
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    mask = np.abs((df[col] - mean) / std) <= threshold
                    report["outliers_removed"] += int((~mask).sum())
                    df = df[mask]
        return df, report

    def provenance(self) -> Dict[str, Any]:
        mode = "mock" if self.use_mock_data else (
            "degraded_mock" if any(v == "degraded_mock" for v in self._source_log.values()) else "live"
        )
        return provenance_blob(mode=mode, sources=dict(self._source_log), notes=list(self._notes))


def fetch_all_data(
    region: str,
    use_mock: Optional[bool] = None,
    perils: List[str] = None,
) -> Dict:
    """
    Fetch all required data for a region.

    ``use_mock`` defaults to the process policy (live-first unless CATIA_USE_MOCK_DATA=1).
    """
    if use_mock is None:
        use_mock = use_mock_by_default()
    da = DataAcquisition(use_mock_data=use_mock)
    perils = perils or DEFAULT_PERILS

    climate_data = da.fetch_climate_data(region, "2020-01-01", "2023-12-31")
    socioeconomic_data = da.fetch_socioeconomic_data(region)
    historical_events = da.fetch_multi_peril_events(region, perils)

    events_by_peril = {}
    for peril in perils:
        events_by_peril[peril] = da.fetch_historical_events(region, peril)

    climate_data, _ = da.validate_data(climate_data, "climate")
    socioeconomic_data, _ = da.validate_data(socioeconomic_data, "socioeconomic")
    if not historical_events.empty:
        historical_events, _ = da.validate_data(historical_events, "events")

    return {
        "climate": climate_data,
        "socioeconomic": socioeconomic_data,
        "historical_events": historical_events,
        "events_by_peril": events_by_peril,
        "perils_analyzed": perils,
        "provenance": da.provenance(),
    }


def fetch_single_peril_data(region: str, peril: str, use_mock: Optional[bool] = None) -> Dict:
    if use_mock is None:
        use_mock = use_mock_by_default()
    da = DataAcquisition(use_mock_data=use_mock)
    climate_data = da.fetch_climate_data(region, "2020-01-01", "2023-12-31")
    socioeconomic_data = da.fetch_socioeconomic_data(region)
    historical_events = da.fetch_historical_events(region, peril)
    climate_data, _ = da.validate_data(climate_data, "climate")
    socioeconomic_data, _ = da.validate_data(socioeconomic_data, "socioeconomic")
    historical_events, _ = da.validate_data(historical_events, "events")
    return {
        "climate": climate_data,
        "socioeconomic": socioeconomic_data,
        "historical_events": historical_events,
        "peril": peril,
        "provenance": da.provenance(),
    }
