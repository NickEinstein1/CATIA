# Transparency: what CATIA is doing

CATIA is open source so you can **inspect code, logs, and reports**. This page summarizes what each layer does and what it does *not* guarantee.

## End-to-end pipeline (`catia.pipeline.run_catia_analysis`)

1. **Data** (`catia.data_acquisition`): loads climate features, socioeconomic scalars, and a per-peril event table used for training targets and hazard calibration.
2. **Risk model** (`catia.risk_prediction`): trains a `RiskPredictor` (probability + severity heads) on engineered features.
3. **Actuarial simulation** (`catia.financial_impact`): calibrates peril rates from acquired events, builds indicative exposure from socioeconomic data, then runs exposure × vulnerability Monte Carlo (with optional EVT tails and bootstrap uncertainty).
4. **Mitigation** (`catia.mitigation`): derives recommendations from the simulated baseline loss.
5. **Artifacts**: writes JSON/HTML under the output directory according to the **artifacts** filter (or all by default).

Every `catia_report.json` includes a **`metadata.transparency`** block (manifest) when produced from current releases: data-source wording, perils, scenario id, iteration count, severity family, and explicit limitations.

## Live vs mock data

- **Live (default)**: Open-Meteo archive climate (NOAA CDO when `NOAA_API_TOKEN` is set), World Bank socioeconomic, USGS FDSN for earthquakes; weather peril history is derived from live climate extremes. Failures raise `DataUnavailableError` unless `CATIA_ALLOW_MOCK_FALLBACK=1`.
- **Mock (opt-in)**: `use_mock_data=True`, `catia --mock-data`, or `CATIA_USE_MOCK_DATA=1` generates in-process tables for offline demos and CI.

Reports include `metadata.data_provenance` and `metadata.peril_calibration`.

## Regions

Named regions (e.g. `US_Gulf_Coast`) are **coarse labels** used for configuration and visualization centroids. They are **not** a replacement for geo-coded exposure unless you add your own data and hooks.

## How to see what ran

| Channel | What you get |
| -------- | ------------ |
| **CLI** | `catia … --explain` or `catia-agent run --explain` — prints a step list before work starts |
| **Logs** | `CATIA_LOG_LEVEL=DEBUG` and `logs/catia.log` (when file logging is configured) |
| **Report** | `outputs/catia_report.json` → `metadata` and `metadata.transparency` |
| **Audit** | `audit` snapshot in the same report lists config copies for the run |

## Governance

CATIA can produce compliance-style HTML and assumption registers for **documentation-oriented** workflows. **You** remain responsible for model validation, data licensing, and regulatory suitability in your jurisdiction.
