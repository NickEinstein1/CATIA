# Transparency: what CATIA is doing

CATIA is open source so you can **inspect code, logs, and reports**. This page summarizes what each layer does and what it does *not* guarantee.

## End-to-end pipeline (`catia.pipeline.run_catia_analysis`)

1. **Data** (`catia.data_acquisition`): loads climate-like features, socioeconomic scalars, and a per-peril event table used for training targets.
2. **Risk model** (`catia.risk_prediction`): trains a `RiskPredictor` (probability + severity heads) on engineered features.
3. **Actuarial simulation** (`catia.financial_impact`): runs multi-peril Monte Carlo using config frequency/severity assumptions; optional EVT tail analysis and bootstrap uncertainty.
4. **Mitigation** (`catia.mitigation`): derives recommendations from the simulated baseline loss.
5. **Artifacts**: writes JSON/HTML under the output directory according to the **artifacts** filter (or all by default).

Every `catia_report.json` includes a **`metadata.transparency`** block (manifest) when produced from current releases: data-source wording, perils, scenario id, iteration count, severity family, and explicit limitations.

## Mock vs “real” data

- **`use_mock_data=True` (default)**: tables are **generated in code** for a frictionless demo. This is not a secret: it keeps install-to-first-run fast and tests stable.
- **`use_mock_data=False`**: the stack **attempts** NOAA and World Bank fetches where wired; missing tokens or API errors **fall back to mock**. Historical catastrophe catalogs (e.g. best-track archives) are **not** fully integrated for the main training path yet—check `fetch_historical_events` in `catia/data_acquisition.py` for the current behavior.

## Regions

Named regions (e.g. `US_Gulf_Coast`) are **coarse labels** used for configuration and visualization centroids. They are **not** a replacement for geo-coded exposure unless you add your own data and hooks.

## How to see what ran

| Channel | What you get |
| --------|--------------|
| **CLI** | `catia … --explain` or `catia-agent run --explain` — prints a step list before work starts |
| **Logs** | `CATIA_LOG_LEVEL=DEBUG` and `logs/catia.log` (when file logging is configured) |
| **Report** | `outputs/catia_report.json` → `metadata` and `metadata.transparency` |
| **Audit** | `audit` snapshot in the same report lists config copies for the run |

## Governance

CATIA can produce compliance-style HTML and assumption registers for **documentation-oriented** workflows. **You** remain responsible for model validation, data licensing, and regulatory suitability in your jurisdiction.
