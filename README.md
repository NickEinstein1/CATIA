# CATIA

**Catastrophe AI for Climate Risk Modeling** — a production-grade platform that unifies data, ML, actuarial simulation, and mitigation optimization for multi-peril climate risk.

---

## Why CATIA

- **End-to-end** — Ingest climate data, train risk models, run Monte Carlo simulations, and optimize mitigation in one pipeline.
- **Actuarially rigorous** — Frequency–severity models, VaR/TVaR, return periods, EVT tail modeling, and copula-based multi-peril correlation.
- **Explainable & auditable** — Optional SHAP feature importance, compliance reports (CAS/SOA/NAIC), run IDs, and config snapshots.
- **Production-ready** — REST API with health checks, rate limiting, async jobs, structured errors, and observability.

---

## Installation

```bash
cd CATIA
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]" -r requirements.txt
```

---

## Quick Start

**Full pipeline (data → model → simulation → mitigation → report):**

```python
from catia.pipeline import run_catia_analysis

results = run_catia_analysis(
    region="US_Gulf_Coast",
    use_mock_data=True,
    perils=["hurricane", "flood"]
)
# Outputs: report JSON, dashboards, compliance report, optional feature importance
```

**REST API:**

```bash
uvicorn catia.api.app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

**CLI:** `catia --api --port 8000`

**System dashboard (Dash):** futuristic command-center UI with an **orthographic globe**, an **OpenStreetMap** 2D map (`dash-leaflet`) with the same markers, charts, and assumptions:

```bash
catia --dashboard
# or: catia-dashboard
# http://127.0.0.1:8050 — use --dashboard-port to change port
```

**Terminal agent (Rich + Click REPL):** optional `pip install -e ".[agent]"` (or use dev extra, which includes Click & Rich). On startup you get **colorized example prompts** (Windows Terminal or any ANSI terminal); set `RICH_FORCE_COLOR=1` if colors do not show.

```bash
catia-agent
# or: catia-agent repl

# Same entry points as `catia`:
catia-agent run -r US_Gulf_Coast -p hurricane -p flood
catia-agent run -c examples/runs/baseline.yaml
catia-agent api --port 8000
catia-agent dashboard --port 8050
```

Try `/help`, `/run --perils hurricane flood`, or plain English such as *simulate hurricane gulf coast*. The REPL maps **RiskAnalysis** (`data_acquisition` + `risk_prediction`) and **ActuarialScience** (`financial_impact`) through `catia.agent_bridge`. For a plain-language disclosure of data sources, pipeline steps, and limits, see **[Transparency](notebooks/docs/transparency.md)** (MkDocs: *Transparency*). Use **`catia --explain`** / **`catia-agent run --explain`** (or set **`CATIA_EXPLAIN=1`**) to print the same manifest to logs before a run; every **`catia_report.json`** includes **`metadata.transparency`**.

---
## Capabilities

| Area | Features |
| ---- | -------- |
| **Data** | NOAA/ECMWF/World Bank connectors; cache; mock data for development |
| **Risk model** | Probability & severity models (RF, GB, optional MLP); ensemble (`CATIA_USE_ENSEMBLE=1`); model registry |
| **Simulation** | Multi-peril Monte Carlo; Lognormal, Pareto, Weibull, Gamma, spliced severity; parallel runs; VaR/TVaR, return periods |
| **Tail & uncertainty** | EVT/GPD; bootstrap confidence intervals; correlation (Gaussian/t/Clayton/Gumbel copulas) |
| **Explainability** | SHAP feature importance (`CATIA_USE_SHAP=1`); written to `outputs/feature_importance.json` |
| **Mitigation** | Budget-constrained optimization; cost–benefit analysis; priority strategies |
| **API** | Health/ready; rate limiting; async jobs; request IDs; structured errors |

---

## API Overview

| Endpoint | Description |
| -------- | ----------- |
| `GET /api/v1/health` | Liveness |
| `GET /api/v1/ready` | Readiness |
| `GET /api/v1/perils/` | List perils and config |
| `POST /api/v1/simulation/run` | Multi-peril Monte Carlo |
| `POST /api/v1/analysis/run` | Full analysis pipeline |
| `POST /api/v1/analysis/jobs` | Submit async job; poll `GET .../jobs/{id}` and `.../jobs/{id}/result` |
| `POST /api/v1/analysis/stress` | Solvency-II-style stress scenarios (baseline or quick sim + stressed metrics) |
| `POST /api/v1/mitigation/optimize` | Mitigation recommendations |

---

## Documentation

- **`notebooks/docs/`** — Markdown guides ([updates](notebooks/docs/updates.md) — when you pull or upgrade, [transparency](notebooks/docs/transparency.md), [regions](notebooks/docs/regions.md), [perils](notebooks/docs/perils.md)). Build with [MkDocs](https://www.mkdocs.org/) (`mkdocs.yml` at repo root):

  ```bash
  pip install -e ".[docs]"
  mkdocs serve
  ```

- **[Tutorial](notebooks/tutorial.ipynb)** — Step-by-step notebook

---

## Common issues

### `catia-agent` or `catia` is not recognized (Windows PowerShell)

PowerShell only runs commands that are on **`PATH`**. Console scripts are installed into your virtual environment’s **`Scripts`** folder (e.g. `C:\Users\you\Projects\CATIA\.venv\Scripts\`). If that venv is not active—or you installed without editable mode—the name won’t resolve.

**Fix:**

1. **Activate the venv** from the repo root (prompt should show `(venv)` or `(.venv)`):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Reinstall the package in editable mode** so entry points are generated:

   ```powershell
   pip install -e ".[dev]"
   ```

   The **`dev`** extra includes Click and Rich (required for `catia-agent`). You can use `pip install -e ".[agent]"` instead if you only want agent dependencies.

3. **Confirm scripts exist:**

   ```powershell
   Get-Command catia-agent, catia | Format-Table Name, Source
   ```

   If `Source` is under `.venv\Scripts\`, you’re good.

**Always-available fallback (no reliance on `PATH` to `Scripts`):** run the module with the same interpreter:

```powershell
python -m catia.agent_repl
python -m catia.agent_repl run -r US_Gulf_Coast -p hurricane
python -m catia.cli --help
```

Use `python` that belongs to your venv (after `Activate.ps1`, `python` should be the venv one).

### Interactive REPL has no colors (plain text only)

The REPL uses **Rich** for the welcome panel, **`catia›`** prompt, and `/help`. Use **Windows Terminal**, **VS Code integrated terminal**, or another **ANSI-capable** console. If output is unstyled, force Rich to emit color:

```powershell
$env:RICH_FORCE_COLOR = "1"
catia-agent
```

### Agent REPL fails on `ImportError` (Click / Rich)

Install extras: `pip install -e ".[dev]"` or `pip install -e ".[agent]"`.

### Dashboard or API won’t start

- **Dashboard:** `pip install dash` (included in the main `pyproject` dependencies for a normal install).
- **API:** `pip install uvicorn` (listed with FastAPI in project dependencies; if you used a minimal env, install explicitly).

---

## Tests

```bash
pytest tests/ -v --tb=short
```

---

## Compliance

Designed for alignment with **CAS** catastrophe modeling guidelines, **SOA** risk management frameworks, and **NAIC** model act requirements for insurance applications.
