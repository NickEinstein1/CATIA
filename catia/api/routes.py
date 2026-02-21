"""
API Routes for CATIA REST API.
"""

import logging
import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from catia.config import PERIL_CONFIG, DEFAULT_PERILS, SIMULATION_CONFIG
from catia.config import OUTPUT_CONFIG
from catia.api.schemas import (
    PerilType, PerilInfo, PerilListResponse,
    AnalysisRequest, AnalysisResponse,
    SimulationRequest, SimulationResponse,
    MitigationRequest, MitigationResponse, MitigationStrategy,
    RiskMetrics, ReturnPeriods, PerilContribution,
    HealthResponse,
    ReadinessResponse,
    ReadinessCheck,
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse,
    StressScenarioRequest,
    StressScenarioResponse,
    StressedMetrics,
)

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter()
perils_router = APIRouter(prefix="/perils", tags=["Perils"])
analysis_router = APIRouter(prefix="/analysis", tags=["Analysis"])
simulation_router = APIRouter(prefix="/simulation", tags=["Simulation"])
mitigation_router = APIRouter(prefix="/mitigation", tags=["Mitigation"])


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness: is the API process up."""
    from catia import __version__
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat()
    )


@router.get("/metrics", tags=["Health"])
async def metrics_endpoint():
    """Prometheus metrics endpoint (if CATIA_METRICS=1)."""
    try:
        from catia.metrics import get_registry
        reg = get_registry()
        if reg.enabled:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(reg.to_prometheus(), media_type="text/plain")
        return {"message": "Metrics disabled. Set CATIA_METRICS=1 to enable."}
    except ImportError:
        return {"message": "Metrics module not available"}


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness: can the API serve traffic (output dir writable, config loadable).
    Use for Kubernetes readiness probes and load balancers.
    """
    from catia import __version__
    checks = []
    all_ok = True

    # Output directory writable
    out_dir = OUTPUT_CONFIG.get("output_dir", "outputs")
    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, ".ready_check")
        with open(test_file, "w") as f:
            f.write("")
        os.remove(test_file)
        checks.append(ReadinessCheck(name="output_dir", status="ok", message=f"Writable: {out_dir}"))
    except Exception as e:
        checks.append(ReadinessCheck(name="output_dir", status="error", message=str(e)))
        all_ok = False

    # Config loaded
    try:
        assert PERIL_CONFIG and SIMULATION_CONFIG
        checks.append(ReadinessCheck(name="config", status="ok", message="Config loaded"))
    except Exception as e:
        checks.append(ReadinessCheck(name="config", status="error", message=str(e)))
        all_ok = False

    return ReadinessResponse(
        ready=all_ok,
        version=__version__,
        timestamp=datetime.now().isoformat(),
        checks=checks,
    )


# ============================================================================
# PERILS ENDPOINTS
# ============================================================================

@perils_router.get("/", response_model=PerilListResponse)
async def list_perils():
    """List all available peril types with their configurations."""
    perils = []
    for peril_id, config in PERIL_CONFIG.items():
        perils.append(PerilInfo(
            id=peril_id,
            name=config["name"],
            frequency_base=config["frequency_base"],
            severity_mu=config["severity_params"].get("mu", 0.0),
            severity_sigma=config["severity_params"].get("sigma", 0.0),
            climate_drivers=config["climate_drivers"],
            seasonality=config["seasonality"],
            magnitude_scale=config["magnitude_scale"],
            regions=config["regions"]
        ))
    return PerilListResponse(perils=perils, count=len(perils))


@perils_router.get("/{peril_id}", response_model=PerilInfo)
async def get_peril(peril_id: PerilType):
    """Get configuration for a specific peril type."""
    config = PERIL_CONFIG.get(peril_id.value)
    if not config:
        raise HTTPException(status_code=404, detail=f"Peril '{peril_id}' not found")
    
    return PerilInfo(
        id=peril_id.value,
        name=config["name"],
        frequency_base=config["frequency_base"],
        severity_mu=config["severity_params"].get("mu", 0.0),
        severity_sigma=config["severity_params"].get("sigma", 0.0),
        climate_drivers=config["climate_drivers"],
        seasonality=config["seasonality"],
        magnitude_scale=config["magnitude_scale"],
        regions=config["regions"]
    )


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@simulation_router.post("/run", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run Monte Carlo financial impact simulation. Optional exposure uses loss = exposure × vulnerability."""
    from catia.financial_impact import MultiPerilSimulator, run_multi_peril_analysis

    try:
        perils = [p.value for p in request.perils]
        if request.exposure and len(request.exposure) > 0:
            from catia.exposure import ExposureStore
            from catia.vulnerability import VulnerabilitySet
            store = ExposureStore()
            for rec in request.exposure:
                store.add_record(
                    region=rec.region,
                    tiv=rec.tiv,
                    line_of_business=rec.line_of_business,
                    construction_type=rec.construction_type,
                    occupancy=rec.occupancy,
                    peril=rec.peril,
                )
            vuln = VulnerabilitySet()
            out = run_multi_peril_analysis(
                perils=perils,
                exposure_store=store,
                vulnerability_set=vuln,
                include_evt=False,
                include_uncertainty=False,
                num_iterations=request.num_iterations,
            )
            results = out["results"]
            contributions_df = out["contributions"]
            import pandas as pd
            contributions_df = pd.DataFrame(contributions_df)
        else:
            simulator = MultiPerilSimulator(perils=perils)
            results = simulator.simulate_all_perils(num_iterations=request.num_iterations)
            contributions_df = simulator.get_peril_contribution(results)

        agg = results['aggregate']['metrics']

        n_iter = request.num_iterations or SIMULATION_CONFIG["monte_carlo_iterations"]
        return SimulationResponse(
            perils_analyzed=perils,
            num_iterations=n_iter,
            aggregate_metrics=RiskMetrics(
                mean=agg['descriptive_stats']['mean'],
                median=agg['descriptive_stats']['median'],
                std=agg['descriptive_stats']['std'],
                var_95=agg['risk_metrics']['var'],
                tvar_95=agg['risk_metrics']['tvar']
            ),
            return_periods=ReturnPeriods(
                year_10=agg['return_periods']['10_year'],
                year_25=agg['return_periods']['25_year'],
                year_50=agg['return_periods']['50_year'],
                year_100=agg['return_periods']['100_year'],
                year_250=agg['return_periods']['250_year'],
                year_500=agg['return_periods']['500_year'],
                year_1000=agg['return_periods']['1000_year']
            ),
            peril_contributions=[
                PerilContribution(**row) for row in contributions_df.to_dict('records')
            ]
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MITIGATION ENDPOINTS
# ============================================================================

@mitigation_router.post("/optimize", response_model=MitigationResponse)
async def optimize_mitigation(request: MitigationRequest):
    """Get optimized mitigation recommendations."""
    from catia.mitigation import generate_mitigation_recommendations

    try:
        # Pass budget if provided
        if request.budget:
            results = generate_mitigation_recommendations(
                baseline_loss=request.baseline_loss,
                budget=request.budget
            )
        else:
            results = generate_mitigation_recommendations(
                baseline_loss=request.baseline_loss
            )

        # strategies is a list of dicts from DataFrame.to_dict('records')
        strategies = []
        for record in results['strategies']:
            strategies.append(MitigationStrategy(
                name=record['Strategy'],
                cost=record['Cost'],
                risk_reduction=record['Risk_Reduction'],
                effectiveness=record['Effectiveness'],
                benefit_cost_ratio=record.get('Benefit_Cost_Ratio', 0),
                npv=record.get('NPV', 0)
            ))

        return MitigationResponse(
            baseline_loss=results['summary']['baseline_loss'],
            mitigated_loss=results['summary']['mitigated_loss'],
            total_risk_reduction=results['summary']['total_risk_reduction'],
            total_cost=results['summary']['total_cost'],
            strategies=strategies,
            priority_order=results['priority_order']
        )
    except Exception as e:
        logger.error(f"Mitigation optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FULL ANALYSIS ENDPOINTS
# ============================================================================

@analysis_router.post("/run", response_model=AnalysisResponse)
async def run_full_analysis(request: AnalysisRequest):
    """Run complete CATIA analysis (data + simulation + mitigation)."""
    from catia.data_acquisition import fetch_all_data
    from catia.financial_impact import run_multi_peril_analysis
    from catia.mitigation import generate_mitigation_recommendations

    try:
        perils = [p.value for p in request.perils]

        # Fetch data
        data = fetch_all_data(
            region=request.region,
            use_mock=request.use_mock_data,
            perils=perils
        )

        # Run multi-peril simulation
        sim_results = run_multi_peril_analysis(perils)
        agg = sim_results['aggregate_metrics']

        # Run mitigation optimization
        mitigation = generate_mitigation_recommendations(
            baseline_loss=agg['descriptive_stats']['mean']
        )

        return AnalysisResponse(
            region=request.region,
            perils_analyzed=perils,
            timestamp=datetime.now().isoformat(),
            risk_metrics=RiskMetrics(
                mean=agg['descriptive_stats']['mean'],
                median=agg['descriptive_stats']['median'],
                std=agg['descriptive_stats']['std'],
                var_95=agg['risk_metrics']['var'],
                tvar_95=agg['risk_metrics']['tvar']
            ),
            return_periods=ReturnPeriods(
                year_10=agg['return_periods']['10_year'],
                year_25=agg['return_periods']['25_year'],
                year_50=agg['return_periods']['50_year'],
                year_100=agg['return_periods']['100_year'],
                year_250=agg['return_periods']['250_year'],
                year_500=agg['return_periods']['500_year'],
                year_1000=agg['return_periods']['1000_year']
            ),
            peril_contributions=[
                PerilContribution(**c) for c in sim_results['contributions']
            ],
            mitigation_summary=mitigation['summary'],
            data_summary={
                'climate_records': len(data['climate']),
                'socioeconomic_records': len(data['socioeconomic']),
                'historical_events': len(data['historical_events'])
            }
        )
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STRESS SCENARIOS (Solvency-II-style)
# ============================================================================

@analysis_router.post("/stress", response_model=StressScenarioResponse)
async def run_stress_scenarios(request: StressScenarioRequest):
    """
    Apply predefined stress scenarios to baseline risk metrics.
    Either provide baseline_metrics or let the API run a quick simulation to get baseline.
    """
    from catia.scenario_analysis import apply_stress_scenarios, PREDEFINED_SCENARIOS

    baseline = request.baseline_metrics
    if baseline is None:
        # Run quick multi-peril simulation for baseline
        from catia.financial_impact import run_multi_peril_analysis
        perils = [p.value for p in (request.perils or [PerilType.HURRICANE, PerilType.FLOOD])]
        orig_iter = SIMULATION_CONFIG.get("monte_carlo_iterations")
        try:
            SIMULATION_CONFIG["monte_carlo_iterations"] = min(2000, orig_iter or 10000)
            sim = run_multi_peril_analysis(perils, include_uncertainty=False, include_correlation=False)
        finally:
            SIMULATION_CONFIG["monte_carlo_iterations"] = orig_iter
        agg = sim["aggregate_metrics"]
        baseline = {
            "mean": agg["descriptive_stats"]["mean"],
            "var_95": agg["risk_metrics"]["var"],
            "tvar_95": agg["risk_metrics"]["tvar"],
            "return_periods": dict(agg["return_periods"]),
        }

    if "return_periods" not in baseline:
        baseline["return_periods"] = {}
    out = apply_stress_scenarios(baseline, request.scenario_ids)
    return StressScenarioResponse(
        baseline=out["baseline"],
        scenarios={k: StressedMetrics(**v) for k, v in out["scenarios"].items()},
    )


# ============================================================================
# ASYNC JOB ENDPOINTS (Phase C)
# ============================================================================

def _run_analysis_job(job_id: str, request: AnalysisRequest) -> None:
    """Background runner: run full analysis and store result."""
    from catia.api.jobs import set_job_running, set_job_result, set_job_error
    try:
        set_job_running(job_id)
        from main import run_catia_analysis
        raw = run_catia_analysis(
            region=request.region,
            use_mock_data=request.use_mock_data,
            perils=[p.value for p in request.perils],
        )
        # Build AnalysisResponse from main.py result
        rm = raw["risk_metrics"]
        agg = rm["descriptive_stats"]
        risk = rm["risk_metrics"]
        rp = rm["return_periods"]
        contributions = raw.get("multi_peril_contributions", [])
        result = AnalysisResponse(
            region=raw["metadata"]["region"],
            perils_analyzed=raw["metadata"]["perils_analyzed"],
            timestamp=raw["metadata"]["timestamp"],
            risk_metrics=RiskMetrics(
                mean=agg["mean"],
                median=agg["median"],
                std=agg["std"],
                var_95=risk["var"],
                tvar_95=risk["tvar"],
            ),
            return_periods=ReturnPeriods(
                year_10=rp["10_year"],
                year_25=rp["25_year"],
                year_50=rp["50_year"],
                year_100=rp["100_year"],
                year_250=rp["250_year"],
                year_500=rp["500_year"],
                year_1000=rp["1000_year"],
            ),
            peril_contributions=[PerilContribution(**c) for c in contributions],
            mitigation_summary=raw["mitigation_summary"],
            data_summary={
                "climate_records": raw["data_summary"]["climate_records"],
                "socioeconomic_records": raw["data_summary"]["socioeconomic_records"],
                "historical_events": raw["data_summary"]["historical_events"],
            },
        )
        set_job_result(job_id, result.model_dump())
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        set_job_error(job_id, str(e))


@analysis_router.post("/jobs", response_model=JobSubmitResponse)
async def submit_analysis_job(request: AnalysisRequest):
    """Submit a long-running analysis job. Poll GET /analysis/jobs/{job_id} for status."""
    import threading
    from catia.api.jobs import create_job, get_job

    job_id = create_job()
    job = get_job(job_id)
    t = threading.Thread(target=_run_analysis_job, args=(job_id, request))
    t.daemon = True
    t.start()
    return JobSubmitResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
    )


@analysis_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status (pending | running | completed | failed)."""
    from catia.api.jobs import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )


@analysis_router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result when completed. Returns 202 if still pending/running, 200 with result if done."""
    from catia.api.jobs import get_job, get_job_result as get_result
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "failed":
        return JobResultResponse(job_id=job_id, status="failed", error=job.get("error"))
    if job["status"] != "completed":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": job["status"], "message": "Job not yet completed"},
        )
    result = get_result(job_id)
    return JobResultResponse(job_id=job_id, status="completed", result=AnalysisResponse(**result))

