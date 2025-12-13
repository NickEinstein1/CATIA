"""
API Routes for CATIA REST API.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from catia.config import PERIL_CONFIG, DEFAULT_PERILS, SIMULATION_CONFIG
from catia.api.schemas import (
    PerilType, PerilInfo, PerilListResponse,
    AnalysisRequest, AnalysisResponse,
    SimulationRequest, SimulationResponse,
    MitigationRequest, MitigationResponse, MitigationStrategy,
    RiskMetrics, ReturnPeriods, PerilContribution,
    HealthResponse
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
    """Check API health status."""
    from catia import __version__
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat()
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
            severity_mu=config["severity_params"]["mu"],
            severity_sigma=config["severity_params"]["sigma"],
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
        severity_mu=config["severity_params"]["mu"],
        severity_sigma=config["severity_params"]["sigma"],
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
    """Run Monte Carlo financial impact simulation."""
    from catia.financial_impact import MultiPerilSimulator

    try:
        perils = [p.value for p in request.perils]
        simulator = MultiPerilSimulator(perils=perils)
        results = simulator.simulate_all_perils(num_iterations=request.num_iterations)
        contributions_df = simulator.get_peril_contribution(results)

        agg = results['aggregate']['metrics']

        return SimulationResponse(
            perils_analyzed=perils,
            num_iterations=request.num_iterations or SIMULATION_CONFIG["monte_carlo_iterations"],
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

