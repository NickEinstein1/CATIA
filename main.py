"""
CATIA Main Entry Point
Orchestrates the complete catastrophe AI system workflow.
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict
import numpy as np

from catia.config import LOGGING_CONFIG, OUTPUT_CONFIG, SIMULATION_CONFIG, PERIL_CONFIG, DEFAULT_PERILS
from catia.data_acquisition import fetch_all_data, fetch_single_peril_data
from catia.risk_prediction import train_risk_model
from catia.financial_impact import (
    run_financial_impact_analysis,
    run_multi_peril_analysis,
    FinancialImpactSimulator,
    MultiPerilSimulator
)
from catia.mitigation import generate_mitigation_recommendations
from catia.visualization import create_dashboard

# Phase 1 Enhancements
from catia.sensitivity_analysis import QuickSensitivityAnalysis
from catia.scenario_analysis import ScenarioAnalyzer, get_scenarios_for_peril
from catia.model_comparison import ModelComparison
from catia.risk_alerts import RiskAlertSystem
from catia.export import ReportExporter

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_catia_analysis(region: str = "US_Gulf_Coast",
                       use_mock_data: bool = True,
                       perils: list = None) -> Dict:
    """
    Run complete CATIA analysis workflow.

    Args:
        region: Geographic region for analysis
        use_mock_data: Use mock data if True
        perils: List of peril types to analyze (uses DEFAULT_PERILS if None)

    Returns:
        Dictionary with all analysis results
    """
    perils = perils or DEFAULT_PERILS

    logger.info("=" * 80)
    logger.info("CATIA: Catastrophe AI System for Climate Risk Modeling")
    logger.info("=" * 80)
    logger.info(f"Analysis Region: {region}")
    logger.info(f"Perils: {', '.join(perils)}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 1: DATA ACQUISITION (Multi-Peril)
    # ========================================================================
    logger.info("\n[STEP 1] DATA ACQUISITION (Multi-Peril)")
    logger.info("-" * 80)

    try:
        data = fetch_all_data(region, use_mock=use_mock_data, perils=perils)
        logger.info(f"✓ Climate data: {len(data['climate'])} records")
        logger.info(f"✓ Socioeconomic data: {len(data['socioeconomic'])} records")
        logger.info(f"✓ Historical events: {len(data['historical_events'])} records")
        logger.info(f"✓ Perils analyzed: {', '.join(data['perils_analyzed'])}")
        for peril, events in data.get('events_by_peril', {}).items():
            logger.info(f"    - {peril}: {len(events)} events")
    except Exception as e:
        logger.error(f"✗ Data acquisition failed: {e}")
        raise
    
    # ========================================================================
    # STEP 2: RISK PREDICTION MODEL
    # ========================================================================
    logger.info("\n[STEP 2] RISK PREDICTION MODEL")
    logger.info("-" * 80)
    
    try:
        predictor = train_risk_model(
            data['climate'],
            data['socioeconomic'],
            data['historical_events']
        )
        logger.info("✓ Risk prediction model trained and saved")
    except Exception as e:
        logger.error(f"✗ Risk prediction failed: {e}")
        raise
    
    # ========================================================================
    # STEP 3: FINANCIAL IMPACT SIMULATION (Multi-Peril)
    # ========================================================================
    logger.info("\n[STEP 3] FINANCIAL IMPACT SIMULATION (Multi-Peril)")
    logger.info("-" * 80)

    try:
        # Run multi-peril analysis
        multi_peril_results = run_multi_peril_analysis(perils)

        logger.info(f"✓ Monte Carlo simulations: {SIMULATION_CONFIG['monte_carlo_iterations']} iterations")
        logger.info(f"✓ Perils simulated: {', '.join(perils)}")

        # Log per-peril results
        for contrib in multi_peril_results['contributions']:
            logger.info(f"    - {contrib['peril_name']}: Mean=${contrib['mean_loss']:,.0f} ({contrib['contribution_pct']:.1f}%)")

        # Aggregate results
        agg_metrics = multi_peril_results['aggregate_metrics']
        logger.info(f"✓ Aggregate Mean Annual Loss: ${agg_metrics['descriptive_stats']['mean']:,.0f}")
        logger.info(f"✓ Aggregate VaR (95%): ${agg_metrics['risk_metrics']['var']:,.0f}")
        logger.info(f"✓ Aggregate TVaR (95%): ${agg_metrics['risk_metrics']['tvar']:,.0f}")

        # For backward compatibility, create financial_results dict
        # Calculate loss exceedance curve for visualization
        aggregate_losses = multi_peril_results['results']['aggregate']['losses']
        loss_levels = np.percentile(aggregate_losses, np.linspace(0, 99.9, 100))
        exceedance_probs = 1 - np.linspace(0, 0.999, 100)

        financial_results = {
            'metrics': agg_metrics,
            'simulation_results': {'all_losses': aggregate_losses},
            'loss_exceedance_curve': {
                'loss_levels': loss_levels,
                'exceedance_probabilities': exceedance_probs
            },
            'multi_peril': multi_peril_results
        }
    except Exception as e:
        logger.error(f"✗ Financial impact simulation failed: {e}")
        raise
    
    # ========================================================================
    # STEP 4: MITIGATION RECOMMENDATIONS
    # ========================================================================
    logger.info("\n[STEP 4] MITIGATION RECOMMENDATIONS")
    logger.info("-" * 80)
    
    try:
        baseline_loss = financial_results['metrics']['descriptive_stats']['mean']
        mitigation_results = generate_mitigation_recommendations(baseline_loss)
        
        logger.info(f"✓ Baseline loss: ${mitigation_results['summary']['baseline_loss']:,.0f}")
        logger.info(f"✓ Mitigated loss: ${mitigation_results['summary']['mitigated_loss']:,.0f}")
        logger.info(f"✓ Risk reduction: {mitigation_results['summary']['total_risk_reduction']:.2%}")
        logger.info(f"✓ Priority strategies: {', '.join(mitigation_results['priority_order'][:3])}")
    except Exception as e:
        logger.error(f"✗ Mitigation recommendations failed: {e}")
        raise
    
    # ========================================================================
    # STEP 5: VISUALIZATION & REPORTING
    # ========================================================================
    logger.info("\n[STEP 5] VISUALIZATION & REPORTING")
    logger.info("-" * 80)
    
    try:
        # Create visualizations
        import pandas as pd
        cba_df = pd.DataFrame(mitigation_results['strategies'])
        dashboard_dir = create_dashboard(financial_results, data['climate'], cba_df)
        
        logger.info(f"✓ Dashboard created: {dashboard_dir}")
        logger.info(f"  - loss_exceedance_curve.html")
        logger.info(f"  - risk_distribution.html")
        logger.info(f"  - return_period_curve.html")
        logger.info(f"  - mitigation_comparison.html")
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")
        raise
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    logger.info("\n[STEP 6] RESULTS COMPILATION")
    logger.info("-" * 80)
    
    results = {
        'metadata': {
            'region': region,
            'timestamp': datetime.now().isoformat(),
            'use_mock_data': use_mock_data,
            'perils_analyzed': perils
        },
        'data_summary': {
            'climate_records': len(data['climate']),
            'socioeconomic_records': len(data['socioeconomic']),
            'historical_events': len(data['historical_events']),
            'events_by_peril': {p: len(e) for p, e in data.get('events_by_peril', {}).items()}
        },
        'risk_metrics': financial_results['metrics'],
        'multi_peril_contributions': financial_results.get('multi_peril', {}).get('contributions', []),
        'mitigation_summary': mitigation_results['summary'],
        'mitigation_strategies': mitigation_results['strategies'],
        'priority_order': mitigation_results['priority_order']
    }
    
    # Save results to JSON
    os.makedirs(OUTPUT_CONFIG["output_dir"], exist_ok=True)
    output_file = os.path.join(OUTPUT_CONFIG["output_dir"], "catia_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ Report saved: {output_file}")

    # ========================================================================
    # PHASE 1 ENHANCEMENTS: QUICK WINS
    # ========================================================================
    logger.info("\n[PHASE 1] QUICK WINS ENHANCEMENTS")
    logger.info("-" * 80)

    try:
        # Create simulator for enhancements
        event_frequency = len(data['historical_events']) / max(
            data['historical_events']['year'].max() - data['historical_events']['year'].min(), 1
        )
        severity_params = {'mu': 15, 'sigma': 2}
        simulator = FinancialImpactSimulator(event_frequency, severity_params)

        # 1. Sensitivity Analysis
        logger.info("\n[ENHANCEMENT 1] Sensitivity Analysis")
        try:
            analyzer = QuickSensitivityAnalysis(simulator)
            sensitivity_results = analyzer.analyze({
                'event_frequency': [0.3, 0.4, 0.5, 0.6, 0.7],
                'severity_mu': [14, 15, 16, 17, 18]
            })
            analyzer.plot_tornado(sensitivity_results).write_html(
                os.path.join(OUTPUT_CONFIG["output_dir"], "sensitivity_tornado.html")
            )
            analyzer.plot_sensitivity_heatmap(sensitivity_results).write_html(
                os.path.join(OUTPUT_CONFIG["output_dir"], "sensitivity_heatmap.html")
            )
            logger.info(f"✓ Sensitivity analysis complete")
            logger.info(analyzer.generate_summary(sensitivity_results))
        except Exception as e:
            logger.error(f"✗ Sensitivity analysis failed: {e}")

        # 2. Scenario Analysis
        logger.info("\n[ENHANCEMENT 2] Scenario Analysis")
        try:
            scenario_analyzer = ScenarioAnalyzer(simulator)
            scenario_results = scenario_analyzer.run_scenarios()
            scenario_analyzer.plot_scenarios(scenario_results).write_html(
                os.path.join(OUTPUT_CONFIG["output_dir"], "scenarios.html")
            )
            scenario_analyzer.plot_return_periods(scenario_results).write_html(
                os.path.join(OUTPUT_CONFIG["output_dir"], "return_periods.html")
            )
            logger.info(f"✓ Scenario analysis complete")
            logger.info(scenario_analyzer.generate_summary(scenario_results))
        except Exception as e:
            logger.error(f"✗ Scenario analysis failed: {e}")

        # 3. Risk Alerts
        logger.info("\n[ENHANCEMENT 3] Risk Alerts")
        try:
            alert_system = RiskAlertSystem({
                'var_max': 100,
                'mean_loss_max': 50,
                'loss_ratio_max': 1.1,
                'tvar_max': 150
            })

            # Prepare metrics for alerts
            alert_metrics = {
                'var_95': results['risk_metrics']['risk_metrics']['var'],
                'tvar_95': results['risk_metrics']['risk_metrics']['tvar'],
                'mean_loss': results['risk_metrics']['descriptive_stats']['mean'],
                'loss_ratio': 1.0  # Placeholder
            }

            alerts = alert_system.check_alerts(alert_metrics)
            logger.info(alert_system.format_alerts())
        except Exception as e:
            logger.error(f"✗ Risk alerts failed: {e}")

        # 4. Export Results
        logger.info("\n[ENHANCEMENT 4] Export Results")
        try:
            exporter = ReportExporter(results['risk_metrics'])
            export_paths = exporter.export_all()
            logger.info(f"✓ JSON export: {export_paths['json']}")
            logger.info(f"✓ CSV export: {export_paths['csv']}")
            logger.info(f"✓ HTML export: {export_paths['html']}")
        except Exception as e:
            logger.error(f"✗ Export failed: {e}")

        logger.info("\n✓ Phase 1 Enhancements Complete")

    except Exception as e:
        logger.error(f"✗ Phase 1 enhancements failed: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Region: {region}")
    logger.info(f"Perils Analyzed: {', '.join(perils)}")
    logger.info(f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}")
    logger.info(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
    logger.info(f"Risk Reduction Potential: {results['mitigation_summary']['total_risk_reduction']:.2%}")
    logger.info(f"Output Directory: {OUTPUT_CONFIG['output_dir']}")
    logger.info("=" * 80)

    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Run multi-peril analysis
        results = run_catia_analysis(
            region="US_Gulf_Coast",
            use_mock_data=True,
            perils=["hurricane", "flood", "wildfire", "earthquake"]
        )

        # Print key metrics
        print("\n" + "=" * 80)
        print("KEY METRICS SUMMARY (Multi-Peril)")
        print("=" * 80)
        print(f"Perils Analyzed: {', '.join(results['metadata']['perils_analyzed'])}")
        print(f"\nAggregate Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}")
        print(f"Median Annual Loss: ${results['risk_metrics']['descriptive_stats']['median']:,.0f}")
        print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
        print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")

        print(f"\nLoss by Peril:")
        for contrib in results.get('multi_peril_contributions', []):
            print(f"  {contrib['peril_name']}: ${contrib['mean_loss']:,.0f} ({contrib['contribution_pct']:.1f}%)")

        print(f"\n100-Year Loss: ${results['risk_metrics']['return_periods']['100_year']:,.0f}")
        print(f"500-Year Loss: ${results['risk_metrics']['return_periods']['500_year']:,.0f}")
        print(f"\nMitigation Potential: {results['mitigation_summary']['total_risk_reduction']:.2%}")
        print(f"Mitigated Loss: ${results['mitigation_summary']['mitigated_loss']:,.0f}")
        print("=" * 80)

        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

