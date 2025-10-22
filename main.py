"""
CATIA Main Entry Point
Orchestrates the complete catastrophe AI system workflow.
"""

import logging
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import LOGGING_CONFIG, OUTPUT_CONFIG, SIMULATION_CONFIG
from data_acquisition import fetch_all_data
from risk_prediction import train_risk_model
from financial_impact import run_financial_impact_analysis
from mitigation import generate_mitigation_recommendations
from visualization import create_dashboard

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

def run_catia_analysis(region: str = "US_Gulf_Coast", use_mock_data: bool = True) -> Dict:
    """
    Run complete CATIA analysis workflow.
    
    Args:
        region: Geographic region for analysis
        use_mock_data: Use mock data if True
    
    Returns:
        Dictionary with all analysis results
    """
    logger.info("=" * 80)
    logger.info("CATIA: Catastrophe AI System for Climate Risk Modeling")
    logger.info("=" * 80)
    logger.info(f"Analysis Region: {region}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # ========================================================================
    # STEP 1: DATA ACQUISITION
    # ========================================================================
    logger.info("\n[STEP 1] DATA ACQUISITION")
    logger.info("-" * 80)
    
    try:
        data = fetch_all_data(region, use_mock=use_mock_data)
        logger.info(f"✓ Climate data: {len(data['climate'])} records")
        logger.info(f"✓ Socioeconomic data: {len(data['socioeconomic'])} records")
        logger.info(f"✓ Historical events: {len(data['historical_events'])} records")
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
    # STEP 3: FINANCIAL IMPACT SIMULATION
    # ========================================================================
    logger.info("\n[STEP 3] FINANCIAL IMPACT SIMULATION")
    logger.info("-" * 80)
    
    try:
        # Calculate event frequency from historical data
        years_of_data = data['historical_events']['year'].max() - data['historical_events']['year'].min()
        event_frequency = len(data['historical_events']) / max(years_of_data, 1)
        
        # Severity parameters (lognormal)
        severity_params = {
            'mu': 15,      # Mean of log-normal
            'sigma': 2     # Std of log-normal
        }
        
        financial_results = run_financial_impact_analysis(event_frequency, severity_params)
        
        logger.info(f"✓ Monte Carlo simulations: {SIMULATION_CONFIG['monte_carlo_iterations']} iterations")
        logger.info(f"✓ Mean annual loss: ${financial_results['metrics']['descriptive_stats']['mean']:,.0f}")
        logger.info(f"✓ VaR (95%): ${financial_results['metrics']['risk_metrics']['var']:,.0f}")
        logger.info(f"✓ TVaR (95%): ${financial_results['metrics']['risk_metrics']['tvar']:,.0f}")
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
            'use_mock_data': use_mock_data
        },
        'data_summary': {
            'climate_records': len(data['climate']),
            'socioeconomic_records': len(data['socioeconomic']),
            'historical_events': len(data['historical_events'])
        },
        'risk_metrics': financial_results['metrics'],
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
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Region: {region}")
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
        # Run analysis
        results = run_catia_analysis(region="US_Gulf_Coast", use_mock_data=True)
        
        # Print key metrics
        print("\n" + "=" * 80)
        print("KEY METRICS SUMMARY")
        print("=" * 80)
        print(f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}")
        print(f"Median Annual Loss: ${results['risk_metrics']['descriptive_stats']['median']:,.0f}")
        print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
        print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")
        print(f"\n100-Year Loss: ${results['risk_metrics']['return_periods']['100_year']:,.0f}")
        print(f"500-Year Loss: ${results['risk_metrics']['return_periods']['500_year']:,.0f}")
        print(f"\nMitigation Potential: {results['mitigation_summary']['total_risk_reduction']:.2%}")
        print(f"Mitigated Loss: ${results['mitigation_summary']['mitigated_loss']:,.0f}")
        print("=" * 80)
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

