"""
CATIA Example Analysis Script
Demonstrates complete workflow with detailed explanations.
"""

import pandas as pd
import json

from catia.data_acquisition import fetch_all_data
from catia.risk_prediction import train_risk_model
from catia.financial_impact import run_financial_impact_analysis
from catia.mitigation import generate_mitigation_recommendations
from catia.visualization import CATIAVisualizer

# ============================================================================
# EXAMPLE 1: BASIC ANALYSIS
# ============================================================================

def example_basic_analysis():
    """Run basic CATIA analysis with mock data."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC ANALYSIS")
    print("="*80)
    
    # Step 1: Fetch data
    print("\n[Step 1] Fetching data...")
    data = fetch_all_data("US_Gulf_Coast", use_mock=True)
    print(f"  ✓ Climate data: {len(data['climate'])} records")
    print(f"  ✓ Socioeconomic data: {len(data['socioeconomic'])} records")
    print(f"  ✓ Historical events: {len(data['historical_events'])} records")
    
    # Step 2: Train model
    print("\n[Step 2] Training risk prediction model...")
    predictor = train_risk_model(
        data['climate'],
        data['socioeconomic'],
        data['historical_events']
    )
    print("  ✓ Model trained and saved")
    
    # Step 3: Run financial analysis
    print("\n[Step 3] Running financial impact simulation...")
    event_frequency = 0.5  # 0.5 events per year
    severity_params = {'mu': 15, 'sigma': 2}
    
    financial_results = run_financial_impact_analysis(
        event_frequency,
        severity_params
    )
    
    metrics = financial_results['metrics']
    print(f"  ✓ Mean annual loss: ${metrics['descriptive_stats']['mean']:,.0f}")
    print(f"  ✓ VaR (95%): ${metrics['risk_metrics']['var']:,.0f}")
    print(f"  ✓ TVaR (95%): ${metrics['risk_metrics']['tvar']:,.0f}")
    
    # Step 4: Generate recommendations
    print("\n[Step 4] Generating mitigation recommendations...")
    baseline_loss = metrics['descriptive_stats']['mean']
    recommendations = generate_mitigation_recommendations(baseline_loss)
    
    print(f"  ✓ Baseline loss: ${recommendations['summary']['baseline_loss']:,.0f}")
    print(f"  ✓ Mitigated loss: ${recommendations['summary']['mitigated_loss']:,.0f}")
    print(f"  ✓ Risk reduction: {recommendations['summary']['total_risk_reduction']:.2%}")
    
    # Step 5: Create visualizations
    print("\n[Step 5] Creating visualizations...")
    visualizer = CATIAVisualizer()
    
    loss_levels, exceedance_probs = financial_results['loss_exceedance_curve'].values()
    var_95 = metrics['risk_metrics']['var']
    tvar_95 = metrics['risk_metrics']['tvar']
    
    fig1 = visualizer.plot_loss_exceedance_curve(loss_levels, exceedance_probs, var_95, tvar_95)
    visualizer.save_figure(fig1, "example_loss_exceedance_curve.html")
    print("  ✓ Loss exceedance curve saved")
    
    fig2 = visualizer.plot_risk_distribution(financial_results['simulation_results']['all_losses'])
    visualizer.save_figure(fig2, "example_risk_distribution.html")
    print("  ✓ Risk distribution saved")
    
    return financial_results, recommendations

# ============================================================================
# EXAMPLE 2: SENSITIVITY ANALYSIS
# ============================================================================

def example_sensitivity_analysis():
    """Demonstrate sensitivity analysis for different event frequencies."""
    print("\n" + "="*80)
    print("EXAMPLE 2: SENSITIVITY ANALYSIS")
    print("="*80)
    
    frequencies = [0.25, 0.5, 1.0, 2.0]
    severity_params = {'mu': 15, 'sigma': 2}
    
    results_summary = []
    
    for freq in frequencies:
        print(f"\nAnalyzing frequency: {freq} events/year...")
        
        results = run_financial_impact_analysis(freq, severity_params)
        metrics = results['metrics']
        
        summary = {
            'frequency': freq,
            'mean_loss': metrics['descriptive_stats']['mean'],
            'var_95': metrics['risk_metrics']['var'],
            'tvar_95': metrics['risk_metrics']['tvar'],
            '100_year_loss': metrics['return_periods']['100_year']
        }
        results_summary.append(summary)
        
        print(f"  Mean loss: ${summary['mean_loss']:,.0f}")
        print(f"  VaR (95%): ${summary['var_95']:,.0f}")
    
    # Create summary table
    df = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    return df

# ============================================================================
# EXAMPLE 3: MULTI-REGION ANALYSIS
# ============================================================================

def example_multi_region_analysis():
    """Analyze multiple regions and compare results."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MULTI-REGION ANALYSIS")
    print("="*80)
    
    regions = ["US_Gulf_Coast", "California_Coast", "Florida_Peninsula"]
    results_by_region = {}
    
    for region in regions:
        print(f"\nAnalyzing region: {region}...")
        
        # Fetch data
        data = fetch_all_data(region, use_mock=True)
        
        # Train model
        predictor = train_risk_model(
            data['climate'],
            data['socioeconomic'],
            data['historical_events']
        )
        
        # Run analysis
        event_frequency = 0.5
        severity_params = {'mu': 15, 'sigma': 2}
        results = run_financial_impact_analysis(event_frequency, severity_params)
        
        metrics = results['metrics']
        results_by_region[region] = {
            'mean_loss': metrics['descriptive_stats']['mean'],
            'var_95': metrics['risk_metrics']['var'],
            'tvar_95': metrics['risk_metrics']['tvar']
        }
        
        print(f"  ✓ Mean loss: ${results_by_region[region]['mean_loss']:,.0f}")
    
    # Create comparison table
    df = pd.DataFrame(results_by_region).T
    print("\n" + "="*80)
    print("MULTI-REGION COMPARISON")
    print("="*80)
    print(df.to_string())
    
    return df

# ============================================================================
# EXAMPLE 4: CUSTOM MITIGATION STRATEGY
# ============================================================================

def example_custom_mitigation():
    """Demonstrate custom mitigation strategy optimization."""
    print("\n" + "="*80)
    print("EXAMPLE 4: CUSTOM MITIGATION STRATEGY")
    print("="*80)
    
    from mitigation import MitigationOptimizer, MitigationStrategy
    
    # Create optimizer with custom budget
    budget = 2_000_000  # $2M budget
    optimizer = MitigationOptimizer(budget)
    
    # Add custom strategies
    strategies = {
        'early_warning_system': {
            'cost': 500_000,
            'risk_reduction': 0.15,
            'implementation_time': 1,
            'effectiveness': 0.80
        },
        'community_relocation': {
            'cost': 1_500_000,
            'risk_reduction': 0.45,
            'implementation_time': 3,
            'effectiveness': 0.95
        },
        'infrastructure_upgrade': {
            'cost': 800_000,
            'risk_reduction': 0.25,
            'implementation_time': 2,
            'effectiveness': 0.85
        }
    }
    
    optimizer.add_strategies_from_dict(strategies)
    
    # Optimize
    print(f"\nOptimizing with budget: ${budget:,.0f}")
    selected = optimizer.optimize_linear_programming()
    
    print(f"\nSelected strategies:")
    for strategy in selected:
        print(f"  • {strategy.name}: ${strategy.cost:,.0f} (reduction: {strategy.risk_reduction:.0%})")
    
    # Cost-benefit analysis
    baseline_loss = 50_000_000
    cba = optimizer.calculate_cost_benefit_analysis(baseline_loss)
    
    print("\n" + "="*80)
    print("COST-BENEFIT ANALYSIS")
    print("="*80)
    print(cba.to_string(index=False))
    
    return cba

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CATIA EXAMPLE ANALYSIS SUITE")
    print("="*80)
    
    try:
        # Run examples
        print("\nRunning examples...")
        
        # Example 1: Basic analysis
        financial_results, recommendations = example_basic_analysis()
        
        # Example 2: Sensitivity analysis
        sensitivity_df = example_sensitivity_analysis()
        
        # Example 3: Multi-region analysis
        multi_region_df = example_multi_region_analysis()
        
        # Example 4: Custom mitigation
        cba_df = example_custom_mitigation()
        
        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        os.makedirs("outputs", exist_ok=True)
        
        sensitivity_df.to_csv("outputs/sensitivity_analysis.csv", index=False)
        print("✓ Sensitivity analysis saved to outputs/sensitivity_analysis.csv")
        
        multi_region_df.to_csv("outputs/multi_region_analysis.csv")
        print("✓ Multi-region analysis saved to outputs/multi_region_analysis.csv")
        
        cba_df.to_csv("outputs/custom_mitigation_cba.csv", index=False)
        print("✓ Custom mitigation CBA saved to outputs/custom_mitigation_cba.csv")
        
        print("\n" + "="*80)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated files:")
        print("  • outputs/example_loss_exceedance_curve.html")
        print("  • outputs/example_risk_distribution.html")
        print("  • outputs/sensitivity_analysis.csv")
        print("  • outputs/multi_region_analysis.csv")
        print("  • outputs/custom_mitigation_cba.csv")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

