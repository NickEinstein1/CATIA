"""
Unit tests for financial impact simulation module.
"""

import pytest
import numpy as np

from catia.config import SIMULATION_CONFIG
from catia.financial_impact import FinancialImpactSimulator, run_financial_impact_analysis


@pytest.fixture(autouse=True)
def force_single_process():
    """Use n_jobs=1 to avoid joblib/loky process pool (fails in sandbox/restricted envs)."""
    orig = SIMULATION_CONFIG.get("n_jobs", 1)
    SIMULATION_CONFIG["n_jobs"] = 1
    yield
    SIMULATION_CONFIG["n_jobs"] = orig


class TestFinancialImpactSimulator:
    """Test cases for FinancialImpactSimulator class."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        severity_params = {'mu': 15, 'sigma': 2}
        return FinancialImpactSimulator(event_frequency=0.5, severity_params=severity_params)
    
    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.event_frequency == 0.5
        assert simulator.severity_params['mu'] == 15
        assert simulator.severity_params['sigma'] == 2
    
    def test_simulate_annual_losses(self, simulator):
        """Test annual loss simulation."""
        losses = simulator.simulate_annual_losses(num_years=100)
        
        assert len(losses) == 100
        assert np.all(losses >= 0)
        assert np.mean(losses) > 0
    
    def test_monte_carlo_simulation(self, simulator):
        """Test Monte Carlo simulation."""
        results = simulator.monte_carlo_simulation()
        
        assert 'all_losses' in results
        assert 'mean_loss' in results
        assert 'median_loss' in results
        assert 'std_loss' in results
        assert len(results['all_losses']) > 0
        assert results['mean_loss'] > 0
    
    def test_var_tvar_calculation(self, simulator):
        """Test VaR and TVaR calculation."""
        losses = simulator.simulate_annual_losses(num_years=1000)
        metrics = simulator.calculate_var_tvar(losses)
        
        assert 'var' in metrics
        assert 'tvar' in metrics
        assert metrics['var'] > 0
        assert metrics['tvar'] >= metrics['var']
        assert metrics['var_confidence'] == 0.95
    
    def test_loss_exceedance_curve(self, simulator):
        """Test loss exceedance curve calculation."""
        losses = simulator.simulate_annual_losses(num_years=1000)
        loss_levels, exceedance_probs = simulator.calculate_loss_exceedance_curve(losses)
        
        assert len(loss_levels) == len(exceedance_probs)
        assert np.all(loss_levels >= 0)
        assert np.all(exceedance_probs >= 0)
        assert np.all(exceedance_probs <= 1)
        # Exceedance probabilities should be decreasing
        assert np.all(np.diff(exceedance_probs) <= 0)
    
    def test_return_periods(self, simulator):
        """Test return period calculation."""
        losses = simulator.simulate_annual_losses(num_years=1000)
        return_periods = simulator.calculate_return_periods(losses)
        
        assert '10_year' in return_periods
        assert '100_year' in return_periods
        assert '1000_year' in return_periods
        
        # Higher return periods should have higher losses
        assert return_periods['10_year'] <= return_periods['100_year']
        assert return_periods['100_year'] <= return_periods['1000_year']

    def test_weibull_severity(self):
        """Test Weibull severity distribution (shape c, scale)."""
        old_dist = SIMULATION_CONFIG.get("severity_distribution")
        SIMULATION_CONFIG["severity_distribution"] = "Weibull"
        try:
            sim = FinancialImpactSimulator(0.5, {"c": 2.0, "scale": 1e6})
            losses = sim.simulate_annual_losses(num_years=200)
            assert len(losses) == 200
            assert np.all(losses >= 0)
            assert np.mean(losses) >= 0
        finally:
            SIMULATION_CONFIG["severity_distribution"] = old_dist or "Lognormal"

    def test_gamma_severity(self):
        """Test Gamma severity distribution (shape a, scale)."""
        old_dist = SIMULATION_CONFIG.get("severity_distribution")
        SIMULATION_CONFIG["severity_distribution"] = "Gamma"
        try:
            sim = FinancialImpactSimulator(0.5, {"a": 2.0, "scale": 5e5})
            losses = sim.simulate_annual_losses(num_years=200)
            assert len(losses) == 200
            assert np.all(losses >= 0)
            assert np.mean(losses) >= 0
        finally:
            SIMULATION_CONFIG["severity_distribution"] = old_dist or "Lognormal"

    def test_spliced_severity(self):
        """Test Spliced severity (lognormal body + Pareto tail)."""
        old_dist = SIMULATION_CONFIG.get("severity_distribution")
        SIMULATION_CONFIG["severity_distribution"] = "Spliced"
        try:
            # threshold optional; if omitted uses config percentile of body
            sim = FinancialImpactSimulator(
                0.5,
                {
                    "body_mu": 15,
                    "body_sigma": 2,
                    "tail_shape": 2.0,
                    "threshold": 5e6,
                },
            )
            losses = sim.simulate_annual_losses(num_years=200)
            assert len(losses) == 200
            assert np.all(losses >= 0)
            assert np.mean(losses) >= 0
        finally:
            SIMULATION_CONFIG["severity_distribution"] = old_dist or "Lognormal"
    
    def test_aggregate_metrics(self, simulator):
        """Test aggregate metrics calculation."""
        losses = simulator.simulate_annual_losses(num_years=1000)
        metrics = simulator.calculate_aggregate_metrics(losses)
        
        assert 'descriptive_stats' in metrics
        assert 'risk_metrics' in metrics
        assert 'return_periods' in metrics
        assert 'percentiles' in metrics
        
        # Check descriptive stats
        assert metrics['descriptive_stats']['mean'] > 0
        assert metrics['descriptive_stats']['std'] >= 0
        
        # Check percentiles are ordered
        assert metrics['percentiles']['50th'] <= metrics['percentiles']['75th']
        assert metrics['percentiles']['75th'] <= metrics['percentiles']['90th']
        assert metrics['percentiles']['90th'] <= metrics['percentiles']['95th']
        assert metrics['percentiles']['95th'] <= metrics['percentiles']['99th']
    
    def test_run_financial_impact_analysis(self):
        """Test complete financial impact analysis."""
        event_frequency = 0.5
        severity_params = {'mu': 15, 'sigma': 2}
        
        results = run_financial_impact_analysis(event_frequency, severity_params)
        
        assert 'simulation_results' in results
        assert 'metrics' in results
        assert 'loss_exceedance_curve' in results
        
        # Verify metrics structure
        metrics = results['metrics']
        assert 'descriptive_stats' in metrics
        assert 'risk_metrics' in metrics
        assert 'return_periods' in metrics
    
    def test_loss_distribution_properties(self, simulator):
        """Test loss distribution properties."""
        losses = simulator.simulate_annual_losses(num_years=10000)
        
        # Losses should be non-negative
        assert np.all(losses >= 0)
        
        # Mean should be positive
        assert np.mean(losses) > 0
        
        # Distribution should be right-skewed (typical for catastrophe losses)
        mean = np.mean(losses)
        median = np.median(losses)
        assert mean >= median  # Right-skewed distribution

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

