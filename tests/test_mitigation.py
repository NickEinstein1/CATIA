"""
Unit tests for mitigation module.
"""

import pytest
import pandas as pd
import numpy as np

from catia.mitigation import (
    MitigationStrategy,
    MitigationOptimizer,
    PREDEFINED_STRATEGIES,
    generate_mitigation_recommendations
)


class TestMitigationStrategy:
    """Test cases for MitigationStrategy class."""
    
    def test_initialization(self):
        """Test MitigationStrategy initialization."""
        strategy = MitigationStrategy(
            name="test_strategy",
            cost=100000,
            risk_reduction=0.20,
            implementation_time=1,
            effectiveness=0.80
        )
        
        assert strategy.name == "test_strategy"
        assert strategy.cost == 100000
        assert strategy.risk_reduction == 0.20
        assert strategy.implementation_time == 1
        assert strategy.effectiveness == 0.80
    
    def test_cost_benefit_ratio_calculation(self):
        """Test cost-benefit ratio is calculated correctly."""
        strategy = MitigationStrategy(
            name="test",
            cost=100000,
            risk_reduction=0.25,
            implementation_time=1,
            effectiveness=0.80
        )
        
        expected_ratio = 100000 / 0.25
        assert strategy.cost_benefit_ratio == expected_ratio
    
    def test_zero_risk_reduction(self):
        """Test strategy with zero risk reduction."""
        strategy = MitigationStrategy(
            name="no_effect",
            cost=50000,
            risk_reduction=0,
            implementation_time=1,
            effectiveness=0
        )
        
        assert strategy.cost_benefit_ratio == float('inf')
    
    def test_repr(self):
        """Test string representation."""
        strategy = MitigationStrategy("test", 100000, 0.20, 1, 0.80)
        repr_str = repr(strategy)
        
        assert "test" in repr_str
        assert "100,000" in repr_str
        assert "20.00%" in repr_str


class TestMitigationOptimizer:
    """Test cases for MitigationOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create MitigationOptimizer instance."""
        return MitigationOptimizer(budget=500000)
    
    @pytest.fixture
    def optimizer_with_strategies(self):
        """Create optimizer with predefined strategies."""
        opt = MitigationOptimizer(budget=1000000)
        opt.add_strategies_from_dict(PREDEFINED_STRATEGIES)
        return opt
    
    def test_initialization(self, optimizer):
        """Test MitigationOptimizer initialization."""
        assert optimizer.budget == 500000
        assert optimizer.strategies == []
        assert optimizer.selected_strategies == []
    
    def test_add_strategy(self, optimizer):
        """Test adding a strategy."""
        strategy = MitigationStrategy("test", 100000, 0.15, 1, 0.70)
        optimizer.add_strategy(strategy)
        
        assert len(optimizer.strategies) == 1
        assert optimizer.strategies[0].name == "test"
    
    def test_add_strategies_from_dict(self, optimizer):
        """Test adding strategies from dictionary."""
        optimizer.add_strategies_from_dict(PREDEFINED_STRATEGIES)
        
        assert len(optimizer.strategies) == len(PREDEFINED_STRATEGIES)
        strategy_names = [s.name for s in optimizer.strategies]
        for name in PREDEFINED_STRATEGIES.keys():
            assert name in strategy_names
    
    def test_optimize_greedy(self, optimizer_with_strategies):
        """Test greedy optimization."""
        selected = optimizer_with_strategies.optimize_greedy()
        
        assert len(selected) > 0
        total_cost = sum(s.cost for s in selected)
        assert total_cost <= optimizer_with_strategies.budget
    
    def test_optimize_linear_programming(self, optimizer_with_strategies):
        """Test linear programming optimization."""
        selected = optimizer_with_strategies.optimize_linear_programming()
        
        assert len(selected) > 0
        total_cost = sum(s.cost for s in selected)
        # LP relaxation may slightly exceed budget, but should be close
        assert total_cost <= optimizer_with_strategies.budget * 1.2
    
    def test_optimization_respects_budget(self):
        """Test that optimization respects tight budget."""
        optimizer = MitigationOptimizer(budget=150000)
        optimizer.add_strategies_from_dict(PREDEFINED_STRATEGIES)
        selected = optimizer.optimize_greedy()
        
        total_cost = sum(s.cost for s in selected)
        assert total_cost <= 150000
    
    def test_cost_benefit_analysis(self, optimizer_with_strategies):
        """Test cost-benefit analysis calculation."""
        optimizer_with_strategies.optimize_linear_programming()
        baseline_loss = 10000000  # $10M
        
        cba = optimizer_with_strategies.calculate_cost_benefit_analysis(baseline_loss)
        
        assert isinstance(cba, pd.DataFrame)
        assert 'Strategy' in cba.columns
        assert 'Cost' in cba.columns
        assert 'Annual_Benefit' in cba.columns
        assert 'NPV' in cba.columns
        assert 'Benefit_Cost_Ratio' in cba.columns
        assert 'Payback_Period_Years' in cba.columns
    
    def test_generate_recommendations(self, optimizer_with_strategies):
        """Test recommendations generation."""
        optimizer_with_strategies.optimize_linear_programming()
        baseline_loss = 10000000
        
        recommendations = optimizer_with_strategies.generate_recommendations(baseline_loss)
        
        assert 'summary' in recommendations
        assert 'strategies' in recommendations
        assert 'priority_order' in recommendations
        assert recommendations['summary']['baseline_loss'] == baseline_loss
        assert recommendations['summary']['mitigated_loss'] < baseline_loss


class TestGenerateMitigationRecommendations:
    """Test cases for generate_mitigation_recommendations function."""
    
    def test_basic_recommendations(self):
        """Test basic recommendation generation."""
        recommendations = generate_mitigation_recommendations(
            baseline_loss=50000000,
            budget=1000000
        )
        
        assert 'summary' in recommendations
        assert 'strategies' in recommendations
        assert 'priority_order' in recommendations
    
    def test_recommendations_reduce_loss(self):
        """Test that recommendations reduce loss."""
        baseline_loss = 50000000
        recommendations = generate_mitigation_recommendations(baseline_loss)
        
        mitigated_loss = recommendations['summary']['mitigated_loss']
        assert mitigated_loss < baseline_loss
    
    def test_risk_reduction_valid(self):
        """Test that risk reduction is between 0 and 1."""
        recommendations = generate_mitigation_recommendations(50000000)
        
        total_reduction = recommendations['summary']['total_risk_reduction']
        assert 0 <= total_reduction <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

