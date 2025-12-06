"""
Mitigation Recommendations Module for CATIA
Data-driven mitigation strategies with optimization under budget constraints.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import linprog

from catia.config import MITIGATION_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# ============================================================================
# MITIGATION STRATEGY CLASS
# ============================================================================

class MitigationStrategy:
    """Represents a single mitigation strategy."""
    
    def __init__(self, name: str, cost: float, risk_reduction: float, 
                 implementation_time: int, effectiveness: float):
        """
        Initialize mitigation strategy.
        
        Args:
            name: Strategy name
            cost: Implementation cost (USD)
            risk_reduction: Expected risk reduction (0-1)
            implementation_time: Time to implement (years)
            effectiveness: Effectiveness rating (0-1)
        """
        self.name = name
        self.cost = cost
        self.risk_reduction = risk_reduction
        self.implementation_time = implementation_time
        self.effectiveness = effectiveness
        self.cost_benefit_ratio = cost / risk_reduction if risk_reduction > 0 else float('inf')
    
    def __repr__(self):
        return f"MitigationStrategy({self.name}, cost=${self.cost:,.0f}, reduction={self.risk_reduction:.2%})"

# ============================================================================
# MITIGATION OPTIMIZER CLASS
# ============================================================================

class MitigationOptimizer:
    """Optimizes mitigation strategy selection under budget constraints."""
    
    def __init__(self, budget: float = MITIGATION_CONFIG["budget_constraint"]):
        """
        Initialize optimizer.
        
        Args:
            budget: Total budget for mitigation (USD)
        """
        self.budget = budget
        self.strategies = []
        self.selected_strategies = []
        logger.info(f"MitigationOptimizer initialized (budget=${budget:,.0f})")
    
    def add_strategy(self, strategy: MitigationStrategy):
        """Add a mitigation strategy."""
        self.strategies.append(strategy)
    
    def add_strategies_from_dict(self, strategies_dict: Dict):
        """
        Add multiple strategies from dictionary.
        
        Args:
            strategies_dict: Dictionary with strategy definitions
        """
        for name, params in strategies_dict.items():
            strategy = MitigationStrategy(
                name=name,
                cost=params['cost'],
                risk_reduction=params['risk_reduction'],
                implementation_time=params['implementation_time'],
                effectiveness=params['effectiveness']
            )
            self.add_strategy(strategy)
    
    def optimize_greedy(self) -> List[MitigationStrategy]:
        """
        Greedy optimization: select strategies with best cost-benefit ratio.
        
        Returns:
            List of selected strategies
        """
        logger.info("Running greedy optimization...")
        
        # Sort by cost-benefit ratio
        sorted_strategies = sorted(self.strategies, key=lambda s: s.cost_benefit_ratio)
        
        selected = []
        remaining_budget = self.budget
        total_risk_reduction = 0
        
        for strategy in sorted_strategies:
            if strategy.cost <= remaining_budget:
                selected.append(strategy)
                remaining_budget -= strategy.cost
                total_risk_reduction += strategy.risk_reduction
                logger.info(f"  Selected: {strategy.name} (cost=${strategy.cost:,.0f})")
        
        logger.info(f"Total risk reduction: {total_risk_reduction:.2%}")
        logger.info(f"Remaining budget: ${remaining_budget:,.0f}")
        
        self.selected_strategies = selected
        return selected
    
    def optimize_linear_programming(self) -> List[MitigationStrategy]:
        """
        Linear programming optimization: maximize risk reduction under budget.
        
        Returns:
            List of selected strategies
        """
        logger.info("Running linear programming optimization...")
        
        n = len(self.strategies)
        
        # Objective: maximize risk reduction (minimize negative risk reduction)
        c = [-s.risk_reduction for s in self.strategies]
        
        # Constraint: total cost <= budget
        A_ub = [[s.cost for s in self.strategies]]
        b_ub = [self.budget]
        
        # Bounds: binary (0 or 1)
        bounds = [(0, 1) for _ in range(n)]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            selected = [s for i, s in enumerate(self.strategies) if result.x[i] > 0.5]
            total_cost = sum(s.cost for s in selected)
            total_reduction = sum(s.risk_reduction for s in selected)
            
            logger.info(f"Optimization successful")
            logger.info(f"  Total cost: ${total_cost:,.0f}")
            logger.info(f"  Total risk reduction: {total_reduction:.2%}")
            
            self.selected_strategies = selected
            return selected
        else:
            logger.warning("Optimization failed, using greedy approach")
            return self.optimize_greedy()
    
    def calculate_cost_benefit_analysis(self, baseline_loss: float) -> pd.DataFrame:
        """
        Calculate cost-benefit analysis for selected strategies.
        
        Args:
            baseline_loss: Baseline annual loss without mitigation
        
        Returns:
            DataFrame with cost-benefit analysis
        """
        discount_rate = MITIGATION_CONFIG["cost_benefit_discount_rate"]
        analysis_period = 30  # years
        
        results = []
        
        for strategy in self.selected_strategies:
            # Calculate annual benefit
            annual_benefit = baseline_loss * strategy.risk_reduction
            
            # Calculate NPV
            npv = 0
            for year in range(1, analysis_period + 1):
                discount_factor = (1 + discount_rate) ** (-year)
                npv += annual_benefit * discount_factor
            
            # Subtract implementation cost
            npv -= strategy.cost
            
            # Calculate benefit-cost ratio
            bcr = npv / strategy.cost if strategy.cost > 0 else 0
            
            # Payback period
            payback = strategy.cost / annual_benefit if annual_benefit > 0 else float('inf')
            
            results.append({
                'Strategy': strategy.name,
                'Cost': strategy.cost,
                'Annual_Benefit': annual_benefit,
                'NPV': npv,
                'Benefit_Cost_Ratio': bcr,
                'Payback_Period_Years': payback,
                'Risk_Reduction': strategy.risk_reduction,
                'Effectiveness': strategy.effectiveness
            })
        
        df = pd.DataFrame(results)
        logger.info("\nCost-Benefit Analysis:")
        logger.info(df.to_string())
        
        return df
    
    def generate_recommendations(self, baseline_loss: float) -> Dict:
        """
        Generate actionable recommendations.
        
        Args:
            baseline_loss: Baseline annual loss
        
        Returns:
            Dictionary with recommendations
        """
        cba = self.calculate_cost_benefit_analysis(baseline_loss)
        
        recommendations = {
            'summary': {
                'total_budget': self.budget,
                'total_cost': sum(s.cost for s in self.selected_strategies),
                'total_risk_reduction': sum(s.risk_reduction for s in self.selected_strategies),
                'baseline_loss': baseline_loss,
                'mitigated_loss': baseline_loss * (1 - sum(s.risk_reduction for s in self.selected_strategies))
            },
            'strategies': cba.to_dict('records'),
            'priority_order': cba.sort_values('Benefit_Cost_Ratio', ascending=False)['Strategy'].tolist()
        }
        
        return recommendations

# ============================================================================
# PREDEFINED STRATEGIES
# ============================================================================

PREDEFINED_STRATEGIES = {
    'infrastructure_hardening': {
        'cost': 500_000,
        'risk_reduction': 0.25,
        'implementation_time': 2,
        'effectiveness': 0.85
    },
    'insurance_coverage': {
        'cost': 100_000,
        'risk_reduction': 0.15,
        'implementation_time': 0.5,
        'effectiveness': 0.70
    },
    'relocation': {
        'cost': 1_000_000,
        'risk_reduction': 0.40,
        'implementation_time': 3,
        'effectiveness': 0.90
    },
    'early_warning_systems': {
        'cost': 200_000,
        'risk_reduction': 0.10,
        'implementation_time': 1,
        'effectiveness': 0.75
    },
    'land_use_planning': {
        'cost': 300_000,
        'risk_reduction': 0.20,
        'implementation_time': 2,
        'effectiveness': 0.80
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_mitigation_recommendations(baseline_loss: float, 
                                       budget: float = MITIGATION_CONFIG["budget_constraint"]) -> Dict:
    """
    Generate mitigation recommendations.
    
    Args:
        baseline_loss: Baseline annual loss
        budget: Available budget
    
    Returns:
        Dictionary with recommendations
    """
    optimizer = MitigationOptimizer(budget)
    optimizer.add_strategies_from_dict(PREDEFINED_STRATEGIES)
    optimizer.optimize_linear_programming()
    
    return optimizer.generate_recommendations(baseline_loss)

if __name__ == "__main__":
    # Example usage
    baseline_loss = 50_000_000  # $50M annual loss
    recommendations = generate_mitigation_recommendations(baseline_loss)
    
    print("\nMitigation Recommendations Summary:")
    print(f"Baseline Loss: ${recommendations['summary']['baseline_loss']:,.0f}")
    print(f"Mitigated Loss: ${recommendations['summary']['mitigated_loss']:,.0f}")
    print(f"Risk Reduction: {recommendations['summary']['total_risk_reduction']:.2%}")
    print(f"\nPriority Order: {recommendations['priority_order']}")

