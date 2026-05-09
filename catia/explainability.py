"""
Explainability Module for CATIA

Provides SHAP-based model interpretability for catastrophe risk predictions.
Helps stakeholders understand WHY risk is high or low for specific scenarios.

Key features:
- Global feature importance: Which factors drive risk overall
- Local explanations: Why this specific prediction was made
- Feature interactions: How factors combine to affect risk
- Counterfactual analysis: What would change the prediction

Use cases in catastrophe modeling:
- Regulatory compliance (explainable AI requirements)
- Underwriting decisions (justify risk assessments)
- Risk management (identify key risk drivers)
- Model validation (ensure predictions make physical sense)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from catia.config import LOGGING_CONFIG

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Contribution of a single feature to a prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_pct: float  # Percentage of total absolute SHAP


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction."""
    prediction: float
    base_value: float  # Expected prediction (average)
    contributions: List[FeatureContribution]
    top_positive: List[FeatureContribution]  # Top factors increasing risk
    top_negative: List[FeatureContribution]  # Top factors decreasing risk


@dataclass
class GlobalImportance:
    """Global feature importance across all predictions."""
    feature_names: List[str]
    importance_scores: np.ndarray  # Mean absolute SHAP values
    importance_std: np.ndarray  # Standard deviation
    ranking: List[Tuple[str, float]]  # Sorted (feature, importance)


# ============================================================================
# MODEL EXPLAINER
# ============================================================================

class RiskExplainer:
    """
    SHAP-based explainer for risk prediction models.
    
    Provides both global feature importance and local prediction explanations.
    """
    
    def __init__(self, model: Any, feature_names: List[str] = None,
                 background_samples: int = 100):
        """
        Initialize the explainer.
        
        Args:
            model: Trained sklearn-compatible model
            feature_names: Names of input features
            background_samples: Number of background samples for SHAP
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Run: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        logger.info("RiskExplainer initialized")
    
    def fit(self, X: np.ndarray) -> 'RiskExplainer':
        """
        Fit the explainer with background data.
        
        Args:
            X: Training data for background distribution
        """
        # Sample background data if needed
        if len(X) > self.background_samples:
            indices = np.random.choice(len(X), self.background_samples, replace=False)
            self.background_data = X[indices]
        else:
            self.background_data = X
        
        # Create appropriate explainer based on model type
        model_type = type(self.model).__name__
        
        if hasattr(self.model, 'estimators_'):
            # Tree-based ensemble (RandomForest, GradientBoosting)
            logger.info(f"Using TreeExplainer for {model_type}")
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Generic model - use KernelExplainer
            logger.info(f"Using KernelExplainer for {model_type}")
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, self.background_data
                )
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict, self.background_data
                )
        
        return self
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for given samples."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        shap_values = self.explainer.shap_values(X)

        # Handle multi-class output (take positive class for binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Handle 3D arrays (n_samples, n_features, n_classes) - take positive class
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]

        return shap_values

    def get_global_importance(self, X: np.ndarray) -> GlobalImportance:
        """
        Calculate global feature importance using mean |SHAP|.

        Args:
            X: Data to explain

        Returns:
            GlobalImportance with rankings
        """
        shap_values = self.get_shap_values(X)

        # Ensure 2D array
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        importance_std = np.abs(shap_values).std(axis=0)

        # Handle case where importance might still be multi-dimensional
        if importance.ndim > 1:
            importance = importance.mean(axis=-1)
            importance_std = importance_std.mean(axis=-1)

        # Create ranking
        names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        ranking = sorted(zip(names, importance.tolist()), key=lambda x: -x[1])

        return GlobalImportance(
            feature_names=names,
            importance_scores=importance,
            importance_std=importance_std,
            ranking=ranking
        )

    def explain_prediction(self, X: np.ndarray, index: int = 0) -> PredictionExplanation:
        """
        Get detailed explanation for a single prediction.

        Args:
            X: Input data
            index: Which sample to explain

        Returns:
            PredictionExplanation with feature contributions
        """
        shap_values = self.get_shap_values(X)
        sample_shap = shap_values[index]
        sample_features = X[index]

        # Get prediction and base value
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X[index:index+1])[0, 1]
        else:
            prediction = self.model.predict(X[index:index+1])[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) == 2 else base_value[0]

        # Build contributions
        names = self.feature_names or [f"feature_{i}" for i in range(len(sample_shap))]
        total_abs_shap = np.abs(sample_shap).sum()

        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(names, sample_features, sample_shap)):
            contrib_pct = abs(shap_val) / total_abs_shap * 100 if total_abs_shap > 0 else 0
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(value),
                shap_value=float(shap_val),
                contribution_pct=float(contrib_pct)
            ))

        # Sort for top positive/negative
        sorted_contribs = sorted(contributions, key=lambda x: x.shap_value, reverse=True)
        top_positive = [c for c in sorted_contribs if c.shap_value > 0][:5]
        top_negative = [c for c in sorted_contribs if c.shap_value < 0][-5:][::-1]

        return PredictionExplanation(
            prediction=float(prediction),
            base_value=float(base_value),
            contributions=contributions,
            top_positive=top_positive,
            top_negative=top_negative
        )

    def explain_batch(self, X: np.ndarray) -> List[PredictionExplanation]:
        """Explain multiple predictions."""
        return [self.explain_prediction(X, i) for i in range(len(X))]


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_global_importance(importance: GlobalImportance, top_n: int = 10) -> str:
    """Format global importance as a readable report."""
    lines = [
        "=" * 60,
        "GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)",
        "=" * 60,
        ""
    ]

    for rank, (feature, score) in enumerate(importance.ranking[:top_n], 1):
        bar_len = int(score / importance.ranking[0][1] * 30)
        bar = "█" * bar_len
        lines.append(f"{rank:2d}. {feature:25s} {score:8.4f} {bar}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_prediction_explanation(explanation: PredictionExplanation,
                                   title: str = "Prediction Explanation") -> str:
    """Format prediction explanation as a readable report."""
    lines = [
        "=" * 60,
        title,
        "=" * 60,
        "",
        f"Prediction: {explanation.prediction:.4f}",
        f"Base value (avg): {explanation.base_value:.4f}",
        f"Difference: {explanation.prediction - explanation.base_value:+.4f}",
        ""
    ]

    # Top factors increasing risk
    lines.append("📈 TOP FACTORS INCREASING RISK:")
    lines.append("-" * 40)
    for contrib in explanation.top_positive:
        lines.append(
            f"  {contrib.feature_name:20s} = {contrib.feature_value:8.2f} "
            f"→ {contrib.shap_value:+.4f} ({contrib.contribution_pct:.1f}%)"
        )

    lines.append("")

    # Top factors decreasing risk
    lines.append("📉 TOP FACTORS DECREASING RISK:")
    lines.append("-" * 40)
    for contrib in explanation.top_negative:
        lines.append(
            f"  {contrib.feature_name:20s} = {contrib.feature_value:8.2f} "
            f"→ {contrib.shap_value:+.4f} ({contrib.contribution_pct:.1f}%)"
        )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def get_risk_drivers(explanation: PredictionExplanation,
                     threshold_pct: float = 5.0) -> Dict[str, List[Dict]]:
    """
    Extract key risk drivers from an explanation.

    Args:
        explanation: Prediction explanation
        threshold_pct: Minimum contribution percentage to include

    Returns:
        Dict with 'increasing' and 'decreasing' risk factors
    """
    increasing = []
    decreasing = []

    for contrib in explanation.contributions:
        if contrib.contribution_pct >= threshold_pct:
            factor = {
                'feature': contrib.feature_name,
                'value': contrib.feature_value,
                'impact': contrib.shap_value,
                'contribution_pct': contrib.contribution_pct
            }
            if contrib.shap_value > 0:
                increasing.append(factor)
            else:
                decreasing.append(factor)

    return {
        'increasing': sorted(increasing, key=lambda x: -x['impact']),
        'decreasing': sorted(decreasing, key=lambda x: x['impact'])
    }


# ============================================================================
# ENSEMBLE EXPLAINER
# ============================================================================

class EnsembleExplainer:
    """
    Explainer for ensemble risk prediction models.

    Aggregates explanations across ensemble members.
    """

    def __init__(self, ensemble_model: Any, feature_names: List[str] = None):
        """
        Initialize ensemble explainer.

        Args:
            ensemble_model: Trained ensemble model
            feature_names: Names of input features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Run: pip install shap")

        self.ensemble_model = ensemble_model
        self.feature_names = feature_names
        self.member_explainers = {}

        logger.info("EnsembleExplainer initialized")

    def fit(self, X: np.ndarray) -> 'EnsembleExplainer':
        """Fit explainers for each ensemble member."""
        # Check for fitted_estimators_ attribute (our custom ensembles)
        if hasattr(self.ensemble_model, 'fitted_estimators_'):
            estimators = self.ensemble_model.fitted_estimators_
        elif hasattr(self.ensemble_model, 'estimators_'):
            estimators = {f"model_{i}": est for i, est in enumerate(self.ensemble_model.estimators_)}
        else:
            # Single model - wrap it
            estimators = {'model': self.ensemble_model}

        for name, model in estimators.items():
            try:
                explainer = RiskExplainer(model, self.feature_names)
                explainer.fit(X)
                self.member_explainers[name] = explainer
                logger.info(f"Fitted explainer for {name}")
            except Exception as e:
                logger.warning(f"Could not create explainer for {name}: {e}")

        return self

    def get_aggregated_importance(self, X: np.ndarray) -> GlobalImportance:
        """Get feature importance aggregated across ensemble members."""
        all_importances = []

        for name, explainer in self.member_explainers.items():
            importance = explainer.get_global_importance(X)
            all_importances.append(importance.importance_scores)

        if not all_importances:
            raise ValueError("No explainers available")

        # Average importance across members
        avg_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)

        names = self.feature_names or [f"feature_{i}" for i in range(len(avg_importance))]
        ranking = sorted(zip(names, avg_importance), key=lambda x: -x[1])

        return GlobalImportance(
            feature_names=names,
            importance_scores=avg_importance,
            importance_std=std_importance,
            ranking=ranking
        )

    def explain_prediction(self, X: np.ndarray, index: int = 0) -> Dict[str, Any]:
        """
        Explain a prediction with consensus across ensemble members.

        Returns dict with individual member explanations and consensus.
        """
        member_explanations = {}
        all_shap_values = []

        for name, explainer in self.member_explainers.items():
            try:
                shap_values = explainer.get_shap_values(X)
                all_shap_values.append(shap_values[index])
                member_explanations[name] = explainer.explain_prediction(X, index)
            except Exception as e:
                logger.warning(f"Could not explain with {name}: {e}")

        # Consensus SHAP values (average)
        if all_shap_values:
            consensus_shap = np.mean(all_shap_values, axis=0)
            shap_std = np.std(all_shap_values, axis=0)
        else:
            consensus_shap = None
            shap_std = None

        return {
            'member_explanations': member_explanations,
            'consensus_shap': consensus_shap,
            'shap_uncertainty': shap_std,
            'num_members': len(member_explanations)
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def explain_risk_prediction(model: Any, X_train: np.ndarray, X_explain: np.ndarray,
                            feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to explain risk predictions.

    Args:
        model: Trained model
        X_train: Training data for background
        X_explain: Data to explain
        feature_names: Feature names

    Returns:
        Dict with global importance and individual explanations
    """
    explainer = RiskExplainer(model, feature_names)
    explainer.fit(X_train)

    global_importance = explainer.get_global_importance(X_explain)
    explanations = explainer.explain_batch(X_explain)

    return {
        'global_importance': global_importance,
        'explanations': explanations,
        'formatted_importance': format_global_importance(global_importance)
    }
