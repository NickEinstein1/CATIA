"""
Tests for Explainability module.

Tests RiskExplainer, SHAP-based explanations, and feature importance.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification

from catia.explainability import (
    RiskExplainer,
    GlobalImportance,
    PredictionExplanation,
    FeatureContribution
)


class TestFeatureContribution:
    """Tests for FeatureContribution dataclass."""

    def test_creation(self):
        """Test FeatureContribution can be created."""
        contrib = FeatureContribution(
            feature_name="temperature",
            feature_value=35.0,
            shap_value=0.15,
            contribution_pct=25.0
        )
        assert contrib.feature_name == "temperature"
        assert contrib.shap_value == 0.15


class TestGlobalImportance:
    """Tests for GlobalImportance dataclass."""

    def test_creation(self):
        """Test GlobalImportance can be created."""
        importance = GlobalImportance(
            feature_names=["a", "b", "c"],
            importance_scores=np.array([0.5, 0.3, 0.2]),
            importance_std=np.array([0.1, 0.1, 0.1]),
            ranking=[("a", 0.5), ("b", 0.3), ("c", 0.2)]
        )
        assert len(importance.feature_names) == 3
        assert importance.ranking[0][0] == "a"


class TestRiskExplainer:
    """Tests for RiskExplainer."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_classes=2, random_state=42
        )
        return X, y

    @pytest.fixture
    def trained_model(self, classification_data):
        """Create and train a model."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def feature_names(self):
        """Feature names for test data."""
        return ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]

    def test_initialization(self, trained_model, feature_names):
        """Test explainer initialization."""
        explainer = RiskExplainer(
            model=trained_model,
            feature_names=feature_names,
            background_samples=50
        )
        assert explainer.model is trained_model
        assert explainer.feature_names == feature_names
        assert explainer.background_samples == 50

    def test_fit(self, trained_model, classification_data):
        """Test explainer fitting."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model, background_samples=50)
        explainer.fit(X)
        
        assert explainer.explainer is not None
        assert explainer.background_data is not None
        assert len(explainer.background_data) <= 50

    def test_get_shap_values(self, trained_model, classification_data):
        """Test SHAP value computation."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model, background_samples=50)
        explainer.fit(X)
        
        shap_values = explainer.get_shap_values(X[:10])
        
        assert shap_values.shape[0] == 10
        assert shap_values.shape[1] == X.shape[1]

    def test_get_shap_values_unfitted_raises_error(self, trained_model, classification_data):
        """Test error when explainer not fitted."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model)
        
        with pytest.raises(ValueError, match="not fitted"):
            explainer.get_shap_values(X[:5])

    def test_get_global_importance(self, trained_model, classification_data, feature_names):
        """Test global importance calculation."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model, feature_names=feature_names)
        explainer.fit(X)
        
        importance = explainer.get_global_importance(X[:20])
        
        assert isinstance(importance, GlobalImportance)
        assert len(importance.feature_names) == 5
        assert len(importance.ranking) == 5
        # Ranking should be sorted by importance (descending)
        assert importance.ranking[0][1] >= importance.ranking[-1][1]

    def test_explain_prediction(self, trained_model, classification_data, feature_names):
        """Test local prediction explanation."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model, feature_names=feature_names)
        explainer.fit(X)
        
        explanation = explainer.explain_prediction(X[:5], index=0)
        
        assert isinstance(explanation, PredictionExplanation)
        assert len(explanation.contributions) == 5
        # Should have predictions and base value
        assert explanation.prediction is not None
        assert explanation.base_value is not None

    def test_explain_prediction_contributions_sum(self, trained_model, classification_data):
        """Test that contribution percentages sum to ~100%."""
        X, y = classification_data
        explainer = RiskExplainer(trained_model)
        explainer.fit(X)
        
        explanation = explainer.explain_prediction(X[:5], index=0)
        
        total_pct = sum(c.contribution_pct for c in explanation.contributions)
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_with_regressor(self, classification_data):
        """Test explainer works with regressor."""
        X, y = classification_data
        y_reg = y.astype(float)  # Convert to regression target
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y_reg)
        
        explainer = RiskExplainer(model)
        explainer.fit(X)
        
        shap_values = explainer.get_shap_values(X[:5])
        assert shap_values.shape == (5, X.shape[1])

