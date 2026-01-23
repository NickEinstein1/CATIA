"""
Tests for Ensemble Models module.

Tests RobustVotingClassifier, RobustVotingRegressor, and model factories.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from catia.ensemble import (
    RobustVotingClassifier,
    RobustVotingRegressor,
    get_base_classifiers,
    get_base_regressors,
    EnsembleResult
)


class TestModelFactories:
    """Tests for model factory functions."""

    def test_get_base_classifiers(self):
        """Test base classifiers factory."""
        classifiers = get_base_classifiers(random_state=42)
        
        assert isinstance(classifiers, dict)
        assert len(classifiers) >= 3
        assert 'rf' in classifiers
        assert 'gb' in classifiers

    def test_get_base_regressors(self):
        """Test base regressors factory."""
        regressors = get_base_regressors(random_state=42)
        
        assert isinstance(regressors, dict)
        assert len(regressors) >= 3
        assert 'rf' in regressors
        assert 'gb' in regressors

    def test_classifiers_different_random_states(self):
        """Test different random states produce different classifiers."""
        clf1 = get_base_classifiers(random_state=42)
        clf2 = get_base_classifiers(random_state=123)
        
        # Should be different objects
        assert clf1['rf'] is not clf2['rf']


class TestRobustVotingClassifier:
    """Tests for RobustVotingClassifier."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_classes=2, random_state=42
        )
        return X, y

    @pytest.fixture
    def classifier(self):
        """Create classifier with limited estimators for speed."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        return RobustVotingClassifier(
            estimators={
                'rf': RandomForestClassifier(n_estimators=10, random_state=42),
                'lr': LogisticRegression(max_iter=200, random_state=42)
            },
            voting='soft',
            auto_weight=False  # Faster for tests
        )

    def test_initialization(self):
        """Test classifier initialization."""
        clf = RobustVotingClassifier()
        assert clf.voting == 'soft'
        assert clf.auto_weight is True

    def test_fit(self, classifier, classification_data):
        """Test fitting classifier."""
        X, y = classification_data
        classifier.fit(X, y)
        
        assert len(classifier.fitted_estimators_) == 2
        assert classifier.weights_ is not None

    def test_predict(self, classifier, classification_data):
        """Test prediction."""
        X, y = classification_data
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classifier, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        classifier.fit(X, y)
        probas = classifier.predict_proba(X)
        
        assert probas.shape == (len(y), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        np.testing.assert_array_almost_equal(probas.sum(axis=1), 1.0)

    def test_predict_with_uncertainty(self, classifier, classification_data):
        """Test prediction with uncertainty."""
        X, y = classification_data
        classifier.fit(X, y)
        result = classifier.predict_with_uncertainty(X)
        
        assert isinstance(result, EnsembleResult)
        assert len(result.predictions) == len(y)
        assert len(result.prediction_std) == len(y)
        assert len(result.individual_predictions) == 2


class TestRobustVotingRegressor:
    """Tests for RobustVotingRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=5,
            noise=10, random_state=42
        )
        return X, y

    @pytest.fixture
    def regressor(self):
        """Create regressor with limited estimators for speed."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        return RobustVotingRegressor(
            estimators={
                'rf': RandomForestRegressor(n_estimators=10, random_state=42),
                'ridge': Ridge(alpha=1.0)
            },
            auto_weight=False
        )

    def test_initialization(self):
        """Test regressor initialization."""
        reg = RobustVotingRegressor()
        assert reg.auto_weight is True

    def test_fit(self, regressor, regression_data):
        """Test fitting regressor."""
        X, y = regression_data
        regressor.fit(X, y)
        
        assert len(regressor.fitted_estimators_) == 2
        assert regressor.weights_ is not None

    def test_predict(self, regressor, regression_data):
        """Test prediction."""
        X, y = regression_data
        regressor.fit(X, y)
        predictions = regressor.predict(X)
        
        assert len(predictions) == len(y)
        # Predictions should be continuous
        assert predictions.dtype in [np.float64, np.float32]

