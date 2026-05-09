"""
Unit tests for risk prediction module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from catia.risk_prediction import RiskPredictor, train_risk_model
from catia.data_acquisition import fetch_all_data


class TestRiskPredictor:
    """Test cases for RiskPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return fetch_all_data("US_Gulf_Coast", use_mock=True)
    
    @pytest.fixture
    def predictor(self):
        """Create RiskPredictor instance."""
        return RiskPredictor()
    
    @pytest.fixture
    def trained_predictor(self, sample_data):
        """Create trained RiskPredictor instance."""
        predictor = RiskPredictor()
        X, y_prob, y_sev = predictor.prepare_features(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        predictor.train(X, y_prob, y_sev)
        return predictor
    
    def test_initialization(self, predictor):
        """Test RiskPredictor initialization."""
        assert predictor.probability_model is None
        assert predictor.severity_model is None
        assert predictor.scaler is not None
        assert predictor.feature_names is None
        assert predictor.is_trained is False
    
    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation."""
        X, y_prob, y_sev = predictor.prepare_features(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        
        # Check X is a DataFrame with expected shape
        assert isinstance(X, pd.DataFrame)
        assert len(X) > 0
        assert X.shape[1] > 0
        
        # Check y_prob is binary
        assert isinstance(y_prob, pd.Series)
        assert set(y_prob.unique()).issubset({0, 1})
        
        # Check y_sev is non-negative
        assert isinstance(y_sev, pd.Series)
        assert (y_sev >= 0).all()
        
        # Check feature names are set
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == X.shape[1]
    
    def test_train(self, predictor, sample_data):
        """Test model training."""
        X, y_prob, y_sev = predictor.prepare_features(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        
        predictor.train(X, y_prob, y_sev)
        
        assert predictor.probability_model is not None
        assert predictor.severity_model is not None
        assert predictor.is_trained is True
    
    def test_predict_untrained_raises_error(self, predictor, sample_data):
        """Test that prediction without training raises error."""
        X, _, _ = predictor.prepare_features(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        
        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict(X)
    
    def test_predict(self, trained_predictor, sample_data):
        """Test predictions on trained model."""
        X, _, _ = trained_predictor.prepare_features(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        
        probs, severities = trained_predictor.predict(X)
        
        # Check probability predictions
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(X)
        assert (probs >= 0).all() and (probs <= 1).all()
        
        # Check severity predictions
        assert isinstance(severities, np.ndarray)
        assert len(severities) == len(X)
        assert (severities >= 0).all()  # Non-negative
    
    def test_save_and_load_model(self, trained_predictor):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            
            # Save model
            trained_predictor.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model into new predictor
            new_predictor = RiskPredictor()
            new_predictor.load_model(model_path)
            
            assert new_predictor.is_trained is True
            assert new_predictor.probability_model is not None
            assert new_predictor.severity_model is not None
            assert new_predictor.feature_names is not None
    
    def test_train_risk_model_helper(self, sample_data):
        """Test train_risk_model helper function."""
        predictor = train_risk_model(
            sample_data['climate'],
            sample_data['socioeconomic'],
            sample_data['historical_events']
        )
        
        assert isinstance(predictor, RiskPredictor)
        assert predictor.is_trained is True
    
    def test_feature_importance(self, trained_predictor):
        """Test that models have feature importance attributes."""
        prob_importance = trained_predictor.probability_model.feature_importances_
        sev_importance = trained_predictor.severity_model.feature_importances_

        # Check that feature importance arrays have correct length
        assert len(prob_importance) == len(trained_predictor.feature_names)
        assert len(sev_importance) == len(trained_predictor.feature_names)

        # Feature importances should sum to ~1.0 or be all zeros (if no splits made)
        prob_sum = sum(prob_importance)
        sev_sum = sum(sev_importance)
        assert prob_sum == pytest.approx(1.0, rel=1e-5) or prob_sum == 0.0
        assert sev_sum == pytest.approx(1.0, rel=1e-5) or sev_sum == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

