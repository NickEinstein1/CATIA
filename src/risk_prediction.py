"""
Risk Prediction Module for CATIA
Machine learning model for predicting climate catastrophe probability and severity.
Includes model validation using actuarial metrics.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ML_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# ============================================================================
# RISK PREDICTION CLASS
# ============================================================================

class RiskPredictor:
    """Machine learning model for catastrophe risk prediction."""
    
    def __init__(self):
        """Initialize the risk predictor."""
        self.probability_model = None
        self.severity_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        logger.info("RiskPredictor initialized")
    
    def prepare_features(self, climate_data: pd.DataFrame, 
                        socioeconomic_data: pd.DataFrame,
                        historical_events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features for model training.
        
        Args:
            climate_data: Climate variables
            socioeconomic_data: Socioeconomic variables
            historical_events: Historical event data with losses
        
        Returns:
            Tuple of (features_df, event_probability, event_severity)
        """
        # Aggregate climate data to match event frequency
        climate_agg = climate_data.groupby(pd.Grouper(key='date', freq='M')).agg({
            'temperature': 'mean',
            'precipitation': 'sum',
            'wind_speed': 'max',
            'sea_level_pressure': 'min',
            'humidity': 'mean'
        }).reset_index()
        
        # Create features
        features = climate_agg.copy()
        
        # Add socioeconomic features (repeat for each month)
        for col in socioeconomic_data.columns:
            if col != 'region':
                features[col] = socioeconomic_data[col].values[0]
        
        # Create target variables
        # Event probability: 1 if event occurred that month, 0 otherwise
        event_months = set(pd.to_datetime(historical_events['year'].astype(str)).dt.to_period('M'))
        features['event_occurred'] = features['date'].dt.to_period('M').isin(event_months).astype(int)
        
        # Event severity: loss amount (0 if no event)
        severity_map = dict(zip(
            pd.to_datetime(historical_events['year'].astype(str)).dt.to_period('M'),
            historical_events['loss_usd']
        ))
        features['event_severity'] = features['date'].dt.to_period('M').map(severity_map).fillna(0)
        
        # Prepare X and y
        feature_cols = [col for col in features.columns if col not in ['date', 'event_occurred', 'event_severity']]
        X = features[feature_cols].copy()
        y_prob = features['event_occurred']
        y_sev = features['event_severity']
        
        self.feature_names = feature_cols
        logger.info(f"Features prepared: {len(X)} samples, {len(feature_cols)} features")
        
        return X, y_prob, y_sev
    
    def train(self, X: pd.DataFrame, y_probability: pd.Series, y_severity: pd.Series):
        """
        Train probability and severity models.
        
        Args:
            X: Feature matrix
            y_probability: Binary target for event occurrence
            y_severity: Continuous target for loss severity
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        split_ratio = ML_CONFIG["train_test_split"]
        X_train, X_test, y_prob_train, y_prob_test, y_sev_train, y_sev_test = train_test_split(
            X_scaled, y_probability, y_severity,
            test_size=1-split_ratio, random_state=ML_CONFIG["hyperparameters"]["random_state"]
        )
        
        # Train probability model
        logger.info("Training probability model...")
        self.probability_model = RandomForestClassifier(**ML_CONFIG["hyperparameters"])
        self.probability_model.fit(X_train, y_prob_train)
        
        # Train severity model
        logger.info("Training severity model...")
        self.severity_model = RandomForestRegressor(**ML_CONFIG["hyperparameters"])
        self.severity_model.fit(X_train, y_sev_train)
        
        # Validate models
        self._validate_models(X_test, y_prob_test, y_sev_test)
        
        self.is_trained = True
        logger.info("Models trained successfully")
    
    def _validate_models(self, X_test: np.ndarray, y_prob_test: pd.Series, y_sev_test: pd.Series):
        """Validate models using actuarial metrics."""
        # Probability model validation
        y_prob_pred = self.probability_model.predict(X_test)
        accuracy = accuracy_score(y_prob_test, y_prob_pred)
        precision = precision_score(y_prob_test, y_prob_pred, zero_division=0)
        recall = recall_score(y_prob_test, y_prob_pred, zero_division=0)
        f1 = f1_score(y_prob_test, y_prob_pred, zero_division=0)
        
        logger.info(f"Probability Model Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # Severity model validation
        y_sev_pred = self.severity_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_sev_test, y_sev_pred))
        mae = np.mean(np.abs(y_sev_test - y_sev_pred))
        
        logger.info(f"Severity Model Metrics:")
        logger.info(f"  RMSE: ${rmse:,.0f}")
        logger.info(f"  MAE: ${mae:,.0f}")
        
        # Loss ratio accuracy (actuarial metric)
        total_predicted_loss = y_sev_pred.sum()
        total_actual_loss = y_sev_test.sum()
        loss_ratio = total_predicted_loss / total_actual_loss if total_actual_loss > 0 else 0
        logger.info(f"  Loss Ratio Accuracy: {loss_ratio:.4f}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict event probability and severity.
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (probability_predictions, severity_predictions)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get probability predictions
        prob_predictions = self.probability_model.predict_proba(X_scaled)[:, 1]
        
        # Get severity predictions
        severity_predictions = self.severity_model.predict(X_scaled)
        severity_predictions = np.maximum(severity_predictions, 0)  # Ensure non-negative
        
        return prob_predictions, severity_predictions
    
    def save_model(self, path: str = ML_CONFIG["model_path"]):
        """Save trained models to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'probability_model': self.probability_model,
                'severity_model': self.severity_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Models saved to {path}")
    
    def load_model(self, path: str = ML_CONFIG["model_path"]):
        """Load trained models from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.probability_model = data['probability_model']
            self.severity_model = data['severity_model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        self.is_trained = True
        logger.info(f"Models loaded from {path}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_risk_model(climate_data: pd.DataFrame, 
                    socioeconomic_data: pd.DataFrame,
                    historical_events: pd.DataFrame) -> RiskPredictor:
    """
    Train a complete risk prediction model.
    
    Args:
        climate_data: Climate variables
        socioeconomic_data: Socioeconomic variables
        historical_events: Historical event data
    
    Returns:
        Trained RiskPredictor instance
    """
    predictor = RiskPredictor()
    X, y_prob, y_sev = predictor.prepare_features(climate_data, socioeconomic_data, historical_events)
    predictor.train(X, y_prob, y_sev)
    predictor.save_model()
    return predictor

if __name__ == "__main__":
    # Example usage
    from data_acquisition import fetch_all_data
    
    data = fetch_all_data("US_Gulf_Coast", use_mock=True)
    predictor = train_risk_model(
        data['climate'],
        data['socioeconomic'],
        data['historical_events']
    )
    
    # Make predictions
    X, _, _ = predictor.prepare_features(
        data['climate'],
        data['socioeconomic'],
        data['historical_events']
    )
    probs, severities = predictor.predict(X.head(10))
    
    print("\nPredictions (first 10 samples):")
    for i, (prob, sev) in enumerate(zip(probs, severities)):
        print(f"  Sample {i+1}: Probability={prob:.4f}, Severity=${sev:,.0f}")

