"""
Ensemble Models Module for CATIA

Provides robust risk predictions by combining multiple ML models.
Ensemble methods reduce variance, improve stability, and provide
uncertainty estimates for catastrophe risk predictions.

Key approaches:
- Voting Ensemble: Combine classifier predictions (hard/soft voting)
- Stacking Ensemble: Meta-learner on top of base model predictions
- Model Averaging: Weighted average for regression predictions
- Bagging: Bootstrap aggregating for variance reduction
- Boosting: Sequential models focusing on hard cases

Benefits for catastrophe modeling:
- More stable predictions across different scenarios
- Reduced overfitting to historical patterns
- Built-in uncertainty quantification
- Resilience to model specification errors
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import warnings

from catia.config import ML_CONFIG, LOGGING_CONFIG

logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Results from ensemble prediction."""
    predictions: np.ndarray
    prediction_std: np.ndarray  # Uncertainty from disagreement
    model_weights: Dict[str, float]
    individual_predictions: Dict[str, np.ndarray]


# ============================================================================
# BASE MODEL FACTORY
# ============================================================================

def get_base_classifiers(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Get a set of diverse base classifiers for ensemble."""
    return {
        'rf': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        ),
        'ada': AdaBoostClassifier(
            n_estimators=50, random_state=random_state, algorithm='SAMME'
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(50, 25), max_iter=500, random_state=random_state
        ),
    }


def get_base_regressors(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Get a set of diverse base regressors for ensemble."""
    return {
        'rf': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=random_state
        ),
        'ada': AdaBoostRegressor(
            n_estimators=50, random_state=random_state
        ),
        'mlp': MLPRegressor(
            hidden_layer_sizes=(50, 25), max_iter=500, random_state=random_state
        ),
        'ridge': Ridge(alpha=1.0),
    }


# ============================================================================
# VOTING ENSEMBLE
# ============================================================================

class RobustVotingClassifier(ClassifierMixin, BaseEstimator):
    """
    Enhanced voting classifier with uncertainty quantification.
    
    Extends sklearn's VotingClassifier to provide:
    - Prediction uncertainty from model disagreement
    - Individual model predictions for analysis
    - Weighted voting based on cross-validation performance
    """
    
    def __init__(self, estimators: Dict[str, BaseEstimator] = None,
                 voting: str = 'soft',
                 auto_weight: bool = True):
        """
        Initialize robust voting classifier.
        
        Args:
            estimators: Dict of name -> classifier
            voting: 'hard' or 'soft' voting
            auto_weight: Whether to weight by CV performance
        """
        self.estimators = estimators or get_base_classifiers()
        self.voting = voting
        self.auto_weight = auto_weight
        self.weights_ = None
        self.fitted_estimators_ = {}
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustVotingClassifier':
        """Fit all base estimators."""
        self.classes_ = np.unique(y)
        weights = {}
        
        for name, estimator in self.estimators.items():
            logger.info(f"Fitting {name}...")
            est = clone(estimator)
            est.fit(X, y)
            self.fitted_estimators_[name] = est
            
            # Calculate weight from CV score
            if self.auto_weight:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_score = cross_val_score(clone(estimator), X, y, cv=3, scoring='f1').mean()
                weights[name] = max(cv_score, 0.1)  # Minimum weight
        
        # Normalize weights
        if self.auto_weight:
            total = sum(weights.values())
            self.weights_ = {k: v/total for k, v in weights.items()}
        else:
            self.weights_ = {k: 1.0/len(self.estimators) for k in self.estimators}

        logger.info(f"Ensemble weights: {self.weights_}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from ensemble."""
        probas = []
        weights = []

        for name, est in self.fitted_estimators_.items():
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X)
                probas.append(proba)
                weights.append(self.weights_[name])

        if not probas:
            raise ValueError("No estimators support predict_proba")

        # Weighted average
        weights = np.array(weights) / sum(weights)
        weighted_proba = sum(w * p for w, p in zip(weights, probas))
        return weighted_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.voting == 'soft':
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting
            predictions = np.array([
                est.predict(X) for est in self.fitted_estimators_.values()
            ])
            # Weighted majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0, arr=predictions
            )

    def predict_with_uncertainty(self, X: np.ndarray) -> EnsembleResult:
        """Predict with uncertainty quantification."""
        individual_preds = {}
        individual_probas = []

        for name, est in self.fitted_estimators_.items():
            individual_preds[name] = est.predict(X)
            if hasattr(est, 'predict_proba'):
                individual_probas.append(est.predict_proba(X)[:, 1])

        # Ensemble prediction
        predictions = self.predict(X)

        # Uncertainty from model disagreement
        if individual_probas:
            probas_array = np.array(individual_probas)
            prediction_std = np.std(probas_array, axis=0)
        else:
            pred_array = np.array([p for p in individual_preds.values()])
            prediction_std = np.std(pred_array, axis=0)

        return EnsembleResult(
            predictions=predictions,
            prediction_std=prediction_std,
            model_weights=self.weights_,
            individual_predictions=individual_preds
        )


class RobustVotingRegressor(RegressorMixin, BaseEstimator):
    """
    Enhanced voting regressor with uncertainty quantification.

    Provides prediction intervals based on model disagreement.
    """

    def __init__(self, estimators: Dict[str, BaseEstimator] = None,
                 auto_weight: bool = True):
        self.estimators = estimators or get_base_regressors()
        self.auto_weight = auto_weight
        self.weights_ = None
        self.fitted_estimators_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustVotingRegressor':
        """Fit all base estimators."""
        weights = {}

        for name, estimator in self.estimators.items():
            logger.info(f"Fitting {name}...")
            est = clone(estimator)
            est.fit(X, y)
            self.fitted_estimators_[name] = est

            if self.auto_weight:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_score = cross_val_score(
                        clone(estimator), X, y, cv=3,
                        scoring='neg_mean_squared_error'
                    ).mean()
                weights[name] = max(-cv_score, 0.001)  # Use inverse MSE

        if self.auto_weight:
            # Inverse weighting - lower MSE = higher weight
            inv_weights = {k: 1/v for k, v in weights.items()}
            total = sum(inv_weights.values())
            self.weights_ = {k: v/total for k, v in inv_weights.items()}
        else:
            self.weights_ = {k: 1.0/len(self.estimators) for k in self.estimators}

        logger.info(f"Ensemble weights: {self.weights_}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with weighted average."""
        predictions = []
        weights = []

        for name, est in self.fitted_estimators_.items():
            predictions.append(est.predict(X))
            weights.append(self.weights_[name])

        weights = np.array(weights)
        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=weights)

    def predict_with_uncertainty(self, X: np.ndarray) -> EnsembleResult:
        """Predict with uncertainty quantification."""
        individual_preds = {}
        pred_list = []

        for name, est in self.fitted_estimators_.items():
            pred = est.predict(X)
            individual_preds[name] = pred
            pred_list.append(pred)

        predictions = self.predict(X)
        pred_array = np.array(pred_list)
        prediction_std = np.std(pred_array, axis=0)

        return EnsembleResult(
            predictions=predictions,
            prediction_std=prediction_std,
            model_weights=self.weights_,
            individual_predictions=individual_preds
        )


# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

class RobustStackingClassifier(ClassifierMixin, BaseEstimator):
    """
    Stacking classifier with meta-learner.

    Base models make predictions, then a meta-learner combines them.
    More sophisticated than simple voting.
    """

    def __init__(self, base_estimators: Dict[str, BaseEstimator] = None,
                 meta_learner: BaseEstimator = None,
                 use_proba: bool = True):
        self.base_estimators = base_estimators or get_base_classifiers()
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        self.use_proba = use_proba
        self.fitted_base_ = {}
        self.fitted_meta_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustStackingClassifier':
        """Fit base models and meta-learner."""
        self.classes_ = np.unique(y)

        # Get cross-validated predictions from base models
        meta_features = []

        for name, estimator in self.base_estimators.items():
            logger.info(f"Fitting base model {name}...")
            est = clone(estimator)

            # Get CV predictions for meta-features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.use_proba and hasattr(est, 'predict_proba'):
                    cv_pred = cross_val_predict(est, X, y, cv=3, method='predict_proba')
                    meta_features.append(cv_pred[:, 1:])  # Exclude first class
                else:
                    cv_pred = cross_val_predict(est, X, y, cv=3)
                    meta_features.append(cv_pred.reshape(-1, 1))

            # Fit on full data
            est.fit(X, y)
            self.fitted_base_[name] = est

        # Stack meta-features
        meta_X = np.hstack(meta_features)

        # Fit meta-learner
        logger.info("Fitting meta-learner...")
        self.fitted_meta_ = clone(self.meta_learner)
        self.fitted_meta_.fit(meta_X, y)

        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions."""
        meta_features = []

        for name, est in self.fitted_base_.items():
            if self.use_proba and hasattr(est, 'predict_proba'):
                pred = est.predict_proba(X)[:, 1:]
            else:
                pred = est.predict(X).reshape(-1, 1)
            meta_features.append(pred)

        return np.hstack(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble."""
        meta_X = self._get_meta_features(X)
        return self.fitted_meta_.predict(meta_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        meta_X = self._get_meta_features(X)
        if hasattr(self.fitted_meta_, 'predict_proba'):
            return self.fitted_meta_.predict_proba(meta_X)
        else:
            pred = self.fitted_meta_.predict(meta_X)
            return np.column_stack([1 - pred, pred])


class RobustStackingRegressor(RegressorMixin, BaseEstimator):
    """Stacking regressor with meta-learner."""

    def __init__(self, base_estimators: Dict[str, BaseEstimator] = None,
                 meta_learner: BaseEstimator = None):
        self.base_estimators = base_estimators or get_base_regressors()
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.fitted_base_ = {}
        self.fitted_meta_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustStackingRegressor':
        """Fit base models and meta-learner."""
        meta_features = []

        for name, estimator in self.base_estimators.items():
            logger.info(f"Fitting base model {name}...")
            est = clone(estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_pred = cross_val_predict(est, X, y, cv=3)
                meta_features.append(cv_pred.reshape(-1, 1))

            est.fit(X, y)
            self.fitted_base_[name] = est

        meta_X = np.hstack(meta_features)

        logger.info("Fitting meta-learner...")
        self.fitted_meta_ = clone(self.meta_learner)
        self.fitted_meta_.fit(meta_X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble."""
        meta_features = []
        for est in self.fitted_base_.values():
            meta_features.append(est.predict(X).reshape(-1, 1))
        meta_X = np.hstack(meta_features)
        return self.fitted_meta_.predict(meta_X)


# ============================================================================
# ENSEMBLE RISK PREDICTOR
# ============================================================================

class EnsembleRiskPredictor:
    """
    Ensemble-based risk predictor for catastrophe modeling.

    Combines multiple ML models for both probability and severity prediction,
    providing more stable and robust predictions with uncertainty estimates.
    """

    def __init__(self,
                 ensemble_type: str = 'voting',
                 auto_weight: bool = True):
        """
        Initialize ensemble risk predictor.

        Args:
            ensemble_type: 'voting' or 'stacking'
            auto_weight: Whether to weight models by CV performance
        """
        self.ensemble_type = ensemble_type
        self.auto_weight = auto_weight
        self.probability_ensemble = None
        self.severity_ensemble = None
        self.scaler = None
        self.is_trained = False

        logger.info(f"EnsembleRiskPredictor initialized: {ensemble_type}")

    def fit(self, X: np.ndarray, y_probability: np.ndarray,
            y_severity: np.ndarray) -> 'EnsembleRiskPredictor':
        """
        Train ensemble models.

        Args:
            X: Feature matrix
            y_probability: Binary target for event occurrence
            y_severity: Continuous target for loss severity
        """
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create ensembles based on type
        if self.ensemble_type == 'voting':
            self.probability_ensemble = RobustVotingClassifier(
                auto_weight=self.auto_weight
            )
            self.severity_ensemble = RobustVotingRegressor(
                auto_weight=self.auto_weight
            )
        elif self.ensemble_type == 'stacking':
            self.probability_ensemble = RobustStackingClassifier()
            self.severity_ensemble = RobustStackingRegressor()
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

        # Fit models
        logger.info("Training probability ensemble...")
        self.probability_ensemble.fit(X_scaled, y_probability)

        logger.info("Training severity ensemble...")
        self.severity_ensemble.fit(X_scaled, y_severity)

        self.is_trained = True
        logger.info("Ensemble training complete")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict event probability and severity.

        Returns:
            Tuple of (probability_predictions, severity_predictions)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        prob_pred = self.probability_ensemble.predict(X_scaled)
        sev_pred = self.severity_ensemble.predict(X_scaled)

        return prob_pred, sev_pred

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict with uncertainty quantification.

        Returns dict with predictions, uncertainty, and model info.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Get predictions with uncertainty
        prob_result = self.probability_ensemble.predict_with_uncertainty(X_scaled)

        if hasattr(self.severity_ensemble, 'predict_with_uncertainty'):
            sev_result = self.severity_ensemble.predict_with_uncertainty(X_scaled)
        else:
            sev_pred = self.severity_ensemble.predict(X_scaled)
            sev_result = EnsembleResult(
                predictions=sev_pred,
                prediction_std=np.zeros_like(sev_pred),
                model_weights={},
                individual_predictions={}
            )

        return {
            'probability': {
                'predictions': prob_result.predictions,
                'uncertainty': prob_result.prediction_std,
                'model_weights': prob_result.model_weights
            },
            'severity': {
                'predictions': sev_result.predictions,
                'uncertainty': sev_result.prediction_std,
                'model_weights': sev_result.model_weights
            },
            'ensemble_type': self.ensemble_type
        }

    def get_model_contributions(self) -> Dict[str, Dict[str, float]]:
        """Get weight/contribution of each model in the ensemble."""
        return {
            'probability_models': getattr(
                self.probability_ensemble, 'weights_', {}
            ),
            'severity_models': getattr(
                self.severity_ensemble, 'weights_', {}
            )
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compare_ensemble_methods(X: np.ndarray, y_prob: np.ndarray,
                             y_sev: np.ndarray) -> Dict[str, Any]:
    """
    Compare different ensemble methods on the given data.

    Returns performance metrics for each method.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, yp_train, yp_test, ys_train, ys_test = train_test_split(
        X, y_prob, y_sev, test_size=0.2, random_state=42
    )

    results = {}

    # Single RandomForest baseline
    logger.info("Training baseline (single RandomForest)...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_scaled, yp_train)
    baseline_prob_acc = accuracy_score(yp_test, rf_clf.predict(X_test_scaled))

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_scaled, ys_train)
    baseline_sev_rmse = np.sqrt(mean_squared_error(ys_test, rf_reg.predict(X_test_scaled)))

    results['baseline'] = {
        'prob_accuracy': baseline_prob_acc,
        'sev_rmse': baseline_sev_rmse
    }

    # Voting ensemble
    logger.info("Training voting ensemble...")
    voting = EnsembleRiskPredictor(ensemble_type='voting')
    voting.fit(X_train, yp_train, ys_train)
    prob_pred, sev_pred = voting.predict(X_test)

    results['voting'] = {
        'prob_accuracy': accuracy_score(yp_test, prob_pred),
        'sev_rmse': np.sqrt(mean_squared_error(ys_test, sev_pred)),
        'model_weights': voting.get_model_contributions()
    }

    # Stacking ensemble
    logger.info("Training stacking ensemble...")
    stacking = EnsembleRiskPredictor(ensemble_type='stacking')
    stacking.fit(X_train, yp_train, ys_train)
    prob_pred, sev_pred = stacking.predict(X_test)

    results['stacking'] = {
        'prob_accuracy': accuracy_score(yp_test, prob_pred),
        'sev_rmse': np.sqrt(mean_squared_error(ys_test, sev_pred))
    }

    # Summary
    logger.info("=" * 50)
    logger.info("Ensemble Comparison Results:")
    logger.info(f"  Baseline - Prob Acc: {results['baseline']['prob_accuracy']:.4f}, "
                f"Sev RMSE: ${results['baseline']['sev_rmse']:,.0f}")
    logger.info(f"  Voting   - Prob Acc: {results['voting']['prob_accuracy']:.4f}, "
                f"Sev RMSE: ${results['voting']['sev_rmse']:,.0f}")
    logger.info(f"  Stacking - Prob Acc: {results['stacking']['prob_accuracy']:.4f}, "
                f"Sev RMSE: ${results['stacking']['sev_rmse']:,.0f}")

    return results

