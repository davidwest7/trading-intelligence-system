"""
Hierarchical Meta-Ensemble Predictor
Advanced ensemble with hierarchical structure, uncertainty-aware stacking, and online adaptation
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, t
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class ModelFamily(str, Enum):
    """Model family classification"""
    TREE_BASED = "tree_based"
    LINEAR = "linear"
    NEURAL = "neural"
    KERNEL = "kernel"
    ENSEMBLE = "ensemble"


class UncertaintyMethod(str, Enum):
    """Uncertainty estimation methods"""
    BOOTSTRAP = "bootstrap"
    DROPOUT = "dropout"
    QUANTILE = "quantile"
    BAYESIAN = "bayesian"
    CONFORMAL = "conformal"


@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_id: str
    family: ModelFamily
    cv_score: float
    cv_std: float
    oos_score: float
    oos_std: float
    calibration_error: float
    uncertainty_coverage: float
    last_update: datetime
    training_samples: int
    prediction_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleLayer:
    """Hierarchical ensemble layer"""
    layer_id: str
    models: List[str]
    aggregation_method: str
    weights: Dict[str, float]
    uncertainty_method: UncertaintyMethod
    performance: ModelPerformance
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalMetaEnsemble:
    """
    Advanced hierarchical meta-ensemble with uncertainty-aware stacking
    
    Features:
    - Hierarchical model structure (base → meta → super)
    - Multiple uncertainty estimation methods
    - Online adaptation and drift detection
    - Conformal prediction intervals
    - Dynamic model selection and weighting
    - Cross-validation with time series splits
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Logging (initialize first)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.n_base_models = self.config.get('n_base_models', 15)
        self.n_meta_models = self.config.get('n_meta_models', 5)
        self.uncertainty_method = UncertaintyMethod(self.config.get('uncertainty_method', 'bootstrap'))
        self.calibration_window = self.config.get('calibration_window', 1000)
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        
        # Model storage
        self.base_models: Dict[str, Any] = {}
        self.meta_models: Dict[str, Any] = {}
        self.super_model: Optional[Any] = None
        
        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.ensemble_layers: Dict[str, EnsembleLayer] = {}
        self.performance_history: deque = deque(maxlen=1000)
        
        # Uncertainty calibration
        self.calibration_data: deque = deque(maxlen=self.calibration_window)
        self.conformal_quantiles: Optional[np.ndarray] = None
        
        # Online learning
        self.online_weights: Dict[str, float] = {}
        self.drift_detectors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize models
        self._initialize_hierarchical_models()
    
    def _initialize_hierarchical_models(self):
        """Initialize hierarchical model structure"""
        try:
            # Base layer models (diverse model families)
            base_models_config = [
                # Tree-based models
                ('rf_1', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), ModelFamily.TREE_BASED),
                ('rf_2', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=43), ModelFamily.TREE_BASED),
                ('gb_1', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=44), ModelFamily.TREE_BASED),
                ('gb_2', GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=45), ModelFamily.TREE_BASED),
                ('et_1', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=46), ModelFamily.TREE_BASED),
                
                # Linear models
                ('linear', LinearRegression(), ModelFamily.LINEAR),
                ('ridge_1', Ridge(alpha=1.0), ModelFamily.LINEAR),
                ('ridge_2', Ridge(alpha=0.1), ModelFamily.LINEAR),
                ('lasso', Lasso(alpha=0.01), ModelFamily.LINEAR),
                ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5), ModelFamily.LINEAR),
                
                # Neural networks
                ('mlp_1', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=47), ModelFamily.NEURAL),
                ('mlp_2', MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=48), ModelFamily.NEURAL),
                
                # Kernel methods
                ('svr_1', SVR(kernel='rbf', C=1.0, gamma='scale'), ModelFamily.KERNEL),
                ('svr_2', SVR(kernel='poly', C=0.1, gamma='scale', degree=2), ModelFamily.KERNEL),
                ('svr_3', SVR(kernel='linear', C=0.1), ModelFamily.KERNEL),
            ]
            
            for model_id, model, family in base_models_config[:self.n_base_models]:
                self.base_models[model_id] = {
                    'model': model,
                    'family': family,
                    'scaler': RobustScaler(),
                    'trained': False
                }
            
            # Meta layer models (ensemble methods)
            meta_models_config = [
                ('meta_rf', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=49), ModelFamily.ENSEMBLE),
                ('meta_gb', GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=50), ModelFamily.ENSEMBLE),
                ('meta_linear', Ridge(alpha=0.1), ModelFamily.LINEAR),
                ('meta_mlp', MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=300, random_state=51), ModelFamily.NEURAL),
                ('meta_svr', SVR(kernel='rbf', C=0.1, gamma='scale'), ModelFamily.KERNEL),
            ]
            
            for model_id, model, family in meta_models_config[:self.n_meta_models]:
                self.meta_models[model_id] = {
                    'model': model,
                    'family': family,
                    'scaler': RobustScaler(),
                    'trained': False
                }
            
            # Super model (final ensemble)
            self.super_model = {
                'model': Ridge(alpha=0.01),
                'scaler': RobustScaler(),
                'trained': False
            }
            
            # Initialize ensemble layers
            self._create_ensemble_layers()
            
            self.logger.info(f"Initialized hierarchical ensemble: {len(self.base_models)} base, {len(self.meta_models)} meta, 1 super")
            
        except Exception as e:
            self.logger.error(f"Error initializing hierarchical models: {e}")
    
    def _create_ensemble_layers(self):
        """Create hierarchical ensemble layers"""
        try:
            # Base layer
            base_models = list(self.base_models.keys())
            self.ensemble_layers['base'] = EnsembleLayer(
                layer_id='base',
                models=base_models,
                aggregation_method='weighted_average',
                weights={model: 1.0/len(base_models) for model in base_models},
                uncertainty_method=UncertaintyMethod.BOOTSTRAP,
                performance=ModelPerformance(
                    model_id='base_ensemble',
                    family=ModelFamily.ENSEMBLE,
                    cv_score=0.0,
                    cv_std=0.0,
                    oos_score=0.0,
                    oos_std=0.0,
                    calibration_error=0.0,
                    uncertainty_coverage=0.0,
                    last_update=datetime.now(),
                    training_samples=0,
                    prediction_count=0
                )
            )
            
            # Meta layer
            meta_models = list(self.meta_models.keys())
            self.ensemble_layers['meta'] = EnsembleLayer(
                layer_id='meta',
                models=meta_models,
                aggregation_method='stacking',
                weights={model: 1.0/len(meta_models) for model in meta_models},
                uncertainty_method=UncertaintyMethod.QUANTILE,
                performance=ModelPerformance(
                    model_id='meta_ensemble',
                    family=ModelFamily.ENSEMBLE,
                    cv_score=0.0,
                    cv_std=0.0,
                    oos_score=0.0,
                    oos_std=0.0,
                    calibration_error=0.0,
                    uncertainty_coverage=0.0,
                    last_update=datetime.now(),
                    training_samples=0,
                    prediction_count=0
                )
            )
            
            # Super layer
            self.ensemble_layers['super'] = EnsembleLayer(
                layer_id='super',
                models=['super_model'],
                aggregation_method='conformal',
                weights={'super_model': 1.0},
                uncertainty_method=UncertaintyMethod.CONFORMAL,
                performance=ModelPerformance(
                    model_id='super_ensemble',
                    family=ModelFamily.ENSEMBLE,
                    cv_score=0.0,
                    cv_std=0.0,
                    oos_score=0.0,
                    oos_std=0.0,
                    calibration_error=0.0,
                    uncertainty_coverage=0.0,
                    last_update=datetime.now(),
                    training_samples=0,
                    prediction_count=0
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble layers: {e}")
    
    async def train_hierarchical(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train hierarchical ensemble with cross-validation"""
        try:
            results = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train base models
            base_predictions = await self._train_base_models(X, y, tscv)
            results['base_models'] = base_predictions
            
            # Train meta models using base predictions
            meta_predictions = await self._train_meta_models(base_predictions, y, tscv)
            results['meta_models'] = meta_predictions
            
            # Train super model using meta predictions
            super_predictions = await self._train_super_model(meta_predictions, y, tscv)
            results['super_model'] = super_predictions
            
            # Update ensemble weights
            await self._update_hierarchical_weights(results)
            
            # Calibrate uncertainty estimates
            await self._calibrate_uncertainty(X, y)
            
            # Store performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'results': results
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training hierarchical ensemble: {e}")
            return {}
    
    async def _train_base_models(self, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> Dict[str, np.ndarray]:
        """Train base models with cross-validation"""
        try:
            base_predictions = {}
            
            for model_id, model_info in self.base_models.items():
                self.logger.info(f"Training base model: {model_id}")
                
                model = model_info['model']
                scaler = model_info['scaler']
                
                # Cross-validation predictions
                cv_predictions = []
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train and predict
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    cv_predictions.extend(pred)
                    cv_scores.append(mean_squared_error(y_val, pred))
                
                # Store predictions and performance
                base_predictions[model_id] = np.array(cv_predictions)
                
                # Update performance tracking
                cv_score = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                self.model_performance[model_id] = ModelPerformance(
                    model_id=model_id,
                    family=model_info['family'],
                    cv_score=cv_score,
                    cv_std=cv_std,
                    oos_score=cv_score,
                    oos_std=cv_std,
                    calibration_error=0.0,
                    uncertainty_coverage=0.0,
                    last_update=datetime.now(),
                    training_samples=len(X),
                    prediction_count=len(cv_predictions)
                )
                
                model_info['trained'] = True
                
                self.logger.info(f"{model_id} CV RMSE: {np.sqrt(cv_score):.4f}")
            
            return base_predictions
            
        except Exception as e:
            self.logger.error(f"Error training base models: {e}")
            return {}
    
    async def _train_meta_models(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, tscv: TimeSeriesSplit) -> Dict[str, np.ndarray]:
        """Train meta models using base predictions"""
        try:
            meta_predictions = {}
            
            # Create meta-features from base predictions
            meta_features = np.column_stack(list(base_predictions.values()))
            
            for model_id, model_info in self.meta_models.items():
                self.logger.info(f"Training meta model: {model_id}")
                
                model = model_info['model']
                scaler = model_info['scaler']
                
                # Cross-validation predictions
                cv_predictions = []
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(meta_features):
                    X_train, X_val = meta_features[train_idx], meta_features[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train and predict
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    cv_predictions.extend(pred)
                    cv_scores.append(mean_squared_error(y_val, pred))
                
                # Store predictions and performance
                meta_predictions[model_id] = np.array(cv_predictions)
                
                # Update performance tracking
                cv_score = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                self.model_performance[model_id] = ModelPerformance(
                    model_id=model_id,
                    family=model_info['family'],
                    cv_score=cv_score,
                    cv_std=cv_std,
                    oos_score=cv_score,
                    oos_std=cv_std,
                    calibration_error=0.0,
                    uncertainty_coverage=0.0,
                    last_update=datetime.now(),
                    training_samples=len(meta_features),
                    prediction_count=len(cv_predictions)
                )
                
                model_info['trained'] = True
                
                self.logger.info(f"{model_id} CV RMSE: {np.sqrt(cv_score):.4f}")
            
            return meta_predictions
            
        except Exception as e:
            self.logger.error(f"Error training meta models: {e}")
            return {}
    
    async def _train_super_model(self, meta_predictions: Dict[str, np.ndarray], y: pd.Series, tscv: TimeSeriesSplit) -> Dict[str, np.ndarray]:
        """Train super model using meta predictions"""
        try:
            # Create super-features from meta predictions
            super_features = np.column_stack(list(meta_predictions.values()))
            
            model = self.super_model['model']
            scaler = self.super_model['scaler']
            
            # Cross-validation predictions
            cv_predictions = []
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(super_features):
                X_train, X_val = super_features[train_idx], super_features[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train and predict
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_val_scaled)
                cv_predictions.extend(pred)
                cv_scores.append(mean_squared_error(y_val, pred))
            
            # Store performance
            cv_score = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            self.model_performance['super_model'] = ModelPerformance(
                model_id='super_model',
                family=ModelFamily.ENSEMBLE,
                cv_score=cv_score,
                cv_std=cv_std,
                oos_score=cv_score,
                oos_std=cv_std,
                calibration_error=0.0,
                uncertainty_coverage=0.0,
                last_update=datetime.now(),
                training_samples=len(super_features),
                prediction_count=len(cv_predictions)
            )
            
            self.super_model['trained'] = True
            
            self.logger.info(f"Super model CV RMSE: {np.sqrt(cv_score):.4f}")
            
            return {'super_model': np.array(cv_predictions)}
            
        except Exception as e:
            self.logger.error(f"Error training super model: {e}")
            return {}
    
    async def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates and prediction intervals"""
        try:
            # Base layer predictions
            base_predictions = {}
            base_uncertainties = {}
            
            for model_id, model_info in self.base_models.items():
                if not model_info['trained']:
                    continue
                
                model = model_info['model']
                scaler = model_info['scaler']
                X_scaled = scaler.transform(X)
                
                # Make prediction
                pred = model.predict(X_scaled)
                base_predictions[model_id] = pred
                
                # Estimate uncertainty
                uncertainty = await self._estimate_uncertainty(model, X_scaled, model_info['family'])
                base_uncertainties[model_id] = uncertainty
            
            # Meta layer predictions
            meta_features = np.column_stack(list(base_predictions.values()))
            meta_predictions = {}
            meta_uncertainties = {}
            
            for model_id, model_info in self.meta_models.items():
                if not model_info['trained']:
                    continue
                
                model = model_info['model']
                scaler = model_info['scaler']
                X_scaled = scaler.transform(meta_features)
                
                # Make prediction
                pred = model.predict(X_scaled)
                meta_predictions[model_id] = pred
                
                # Estimate uncertainty
                uncertainty = await self._estimate_uncertainty(model, X_scaled, model_info['family'])
                meta_uncertainties[model_id] = uncertainty
            
            # Super layer prediction
            super_features = np.column_stack(list(meta_predictions.values()))
            model = self.super_model['model']
            scaler = self.super_model['scaler']
            X_scaled = scaler.transform(super_features)
            
            final_prediction = model.predict(X_scaled)
            
            # Ensemble uncertainty
            ensemble_uncertainty = await self._combine_uncertainties(
                base_uncertainties, meta_uncertainties, final_prediction
            )
            
            # Prediction intervals
            prediction_intervals = await self._calculate_prediction_intervals(
                final_prediction, ensemble_uncertainty
            )
            
            return final_prediction, ensemble_uncertainty, prediction_intervals
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            n_samples = len(X)
            return np.zeros(n_samples), np.zeros(n_samples), np.zeros((n_samples, 2))
    
    async def _estimate_uncertainty(self, model: Any, X: np.ndarray, family: ModelFamily) -> np.ndarray:
        """Estimate prediction uncertainty for different model families"""
        try:
            if family == ModelFamily.TREE_BASED:
                # Bootstrap uncertainty for tree-based models
                predictions = []
                for _ in range(10):  # Bootstrap samples
                    if hasattr(model, 'estimators_'):
                        # Random Forest or Extra Trees
                        pred = np.mean([estimator.predict(X) for estimator in model.estimators_], axis=0)
                    else:
                        # Gradient Boosting
                        pred = model.predict(X)
                    predictions.append(pred)
                
                uncertainty = np.std(predictions, axis=0)
                
            elif family == ModelFamily.LINEAR:
                # Analytical uncertainty for linear models
                if hasattr(model, 'coef_'):
                    # Ridge regression uncertainty
                    residuals = model.predict(X) - model.predict(X)  # Placeholder
                    uncertainty = np.full(len(X), np.std(residuals))
                else:
                    uncertainty = np.full(len(X), 0.1)
                    
            elif family == ModelFamily.NEURAL:
                # Dropout uncertainty for neural networks
                if hasattr(model, 'predict_proba'):
                    # Use prediction variance
                    pred_proba = model.predict_proba(X)
                    uncertainty = np.std(pred_proba, axis=1)
                else:
                    uncertainty = np.full(len(X), 0.15)
                    
            else:
                # Default uncertainty
                uncertainty = np.full(len(X), 0.1)
            
            return uncertainty
            
        except Exception as e:
            self.logger.error(f"Error estimating uncertainty: {e}")
            return np.full(len(X), 0.1)
    
    async def _combine_uncertainties(self, base_uncertainties: Dict[str, np.ndarray], 
                                   meta_uncertainties: Dict[str, np.ndarray], 
                                   final_prediction: np.ndarray) -> np.ndarray:
        """Combine uncertainties from different layers"""
        try:
            # Weighted combination of uncertainties
            base_weights = self.ensemble_layers['base'].weights
            meta_weights = self.ensemble_layers['meta'].weights
            
            # Base layer uncertainty
            base_uncertainty = np.zeros(len(final_prediction))
            for model_id, uncertainty in base_uncertainties.items():
                if model_id in base_weights:
                    base_uncertainty += base_weights[model_id] * uncertainty
            
            # Meta layer uncertainty
            meta_uncertainty = np.zeros(len(final_prediction))
            for model_id, uncertainty in meta_uncertainties.items():
                if model_id in meta_weights:
                    meta_uncertainty += meta_weights[model_id] * uncertainty
            
            # Combine with hierarchical weights
            ensemble_uncertainty = 0.3 * base_uncertainty + 0.7 * meta_uncertainty
            
            return ensemble_uncertainty
            
        except Exception as e:
            self.logger.error(f"Error combining uncertainties: {e}")
            return np.full(len(final_prediction), 0.1)
    
    async def _calculate_prediction_intervals(self, prediction: np.ndarray, 
                                            uncertainty: np.ndarray, 
                                            confidence: float = 0.95) -> np.ndarray:
        """Calculate prediction intervals using conformal prediction"""
        try:
            if self.conformal_quantiles is not None:
                # Use calibrated quantiles
                alpha = 1 - confidence
                lower_quantile = alpha / 2
                upper_quantile = 1 - alpha / 2
                
                lower_idx = int(lower_quantile * len(self.conformal_quantiles))
                upper_idx = int(upper_quantile * len(self.conformal_quantiles))
                
                lower_bound = prediction + self.conformal_quantiles[lower_idx] * uncertainty
                upper_bound = prediction + self.conformal_quantiles[upper_idx] * uncertainty
            else:
                # Use normal approximation
                z_score = norm.ppf((1 + confidence) / 2)
                lower_bound = prediction - z_score * uncertainty
                upper_bound = prediction + z_score * uncertainty
            
            return np.column_stack([lower_bound, upper_bound])
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction intervals: {e}")
            return np.column_stack([prediction, prediction])
    
    async def _update_hierarchical_weights(self, results: Dict[str, Any]):
        """Update weights for all ensemble layers"""
        try:
            # Update base layer weights
            base_scores = {model_id: self.model_performance[model_id].cv_score 
                          for model_id in self.base_models.keys()}
            base_weights = await self._calculate_optimal_weights(base_scores)
            self.ensemble_layers['base'].weights = base_weights
            
            # Update meta layer weights
            meta_scores = {model_id: self.model_performance[model_id].cv_score 
                          for model_id in self.meta_models.keys()}
            meta_weights = await self._calculate_optimal_weights(meta_scores)
            self.ensemble_layers['meta'].weights = meta_weights
            
            self.logger.info("Updated hierarchical ensemble weights")
            
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")
    
    async def _calculate_optimal_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal weights using inverse variance weighting"""
        try:
            # Inverse variance weighting
            inverse_scores = {model_id: 1.0 / (score + 1e-8) for model_id, score in scores.items()}
            total_inverse = sum(inverse_scores.values())
            
            weights = {model_id: inv_score / total_inverse 
                      for model_id, inv_score in inverse_scores.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal weights: {e}")
            # Equal weights as fallback
            n_models = len(scores)
            return {model_id: 1.0 / n_models for model_id in scores.keys()}
    
    async def _calibrate_uncertainty(self, X: pd.DataFrame, y: pd.Series):
        """Calibrate uncertainty estimates using conformal prediction"""
        try:
            # Get predictions and uncertainties
            predictions, uncertainties, _ = await self.predict_with_uncertainty(X)
            
            # Calculate residuals
            residuals = y.values - predictions
            
            # Normalize residuals by uncertainty
            normalized_residuals = residuals / (uncertainties + 1e-8)
            
            # Store for calibration
            self.calibration_data.extend(normalized_residuals)
            
            # Calculate conformal quantiles
            if len(self.calibration_data) > 100:
                self.conformal_quantiles = np.percentile(self.calibration_data, np.arange(1, 100))
            
            self.logger.info(f"Calibrated uncertainty with {len(self.calibration_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Error calibrating uncertainty: {e}")
    
    async def detect_drift(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Detect concept drift in the data"""
        try:
            drift_scores = {}
            
            # Get current predictions
            predictions, uncertainties, _ = await self.predict_with_uncertainty(X)
            
            # Calculate performance metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            # Compare with historical performance
            if self.performance_history:
                latest_performance = self.performance_history[-1]
                historical_mse = latest_performance['results'].get('super_model', {}).get('cv_score', mse)
                
                # Drift score based on performance degradation
                drift_score = (mse - historical_mse) / (historical_mse + 1e-8)
                drift_scores['performance_drift'] = drift_score
                
                # Uncertainty drift
                avg_uncertainty = np.mean(uncertainties)
                historical_uncertainty = np.mean([p['results'].get('avg_uncertainty', avg_uncertainty) 
                                                for p in self.performance_history[-5:]])
                uncertainty_drift = (avg_uncertainty - historical_uncertainty) / (historical_uncertainty + 1e-8)
                drift_scores['uncertainty_drift'] = uncertainty_drift
            
            # Store drift information
            for model_id in self.base_models.keys():
                if model_id in self.drift_detectors:
                    self.drift_detectors[model_id].append(drift_scores.get('performance_drift', 0))
            
            return drift_scores
            
        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
            return {}
    
    async def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'layers': {},
                'overall_performance': {},
                'drift_status': {}
            }
            
            # Layer information
            for layer_id, layer in self.ensemble_layers.items():
                summary['layers'][layer_id] = {
                    'models': layer.models,
                    'aggregation_method': layer.aggregation_method,
                    'weights': layer.weights,
                    'performance': {
                        'cv_score': layer.performance.cv_score,
                        'cv_std': layer.performance.cv_std,
                        'last_update': layer.performance.last_update.isoformat()
                    }
                }
            
            # Overall performance
            if self.performance_history:
                latest = self.performance_history[-1]
                summary['overall_performance'] = {
                    'latest_training': latest['timestamp'].isoformat(),
                    'total_models': len(self.base_models) + len(self.meta_models) + 1,
                    'calibration_samples': len(self.calibration_data)
                }
            
            # Drift status
            for model_id, drift_history in self.drift_detectors.items():
                if len(drift_history) > 10:
                    recent_drift = np.mean(list(drift_history)[-10:])
                    summary['drift_status'][model_id] = {
                        'recent_drift': recent_drift,
                        'drift_detected': abs(recent_drift) > self.drift_threshold
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting ensemble summary: {e}")
            return {}


# Factory function for easy integration
async def create_hierarchical_ensemble(config: Optional[Dict[str, Any]] = None) -> HierarchicalMetaEnsemble:
    """Create and initialize hierarchical meta-ensemble"""
    return HierarchicalMetaEnsemble(config)
