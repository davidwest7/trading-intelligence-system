"""
QR LightGBM Meta-Weighter with Isotonic Calibration

Advanced meta-learning system that blends agent signals with:
- Quantile regression for uncertainty quantification
- Isotonic calibration for better probability estimates
- Regime-conditional blending
- Uncertainty propagation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from schemas.contracts import Signal, Opportunity, RegimeType, HorizonType, DirectionType


logger = logging.getLogger(__name__)


@dataclass
class BlendingWeights:
    """Weights for blending agent signals"""
    agent_weights: Dict[str, float]
    confidence_weight: float
    uncertainty_weight: float
    regime_weights: Dict[str, float]
    timestamp: datetime


@dataclass
class CalibrationModel:
    """Calibration model for probability estimates"""
    isotonic_regressor: IsotonicRegression
    calibration_curve: np.ndarray
    calibration_error: float
    samples_used: int


class QRLightGBMMetaWeighter:
    """
    Quantile Regression LightGBM Meta-Weighter with Isotonic Calibration
    
    This system blends multiple agent signals into calibrated opportunities with:
    - Uncertainty quantification via quantile regression
    - Regime-conditional blending
    - Isotonic calibration for better probability estimates
    - Expected Calibration Error tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Model configuration
        self.quantiles = config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        self.n_estimators = config.get('n_estimators', 100)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.max_depth = config.get('max_depth', 6)
        self.min_child_samples = config.get('min_child_samples', 20)
        
        # Calibration configuration
        self.calibration_window = config.get('calibration_window', 1000)
        self.min_calibration_samples = config.get('min_calibration_samples', 100)
        self.recalibration_interval = config.get('recalibration_interval', 100)
        
        # Models for each quantile and regime
        self.models: Dict[str, Dict[str, lgb.LGBMRegressor]] = {}
        self.calibration_models: Dict[str, CalibrationModel] = {}
        self.blending_weights: Dict[str, BlendingWeights] = {}
        
        # Performance tracking
        self.calibration_errors: List[float] = []
        self.blending_history: List[Dict[str, Any]] = []
        self.predictions_made = 0
        
        # Training data buffer
        self.training_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = config.get('max_buffer_size', 10000)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize QR models for each regime and quantile"""
        regimes = [regime.value for regime in RegimeType]
        
        for regime in regimes:
            self.models[regime] = {}
            
            for quantile in self.quantiles:
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=quantile,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_child_samples=self.min_child_samples,
                    random_state=42,
                    verbose=-1
                )
                
                self.models[regime][f'q{quantile}'] = model
            
            # Initialize calibration model for regime
            self.calibration_models[regime] = CalibrationModel(
                isotonic_regressor=IsotonicRegression(out_of_bounds='clip'),
                calibration_curve=np.array([]),
                calibration_error=0.0,
                samples_used=0
            )
    
    async def blend_signals(self, signals: List[Signal], 
                          trace_id: Optional[str] = None) -> List[Opportunity]:
        """
        Blend multiple agent signals into calibrated opportunities
        
        Args:
            signals: List of signals from different agents
            trace_id: Optional trace ID for tracking
            
        Returns:
            List of blended opportunities with uncertainty quantification
        """
        if not signals:
            return []
        
        try:
            # Group signals by symbol
            signals_by_symbol = self._group_signals_by_symbol(signals)
            
            opportunities = []
            for symbol, symbol_signals in signals_by_symbol.items():
                opportunity = await self._blend_symbol_signals(
                    symbol, symbol_signals, trace_id
                )
                if opportunity:
                    opportunities.append(opportunity)
            
            self.predictions_made += len(opportunities)
            logger.info(f"Blended {len(signals)} signals into {len(opportunities)} opportunities")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error blending signals: {e}")
            return []
    
    def _group_signals_by_symbol(self, signals: List[Signal]) -> Dict[str, List[Signal]]:
        """Group signals by symbol"""
        signals_by_symbol = {}
        
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        return signals_by_symbol
    
    async def _blend_symbol_signals(self, symbol: str, signals: List[Signal],
                                  trace_id: Optional[str] = None) -> Optional[Opportunity]:
        """Blend signals for a single symbol"""
        if not signals:
            return None
        
        try:
            # Extract features from signals
            features = self._extract_features(signals)
            
            # Determine dominant regime
            regime = self._determine_dominant_regime(signals)
            
            # Get blending weights for regime
            weights = self._get_blending_weights(regime, signals)
            
            # Blend signals
            mu_blended, sigma_blended, confidence_blended = self._blend_with_weights(
                signals, weights
            )
            
            # Apply quantile regression if models are trained
            if self._models_trained(regime):
                mu_blended, sigma_blended = await self._apply_quantile_regression(
                    features, regime, mu_blended, sigma_blended
                )
            
            # Apply calibration
            confidence_blended = self._apply_calibration(
                confidence_blended, regime
            )
            
            # Calculate risk metrics
            var_95, cvar_95, sharpe_ratio = self._calculate_risk_metrics(
                mu_blended, sigma_blended
            )
            
            # Determine direction and horizon
            direction = DirectionType.LONG if mu_blended > 0 else DirectionType.SHORT
            horizon = self._determine_blended_horizon(signals)
            
            # Create opportunity
            opportunity = Opportunity(
                trace_id=trace_id or signals[0].trace_id,
                symbol=symbol,
                mu_blended=mu_blended,
                sigma_blended=sigma_blended,
                confidence_blended=confidence_blended,
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                agent_signals={s.agent_id: s for s in signals},
                agent_weights=weights.agent_weights,
                horizon=horizon,
                regime=regime,
                direction=direction,
                blender_version=self.config.get('version', '1.0.0')
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error blending signals for {symbol}: {e}")
            return None
    
    def _extract_features(self, signals: List[Signal]) -> np.ndarray:
        """Extract features from signals for ML models"""
        features = []
        
        # Agent-specific features
        agent_mus = []
        agent_sigmas = []
        agent_confidences = []
        
        for signal in signals:
            agent_mus.append(signal.mu)
            agent_sigmas.append(signal.sigma)
            agent_confidences.append(signal.confidence)
        
        # Statistical features
        features.extend([
            np.mean(agent_mus),
            np.std(agent_mus),
            np.median(agent_mus),
            np.mean(agent_sigmas),
            np.std(agent_sigmas),
            np.mean(agent_confidences),
            np.std(agent_confidences),
            len(signals),  # Number of agents
        ])
        
        # Cross-signal features
        if len(agent_mus) > 1:
            features.extend([
                np.corrcoef(agent_mus, agent_confidences)[0, 1] if len(agent_mus) > 1 else 0,
                max(agent_mus) - min(agent_mus),  # Range
                np.sum(np.array(agent_mus) > 0),  # Number of positive signals
            ])
        else:
            features.extend([0, 0, 1 if agent_mus[0] > 0 else 0])
        
        # Regime features
        regime_counts = {}
        for signal in signals:
            regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
        
        # Add regime diversity
        features.append(len(regime_counts))
        
        return np.array(features).reshape(1, -1)
    
    def _determine_dominant_regime(self, signals: List[Signal]) -> RegimeType:
        """Determine dominant regime from signals"""
        regime_counts = {}
        
        for signal in signals:
            regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + 1
        
        # Return most common regime
        return max(regime_counts.keys(), key=lambda k: regime_counts[k])
    
    def _get_blending_weights(self, regime: RegimeType, signals: List[Signal]) -> BlendingWeights:
        """Get blending weights for regime"""
        regime_key = regime.value
        
        if regime_key not in self.blending_weights:
            # Initialize default weights
            agent_weights = {}
            for signal in signals:
                agent_weights[signal.agent_id] = 1.0 / len(signals)
            
            self.blending_weights[regime_key] = BlendingWeights(
                agent_weights=agent_weights,
                confidence_weight=1.0,
                uncertainty_weight=1.0,
                regime_weights={regime_key: 1.0},
                timestamp=datetime.utcnow()
            )
        
        return self.blending_weights[regime_key]
    
    def _blend_with_weights(self, signals: List[Signal], 
                          weights: BlendingWeights) -> Tuple[float, float, float]:
        """Blend signals using weights"""
        if not signals:
            return 0.0, 0.0, 0.0
        
        # Weighted average of mu values
        weighted_mus = []
        weighted_sigmas = []
        weighted_confidences = []
        total_weight = 0.0
        
        for signal in signals:
            agent_weight = weights.agent_weights.get(signal.agent_id, 1.0 / len(signals))
            confidence_adjustment = signal.confidence * weights.confidence_weight
            uncertainty_adjustment = (1.0 / signal.sigma) * weights.uncertainty_weight
            
            final_weight = agent_weight * confidence_adjustment * uncertainty_adjustment
            
            weighted_mus.append(signal.mu * final_weight)
            weighted_sigmas.append(signal.sigma * final_weight)
            weighted_confidences.append(signal.confidence * final_weight)
            
            total_weight += final_weight
        
        if total_weight == 0:
            total_weight = 1.0
        
        # Calculate blended values
        mu_blended = sum(weighted_mus) / total_weight
        sigma_blended = np.sqrt(sum(np.array(weighted_sigmas) ** 2)) / total_weight
        confidence_blended = sum(weighted_confidences) / total_weight
        
        # Ensure values are in reasonable ranges
        mu_blended = np.clip(mu_blended, -0.5, 0.5)
        sigma_blended = np.clip(sigma_blended, 0.005, 0.20)
        confidence_blended = np.clip(confidence_blended, 0.0, 1.0)
        
        return mu_blended, sigma_blended, confidence_blended
    
    def _models_trained(self, regime: RegimeType) -> bool:
        """Check if models are trained for regime"""
        regime_key = regime.value
        
        if regime_key not in self.models:
            return False
        
        # Check if any model is trained
        for model in self.models[regime_key].values():
            if hasattr(model, 'booster_') and model.booster_ is not None:
                return True
        
        return False
    
    async def _apply_quantile_regression(self, features: np.ndarray, regime: RegimeType,
                                       mu_base: float, sigma_base: float) -> Tuple[float, float]:
        """Apply quantile regression to refine predictions"""
        try:
            regime_key = regime.value
            models = self.models[regime_key]
            
            # Predict quantiles
            quantile_predictions = {}
            for quantile in self.quantiles:
                model = models[f'q{quantile}']
                if hasattr(model, 'booster_') and model.booster_ is not None:
                    pred = model.predict(features)[0]
                    quantile_predictions[quantile] = pred
            
            if not quantile_predictions:
                return mu_base, sigma_base
            
            # Use median (0.5 quantile) as mu
            mu_refined = quantile_predictions.get(0.5, mu_base)
            
            # Calculate sigma from quantile spread
            if 0.25 in quantile_predictions and 0.75 in quantile_predictions:
                # IQR-based sigma estimation
                iqr = quantile_predictions[0.75] - quantile_predictions[0.25]
                sigma_refined = iqr / 1.35  # Approximate conversion to std
            else:
                sigma_refined = sigma_base
            
            # Ensure reasonable values
            mu_refined = np.clip(mu_refined, -0.5, 0.5)
            sigma_refined = np.clip(sigma_refined, 0.005, 0.20)
            
            return mu_refined, sigma_refined
            
        except Exception as e:
            logger.error(f"Error applying quantile regression: {e}")
            return mu_base, sigma_base
    
    def _apply_calibration(self, confidence: float, regime: RegimeType) -> float:
        """Apply isotonic calibration to confidence"""
        try:
            regime_key = regime.value
            calibration_model = self.calibration_models.get(regime_key)
            
            if (calibration_model and 
                calibration_model.isotonic_regressor and
                calibration_model.samples_used >= self.min_calibration_samples):
                
                calibrated_confidence = calibration_model.isotonic_regressor.predict([confidence])[0]
                return np.clip(calibrated_confidence, 0.0, 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return confidence
    
    def _calculate_risk_metrics(self, mu: float, sigma: float) -> Tuple[float, float, float]:
        """Calculate risk metrics"""
        # 95% VaR (assuming normal distribution)
        var_95 = -(mu - 1.645 * sigma)
        
        # 95% CVaR (Expected Shortfall)
        cvar_95 = -(mu - 2.063 * sigma)  # Approximate for normal distribution
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mu / sigma if sigma > 0 else 0.0
        
        return var_95, cvar_95, sharpe_ratio
    
    def _determine_blended_horizon(self, signals: List[Signal]) -> HorizonType:
        """Determine blended horizon from signals"""
        horizon_counts = {}
        
        for signal in signals:
            horizon_counts[signal.horizon] = horizon_counts.get(signal.horizon, 0) + 1
        
        # Return most common horizon
        return max(horizon_counts.keys(), key=lambda k: horizon_counts[k])
    
    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train QR models with historical data"""
        if len(training_data) < self.min_calibration_samples:
            logger.warning(f"Insufficient training data: {len(training_data)}")
            return
        
        try:
            # Prepare training data by regime
            regime_data = {}
            for data_point in training_data:
                regime = data_point['regime']
                if regime not in regime_data:
                    regime_data[regime] = []
                regime_data[regime].append(data_point)
            
            # Train models for each regime
            for regime, data in regime_data.items():
                if len(data) >= self.min_calibration_samples:
                    await self._train_regime_models(regime, data)
            
            logger.info(f"Trained models for {len(regime_data)} regimes")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    async def _train_regime_models(self, regime: str, training_data: List[Dict[str, Any]]):
        """Train models for a specific regime"""
        try:
            # Prepare features and targets
            X = []
            y = []
            
            for data_point in training_data:
                features = data_point['features']
                target = data_point['target']
                
                X.append(features)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train quantile models
            for quantile in self.quantiles:
                model = self.models[regime][f'q{quantile}']
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                logger.debug(f"Trained {regime} q{quantile} model, MSE: {mse:.4f}")
            
            # Train calibration model
            await self._train_calibration_model(regime, X_test, y_test)
            
        except Exception as e:
            logger.error(f"Error training regime models for {regime}: {e}")
    
    async def _train_calibration_model(self, regime: str, X_test: np.ndarray, y_test: np.ndarray):
        """Train isotonic calibration model"""
        try:
            # Get model predictions (using median quantile)
            model = self.models[regime]['q0.5']
            if hasattr(model, 'booster_') and model.booster_ is not None:
                y_pred_proba = model.predict(X_test)
                
                # Convert to binary classification for calibration
                y_binary = (y_test > 0).astype(int)
                
                # Train isotonic regression
                calibration_model = self.calibration_models[regime]
                calibration_model.isotonic_regressor.fit(y_pred_proba, y_binary)
                calibration_model.samples_used = len(y_test)
                
                # Calculate calibration error
                y_calibrated = calibration_model.isotonic_regressor.predict(y_pred_proba)
                calibration_error = np.mean(np.abs(y_calibrated - y_binary))
                calibration_model.calibration_error = calibration_error
                
                self.calibration_errors.append(calibration_error)
                
                logger.debug(f"Calibration model trained for {regime}, error: {calibration_error:.4f}")
                
        except Exception as e:
            logger.error(f"Error training calibration model for {regime}: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'predictions_made': self.predictions_made,
            'calibration_errors': self.calibration_errors[-10:],  # Last 10
            'avg_calibration_error': np.mean(self.calibration_errors) if self.calibration_errors else 0.0,
            'models_trained': sum(1 for regime in self.models.values() 
                                for model in regime.values() 
                                if hasattr(model, 'booster_') and model.booster_ is not None),
            'training_buffer_size': len(self.training_buffer),
            'regimes_calibrated': len([cm for cm in self.calibration_models.values() 
                                     if cm.samples_used >= self.min_calibration_samples]),
        }
