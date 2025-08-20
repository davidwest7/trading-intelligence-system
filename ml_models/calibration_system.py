#!/usr/bin/env python3
"""
Advanced Calibration System
Full calibration pipeline with quantile heads for alpha & drawdown, regime-conditioned calibration cache
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CalibrationConfig:
    """Configuration for calibration system"""
    quantile_levels: List[float] = None
    regime_window: int = 252  # Trading days for regime detection
    calibration_window: int = 504  # 2 years of trading days
    min_samples_per_regime: int = 50
    temperature_bounds: Tuple[float, float] = (0.1, 10.0)
    cache_expiry_hours: int = 24
    
    def __post_init__(self):
        if self.quantile_levels is None:
            self.quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass 
class MarketRegime:
    """Market regime classification"""
    regime_id: str
    name: str
    volatility_percentile: float
    trend_strength: float
    correlation_level: float
    risk_on_factor: float
    characteristics: Dict[str, float]
    
    def __hash__(self):
        return hash(self.regime_id)


@dataclass
class CalibrationResult:
    """Result of calibration process"""
    regime: MarketRegime
    calibrated_predictions: np.ndarray
    quantile_predictions: Dict[float, np.ndarray]
    uncertainty_bounds: np.ndarray
    temperature: float
    calibration_score: float
    reliability_diagram: Dict[str, Any]
    timestamp: datetime


class RegimeDetector:
    """Detects market regimes for regime-conditioned calibration"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define regime characteristics
        self.regime_definitions = {
            'low_vol_bull': {
                'volatility_range': (0.0, 0.3),
                'trend_range': (0.3, 1.0),
                'correlation_range': (0.0, 0.5),
                'risk_on_range': (0.6, 1.0)
            },
            'high_vol_bull': {
                'volatility_range': (0.7, 1.0),
                'trend_range': (0.3, 1.0),
                'correlation_range': (0.5, 1.0),
                'risk_on_range': (0.4, 0.8)
            },
            'low_vol_bear': {
                'volatility_range': (0.0, 0.3),
                'trend_range': (-1.0, -0.3),
                'correlation_range': (0.0, 0.5),
                'risk_on_range': (0.0, 0.4)
            },
            'high_vol_bear': {
                'volatility_range': (0.7, 1.0),
                'trend_range': (-1.0, -0.3),
                'correlation_range': (0.7, 1.0),
                'risk_on_range': (0.0, 0.3)
            },
            'sideways_low_vol': {
                'volatility_range': (0.0, 0.4),
                'trend_range': (-0.3, 0.3),
                'correlation_range': (0.0, 0.6),
                'risk_on_range': (0.3, 0.7)
            },
            'sideways_high_vol': {
                'volatility_range': (0.6, 1.0),
                'trend_range': (-0.3, 0.3),
                'correlation_range': (0.4, 1.0),
                'risk_on_range': (0.2, 0.8)
            }
        }
    
    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Calculate regime features
            features = self._calculate_regime_features(market_data)
            
            # Score each regime
            regime_scores = {}
            for regime_name, regime_def in self.regime_definitions.items():
                score = self._score_regime_match(features, regime_def)
                regime_scores[regime_name] = score
            
            # Select best matching regime
            best_regime = max(regime_scores, key=regime_scores.get)
            best_score = regime_scores[best_regime]
            
            # Create regime object
            regime = MarketRegime(
                regime_id=f"{best_regime}_{datetime.now().strftime('%Y%m')}",
                name=best_regime,
                volatility_percentile=features['volatility_percentile'],
                trend_strength=features['trend_strength'],
                correlation_level=features['correlation_level'],
                risk_on_factor=features['risk_on_factor'],
                characteristics=features
            )
            
            self.logger.info(f"Detected regime: {best_regime} (score: {best_score:.3f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            # Return default regime
            return MarketRegime(
                regime_id="default",
                name="unknown",
                volatility_percentile=0.5,
                trend_strength=0.0,
                correlation_level=0.5,
                risk_on_factor=0.5,
                characteristics={}
            )
    
    def _calculate_regime_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for regime detection"""
        # Ensure we have required columns
        if 'close' not in market_data.columns:
            market_data['close'] = 100 + np.random.randn(len(market_data)).cumsum()
        
        returns = market_data['close'].pct_change().dropna()
        
        # Calculate rolling volatility (21-day)
        vol_21d = returns.rolling(21).std() * np.sqrt(252)
        current_vol = vol_21d.iloc[-1] if len(vol_21d) > 0 else 0.2
        vol_percentile = stats.percentileofscore(vol_21d.dropna(), current_vol) / 100
        
        # Calculate trend strength (21-day momentum)
        price_momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-21] - 1) if len(market_data) >= 21 else 0.0
        trend_strength = np.tanh(price_momentum * 10)  # Normalize to [-1, 1]
        
        # Calculate correlation (simplified - would use cross-asset in production)
        correlation_proxy = abs(returns.rolling(21).corr(returns.shift(1)).iloc[-1]) if len(returns) > 21 else 0.3
        correlation_level = min(correlation_proxy, 1.0) if not np.isnan(correlation_proxy) else 0.3
        
        # Calculate risk-on factor (based on momentum and volatility)
        risk_on_factor = (0.5 + trend_strength * 0.3 - (vol_percentile - 0.5) * 0.2)
        risk_on_factor = np.clip(risk_on_factor, 0.0, 1.0)
        
        return {
            'volatility_percentile': vol_percentile,
            'trend_strength': trend_strength,
            'correlation_level': correlation_level,
            'risk_on_factor': risk_on_factor,
            'current_volatility': current_vol,
            'momentum_21d': price_momentum
        }
    
    def _score_regime_match(self, features: Dict[str, float], regime_def: Dict[str, Tuple[float, float]]) -> float:
        """Score how well features match a regime definition"""
        score = 0.0
        weights = {
            'volatility_range': 0.3,
            'trend_range': 0.3,
            'correlation_range': 0.2,
            'risk_on_range': 0.2
        }
        
        for feature_name, (min_val, max_val) in regime_def.items():
            feature_key = feature_name.replace('_range', '')
            if feature_key in features:
                feature_val = features[feature_key]
                if min_val <= feature_val <= max_val:
                    score += weights[feature_name]
                else:
                    # Penalize distance from range
                    distance = min(abs(feature_val - min_val), abs(feature_val - max_val))
                    penalty = distance * weights[feature_name]
                    score -= penalty
        
        return max(score, 0.0)


class QuantileRegressor:
    """Multi-quantile regression for distributional forecasts"""
    
    def __init__(self, quantile_levels: List[float]):
        self.quantile_levels = quantile_levels
        self.models = {}
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit quantile regression models"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            for quantile in self.quantile_levels:
                # Use GradientBoostingRegressor with quantile loss
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=quantile,
                    n_estimators=100,
                    max_depth=3,
                    random_state=42
                )
                model.fit(X, y)
                self.models[quantile] = model
                
            self.logger.info(f"Fitted quantile models for {len(self.quantile_levels)} levels")
            
        except ImportError:
            # Fallback to linear quantile regression
            self.logger.warning("GradientBoostingRegressor not available, using linear approximation")
            for quantile in self.quantile_levels:
                # Simple linear approximation
                self.models[quantile] = LinearQuantileApproximation(quantile)
                self.models[quantile].fit(X, y)
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict quantiles"""
        predictions = {}
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X)
        return predictions


class LinearQuantileApproximation:
    """Simple linear approximation for quantile regression"""
    
    def __init__(self, quantile: float):
        self.quantile = quantile
        self.model = None
        self.quantile_adjustment = stats.norm.ppf(quantile)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model"""
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Calculate residual standard deviation
        y_pred = self.model.predict(X)
        residuals = y - y_pred
        self.residual_std = np.std(residuals)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantile"""
        y_pred = self.model.predict(X)
        return y_pred + self.quantile_adjustment * self.residual_std


class TemperatureScaling:
    """Temperature scaling for calibration"""
    
    def __init__(self):
        self.temperature = 1.0
        self.logger = logging.getLogger(__name__)
    
    def fit(self, predictions: np.ndarray, targets: np.ndarray, 
            bounds: Tuple[float, float] = (0.1, 10.0)) -> float:
        """Fit temperature parameter"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temp):
                scaled_predictions = predictions / temp
                # Use cross-entropy loss for binary classification
                # For regression, use negative log-likelihood
                if len(np.unique(targets)) == 2:
                    # Binary classification
                    probs = 1 / (1 + np.exp(-scaled_predictions))
                    probs = np.clip(probs, 1e-15, 1 - 1e-15)
                    loss = -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
                else:
                    # Regression - use MSE with temperature scaling
                    loss = np.mean((scaled_predictions - targets) ** 2)
                return loss
            
            result = minimize_scalar(temperature_loss, bounds=bounds, method='bounded')
            self.temperature = result.x
            
            self.logger.info(f"Fitted temperature: {self.temperature:.3f}")
            return self.temperature
            
        except Exception as e:
            self.logger.error(f"Error fitting temperature: {e}")
            self.temperature = 1.0
            return self.temperature
    
    def apply(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        return predictions / self.temperature


class CalibrationCache:
    """Cache for regime-conditioned calibration results"""
    
    def __init__(self, cache_expiry_hours: int = 24):
        self.cache = {}
        self.cache_expiry_hours = cache_expiry_hours
        self.logger = logging.getLogger(__name__)
    
    def get(self, regime: MarketRegime) -> Optional[CalibrationResult]:
        """Get cached calibration result"""
        cache_key = self._get_cache_key(regime)
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=self.cache_expiry_hours):
                self.logger.debug(f"Cache hit for regime: {regime.name}")
                return result
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
                self.logger.debug(f"Cache expired for regime: {regime.name}")
        
        return None
    
    def set(self, regime: MarketRegime, result: CalibrationResult) -> None:
        """Cache calibration result"""
        cache_key = self._get_cache_key(regime)
        self.cache[cache_key] = (result, datetime.now())
        self.logger.debug(f"Cached result for regime: {regime.name}")
    
    def _get_cache_key(self, regime: MarketRegime) -> str:
        """Generate cache key for regime"""
        # Use regime characteristics for more granular caching
        key_components = [
            regime.name,
            f"vol_{regime.volatility_percentile:.2f}",
            f"trend_{regime.trend_strength:.2f}",
            f"corr_{regime.correlation_level:.2f}"
        ]
        return "_".join(key_components)
    
    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        expired_keys = []
        current_time = datetime.now()
        
        for key, (result, timestamp) in self.cache.items():
            if current_time - timestamp >= timedelta(hours=self.cache_expiry_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        return len(expired_keys)


class AdvancedCalibrationSystem:
    """Advanced calibration system with regime conditioning and quantile heads"""
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.regime_detector = RegimeDetector(self.config)
        self.quantile_regressor = QuantileRegressor(self.config.quantile_levels)
        self.temperature_scaling = TemperatureScaling()
        self.calibration_cache = CalibrationCache(self.config.cache_expiry_hours)
        
        # Initialize isotonic regression for reliability calibration
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        
        self.logger.info("Initialized Advanced Calibration System")
    
    async def calibrate_predictions(self, predictions: np.ndarray, targets: np.ndarray,
                                  features: np.ndarray, market_data: pd.DataFrame) -> CalibrationResult:
        """Calibrate predictions with regime conditioning"""
        try:
            # Detect current market regime
            regime = self.regime_detector.detect_regime(market_data)
            
            # Check cache first
            cached_result = self.calibration_cache.get(regime)
            if cached_result is not None:
                self.logger.info(f"Using cached calibration for regime: {regime.name}")
                return cached_result
            
            # Perform calibration
            calibration_result = await self._perform_calibration(
                predictions, targets, features, regime
            )
            
            # Cache result
            self.calibration_cache.set(regime, calibration_result)
            
            return calibration_result
            
        except Exception as e:
            self.logger.error(f"Error in calibration: {e}")
            # Return uncalibrated predictions as fallback
            return CalibrationResult(
                regime=MarketRegime("error", "error", 0.5, 0.0, 0.5, 0.5, {}),
                calibrated_predictions=predictions,
                quantile_predictions={},
                uncertainty_bounds=np.zeros_like(predictions),
                temperature=1.0,
                calibration_score=0.0,
                reliability_diagram={},
                timestamp=datetime.now()
            )
    
    async def _perform_calibration(self, predictions: np.ndarray, targets: np.ndarray,
                                 features: np.ndarray, regime: MarketRegime) -> CalibrationResult:
        """Perform the actual calibration process"""
        
        # 1. Temperature scaling
        temperature = self.temperature_scaling.fit(predictions, targets, self.config.temperature_bounds)
        temp_scaled_predictions = self.temperature_scaling.apply(predictions)
        
        # 2. Isotonic regression for reliability calibration
        if len(np.unique(targets)) > 2:  # Regression case
            # For regression, use binned calibration
            calibrated_predictions = self._calibrate_regression(temp_scaled_predictions, targets)
        else:
            # Classification case
            self.isotonic_regressor.fit(temp_scaled_predictions, targets)
            calibrated_predictions = self.isotonic_regressor.predict(temp_scaled_predictions)
        
        # 3. Quantile regression for distributional forecasts
        quantile_predictions = {}
        if features is not None and len(features) > 0:
            try:
                self.quantile_regressor.fit(features, targets)
                quantile_predictions = self.quantile_regressor.predict(features)
            except Exception as e:
                self.logger.warning(f"Quantile regression failed: {e}")
                # Fallback to empirical quantiles
                quantile_predictions = self._empirical_quantiles(calibrated_predictions, targets)
        
        # 4. Calculate uncertainty bounds
        uncertainty_bounds = self._calculate_uncertainty_bounds(
            calibrated_predictions, quantile_predictions, regime
        )
        
        # 5. Calculate calibration score
        calibration_score = self._calculate_calibration_score(calibrated_predictions, targets)
        
        # 6. Create reliability diagram
        reliability_diagram = self._create_reliability_diagram(calibrated_predictions, targets)
        
        return CalibrationResult(
            regime=regime,
            calibrated_predictions=calibrated_predictions,
            quantile_predictions=quantile_predictions,
            uncertainty_bounds=uncertainty_bounds,
            temperature=temperature,
            calibration_score=calibration_score,
            reliability_diagram=reliability_diagram,
            timestamp=datetime.now()
        )
    
    def _calibrate_regression(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calibrate regression predictions using binned approach"""
        n_bins = min(10, len(predictions) // 20)  # Adaptive number of bins
        if n_bins < 2:
            return predictions
        
        # Bin predictions and targets
        bin_boundaries = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bin_boundaries[0] = -np.inf
        bin_boundaries[-1] = np.inf
        
        calibrated_predictions = predictions.copy()
        
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                # Replace predictions in this bin with mean target
                calibrated_predictions[mask] = np.mean(targets[mask])
        
        return calibrated_predictions
    
    def _empirical_quantiles(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[float, np.ndarray]:
        """Calculate empirical quantiles as fallback"""
        residuals = targets - predictions
        quantile_predictions = {}
        
        for quantile in self.config.quantile_levels:
            empirical_quantile = np.percentile(residuals, quantile * 100)
            quantile_predictions[quantile] = predictions + empirical_quantile
        
        return quantile_predictions
    
    def _calculate_uncertainty_bounds(self, predictions: np.ndarray, 
                                    quantile_predictions: Dict[float, np.ndarray],
                                    regime: MarketRegime) -> np.ndarray:
        """Calculate uncertainty bounds"""
        if 0.05 in quantile_predictions and 0.95 in quantile_predictions:
            # Use 90% prediction interval
            lower_bound = quantile_predictions[0.05]
            upper_bound = quantile_predictions[0.95]
            uncertainty = (upper_bound - lower_bound) / 2
        else:
            # Fallback to regime-based uncertainty
            base_uncertainty = np.std(predictions) if len(predictions) > 1 else 0.1
            regime_multiplier = 1.0 + regime.volatility_percentile  # Higher uncertainty in high vol regimes
            uncertainty = np.full_like(predictions, base_uncertainty * regime_multiplier)
        
        return uncertainty
    
    def _calculate_calibration_score(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate calibration score (lower is better)"""
        try:
            if len(np.unique(targets)) > 2:
                # Regression: use mean squared calibration error
                mse = np.mean((predictions - targets) ** 2)
                return float(mse)
            else:
                # Classification: use Brier score
                brier_score = np.mean((predictions - targets) ** 2)
                return float(brier_score)
        except Exception:
            return 1.0
    
    def _create_reliability_diagram(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Create reliability diagram data"""
        try:
            n_bins = min(10, len(predictions) // 10)
            if n_bins < 2:
                return {"error": "Insufficient data for reliability diagram"}
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            
            observed_frequencies = []
            predicted_frequencies = []
            bin_counts = []
            
            for i in range(n_bins):
                mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
                if i == n_bins - 1:  # Include upper boundary in last bin
                    mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
                
                if np.sum(mask) > 0:
                    observed_freq = np.mean(targets[mask])
                    predicted_freq = np.mean(predictions[mask])
                    observed_frequencies.append(observed_freq)
                    predicted_frequencies.append(predicted_freq)
                    bin_counts.append(np.sum(mask))
                else:
                    observed_frequencies.append(0.0)
                    predicted_frequencies.append(bin_centers[i])
                    bin_counts.append(0)
            
            return {
                "bin_centers": bin_centers.tolist(),
                "observed_frequencies": observed_frequencies,
                "predicted_frequencies": predicted_frequencies,
                "bin_counts": bin_counts,
                "n_samples": len(predictions)
            }
            
        except Exception as e:
            return {"error": f"Failed to create reliability diagram: {e}"}
    
    async def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration system"""
        return {
            "config": {
                "quantile_levels": self.config.quantile_levels,
                "regime_window": self.config.regime_window,
                "calibration_window": self.config.calibration_window,
                "cache_expiry_hours": self.config.cache_expiry_hours
            },
            "cache": {
                "cached_regimes": len(self.calibration_cache.cache),
                "expired_entries_cleared": self.calibration_cache.clear_expired()
            },
            "quantile_models": {
                "fitted_quantiles": list(self.quantile_regressor.models.keys()),
                "n_quantile_levels": len(self.config.quantile_levels)
            },
            "temperature_scaling": {
                "current_temperature": self.temperature_scaling.temperature
            }
        }


# Factory function
async def create_calibration_system(config: Optional[CalibrationConfig] = None) -> AdvancedCalibrationSystem:
    """Create and initialize calibration system"""
    return AdvancedCalibrationSystem(config)


# Example usage
async def main():
    """Example usage of calibration system"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample predictions and targets
    predictions = np.random.rand(n_samples)
    targets = (predictions + 0.1 * np.random.randn(n_samples) > 0.5).astype(float)
    features = np.random.randn(n_samples, 5)
    
    # Sample market data
    dates = pd.date_range(end=datetime.now(), periods=252, freq='1D')
    market_data = pd.DataFrame({
        'close': 100 + np.random.randn(252).cumsum(),
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Create calibration system
    calibration_system = await create_calibration_system()
    
    # Perform calibration
    result = await calibration_system.calibrate_predictions(
        predictions, targets, features, market_data
    )
    
    print(f"Calibration completed for regime: {result.regime.name}")
    print(f"Temperature: {result.temperature:.3f}")
    print(f"Calibration score: {result.calibration_score:.4f}")
    print(f"Quantile predictions: {len(result.quantile_predictions)} levels")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
