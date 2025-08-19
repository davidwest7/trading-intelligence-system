"""
Hidden Markov Model (HMM) Regime Detection for Market Flow Analysis

Detects market regimes using HMM to identify:
- Trending vs Ranging markets
- High vs Low volatility periods  
- Breakout vs Reversal patterns
- Regime transition probabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None

try:
    from scipy import stats
except ImportError:
    stats = None
import warnings
warnings.filterwarnings('ignore')

from .models import RegimeState, RegimeType


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection
    
    Uses a multi-state HMM to identify different market regimes based on:
    - Price returns
    - Volatility measures  
    - Volume patterns
    - Momentum indicators
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.regime_names = [
            RegimeType.TRENDING_UP,
            RegimeType.TRENDING_DOWN, 
            RegimeType.RANGING,
            RegimeType.VOLATILE
        ]
        
        # Model parameters (will be fitted)
        self.transition_matrix = None
        self.emission_params = None
        self.regime_means = None
        self.regime_covariances = None
        self.initial_probabilities = None
        
        # Fitted model
        self.gmm_model = None
        self.is_fitted = False
        
        # Regime characteristics
        self.regime_characteristics = {}
    
    def fit(self, price_data: pd.DataFrame) -> None:
        """
        Fit the HMM to historical price data
        
        Args:
            price_data: DataFrame with OHLCV data
        """
        # Extract features for regime detection
        features = self._extract_features(price_data)
        
        # Fit Gaussian Mixture Model (simplified HMM approach)
        if GaussianMixture is not None:
            self.gmm_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
        else:
            print("Warning: sklearn not available. Using simple regime detection.")
            self.gmm_model = None
        
        if self.gmm_model is not None:
            self.gmm_model.fit(features)
        
        # Extract regime characteristics
        self._analyze_regime_characteristics(features)
        
        # Estimate transition probabilities
        regime_sequence = self.gmm_model.predict(features)
        self.transition_matrix = self._estimate_transition_matrix(regime_sequence)
        
        self.is_fitted = True
    
    def predict_regime(self, recent_data: pd.DataFrame) -> RegimeState:
        """
        Predict current market regime
        
        Args:
            recent_data: Recent OHLCV data
            
        Returns:
            Current regime state with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features from recent data
        features = self._extract_features(recent_data)
        
        if len(features) == 0:
            return self._default_regime_state()
        
        # Get regime probabilities
        regime_probs = self.gmm_model.predict_proba(features)
        latest_probs = regime_probs[-1]  # Most recent probabilities
        
        # Determine most likely regime
        most_likely_regime_idx = np.argmax(latest_probs)
        regime_type = self.regime_names[most_likely_regime_idx]
        
        # Calculate regime characteristics
        regime_prob = latest_probs[most_likely_regime_idx]
        
        # Estimate persistence (simplified)
        persistence = self._estimate_persistence(most_likely_regime_idx)
        
        # Get regime parameters
        volatility = np.sqrt(self.gmm_model.covariances_[most_likely_regime_idx, 1, 1])
        mean_return = self.gmm_model.means_[most_likely_regime_idx, 0]
        
        # Get transition probabilities
        transition_probs = {
            regime_name.value: self.transition_matrix[most_likely_regime_idx, i]
            for i, regime_name in enumerate(self.regime_names)
        }
        
        return RegimeState(
            regime_type=regime_type,
            probability=regime_prob,
            persistence=persistence,
            volatility=volatility,
            mean_return=mean_return,
            transition_probabilities=transition_probs
        )
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        if len(data) < 5:
            return np.array([]).reshape(0, 4)
        
        features = []
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 3:
            return np.array([]).reshape(0, 4)
        
        # Rolling window features
        window = min(20, len(returns))
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_volume = data['volume'].iloc[i-window:i]
            window_prices = data['close'].iloc[i-window:i]
            
            # Feature 1: Mean return
            mean_return = window_returns.mean()
            
            # Feature 2: Volatility (rolling std)
            volatility = window_returns.std()
            
            # Feature 3: Volume trend
            volume_trend = np.polyfit(range(len(window_volume)), window_volume, 1)[0]
            volume_trend = volume_trend / window_volume.mean()  # Normalize
            
            # Feature 4: Price momentum
            momentum = (window_prices.iloc[-1] - window_prices.iloc[0]) / window_prices.iloc[0]
            
            features.append([mean_return, volatility, volume_trend, momentum])
        
        return np.array(features)
    
    def _analyze_regime_characteristics(self, features: np.ndarray) -> None:
        """Analyze characteristics of each regime"""
        regime_labels = self.gmm_model.predict(features)
        
        for regime_idx in range(self.n_regimes):
            regime_features = features[regime_labels == regime_idx]
            
            if len(regime_features) > 0:
                characteristics = {
                    'mean_return': np.mean(regime_features[:, 0]),
                    'volatility': np.mean(regime_features[:, 1]),
                    'volume_trend': np.mean(regime_features[:, 2]),
                    'momentum': np.mean(regime_features[:, 3]),
                    'frequency': len(regime_features) / len(features)
                }
                
                # Classify regime type based on characteristics
                regime_type = self._classify_regime_type(characteristics)
                
                self.regime_characteristics[regime_idx] = {
                    'type': regime_type,
                    'characteristics': characteristics
                }
    
    def _classify_regime_type(self, characteristics: Dict[str, float]) -> RegimeType:
        """Classify regime type based on characteristics"""
        mean_return = characteristics['mean_return']
        volatility = characteristics['volatility']
        momentum = characteristics['momentum']
        
        # High volatility regime
        if volatility > 0.03:  # 3% daily volatility threshold
            return RegimeType.VOLATILE
        
        # Trending regimes
        if abs(momentum) > 0.05:  # 5% momentum threshold
            if mean_return > 0.001:  # Positive mean return
                return RegimeType.TRENDING_UP
            else:
                return RegimeType.TRENDING_DOWN
        
        # Low volatility/ranging regime
        if volatility < 0.015:  # 1.5% volatility threshold
            return RegimeType.RANGING
        
        # Default to ranging
        return RegimeType.RANGING
    
    def _estimate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Estimate transition probability matrix"""
        n_states = self.n_regimes
        transition_counts = np.zeros((n_states, n_states))
        
        for i in range(len(regime_sequence) - 1):
            current_state = regime_sequence[i]
            next_state = regime_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        transition_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions observed, assume equal probability
                transition_matrix[i, :] = 1.0 / n_states
        
        return transition_matrix
    
    def _estimate_persistence(self, regime_idx: int) -> float:
        """Estimate expected persistence of a regime"""
        if self.transition_matrix is None:
            return 5.0  # Default 5 periods
        
        # Expected persistence = 1 / (1 - self-transition probability)
        self_transition_prob = self.transition_matrix[regime_idx, regime_idx]
        
        if self_transition_prob >= 0.99:
            return 100.0  # Very persistent
        elif self_transition_prob <= 0.01:
            return 1.0   # Not persistent
        else:
            return 1.0 / (1.0 - self_transition_prob)
    
    def _default_regime_state(self) -> RegimeState:
        """Return default regime state when prediction fails"""
        return RegimeState(
            regime_type=RegimeType.RANGING,
            probability=0.5,
            persistence=5.0,
            volatility=0.02,
            mean_return=0.0,
            transition_probabilities={
                regime.value: 0.25 for regime in self.regime_names
            }
        )
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of all regime characteristics"""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        summary = {}
        for regime_idx, regime_data in self.regime_characteristics.items():
            regime_name = self.regime_names[regime_idx].value
            summary[regime_name] = {
                "characteristics": regime_data['characteristics'],
                "transition_probabilities": {
                    self.regime_names[j].value: self.transition_matrix[regime_idx, j]
                    for j in range(self.n_regimes)
                }
            }
        
        return summary


class VolatilityRegimeDetector:
    """
    Specialized detector for volatility regimes
    
    Uses GARCH-like modeling to identify:
    - High volatility periods (stress/crisis)
    - Low volatility periods (calm markets)
    - Volatility clustering
    """
    
    def __init__(self, window: int = 30):
        self.window = window
        self.volatility_threshold_high = 0.03  # 3% daily vol
        self.volatility_threshold_low = 0.01   # 1% daily vol
    
    def detect_volatility_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current volatility regime"""
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < self.window:
            return self._default_volatility_regime()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.window).std()
        current_vol = rolling_vol.iloc[-1]
        
        # Calculate percentiles for context
        vol_percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol)
        
        # Determine regime
        if current_vol > self.volatility_threshold_high:
            regime = RegimeType.VOLATILE
        elif current_vol < self.volatility_threshold_low:
            regime = RegimeType.LOW_VOLATILITY
        else:
            regime = RegimeType.RANGING
        
        # Volatility persistence (autocorrelation)
        vol_changes = rolling_vol.pct_change().dropna()
        persistence = vol_changes.autocorr(lag=1) if len(vol_changes) > 1 else 0.0
        
        return {
            "regime": regime.value,
            "current_volatility": current_vol,
            "volatility_percentile": vol_percentile,
            "persistence": persistence,
            "vol_trend": "increasing" if rolling_vol.iloc[-1] > rolling_vol.iloc[-5] else "decreasing"
        }
    
    def _default_volatility_regime(self) -> Dict[str, Any]:
        """Default volatility regime when insufficient data"""
        return {
            "regime": RegimeType.RANGING.value,
            "current_volatility": 0.02,
            "volatility_percentile": 50.0,
            "persistence": 0.0,
            "vol_trend": "stable"
        }


class BreakoutReversalDetector:
    """
    Detect breakout vs reversal regimes
    
    Identifies when markets are in:
    - Breakout regimes (trending continuation)
    - Reversal regimes (trend changes)
    - Consolidation regimes (sideways)
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect breakout/reversal regime"""
        if len(price_data) < self.lookback:
            return self._default_breakout_regime()
        
        closes = price_data['close']
        highs = price_data['high']
        lows = price_data['low']
        volumes = price_data['volume']
        
        # Recent price action
        recent_high = highs.rolling(self.lookback).max().iloc[-1]
        recent_low = lows.rolling(self.lookback).min().iloc[-1]
        current_price = closes.iloc[-1]
        
        # Breakout detection
        breakout_threshold = 0.02  # 2% threshold
        
        if current_price > recent_high * (1 + breakout_threshold):
            regime = RegimeType.BREAKOUT
            direction = "upward"
        elif current_price < recent_low * (1 - breakout_threshold):
            regime = RegimeType.BREAKOUT
            direction = "downward"
        else:
            # Check for reversal patterns
            returns = closes.pct_change()
            recent_returns = returns.tail(5).sum()
            
            if abs(recent_returns) > 0.05:  # 5% move
                regime = RegimeType.REVERSAL
                direction = "up" if recent_returns > 0 else "down"
            else:
                regime = RegimeType.RANGING
                direction = "sideways"
        
        # Volume confirmation
        avg_volume = volumes.rolling(self.lookback).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        volume_confirmation = current_volume > avg_volume * 1.2
        
        return {
            "regime": regime.value,
            "direction": direction,
            "volume_confirmation": volume_confirmation,
            "strength": min(1.0, abs(recent_returns) / 0.1),  # Normalize to 0-1
            "current_price_vs_range": {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "current": current_price,
                "position_in_range": (current_price - recent_low) / (recent_high - recent_low)
            }
        }
    
    def _default_breakout_regime(self) -> Dict[str, Any]:
        """Default regime when insufficient data"""
        return {
            "regime": RegimeType.RANGING.value,
            "direction": "sideways",
            "volume_confirmation": False,
            "strength": 0.0,
            "current_price_vs_range": {
                "recent_high": 0.0,
                "recent_low": 0.0,
                "current": 0.0,
                "position_in_range": 0.5
            }
        }
