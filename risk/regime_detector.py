#!/usr/bin/env python3
"""
Bayesian Change-Point Detector for Market Regime Identification

Implements regime detection using Bayesian change-point analysis
with separate RL policies per regime and seamless transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import stats
from sklearn.mixture import GaussianMixture
import asyncio

from schemas.contracts import RegimeType
from common.observability.telemetry import log_event, trace_operation


logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """Market regime states"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"


@dataclass
class RegimeFeatures:
    """Market regime features"""
    volatility: float
    returns_mean: float
    returns_std: float
    volume_ratio: float
    spread_ratio: float
    momentum: float
    correlation: float
    skewness: float
    kurtosis: float
    timestamp: datetime
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array"""
        return np.array([
            self.volatility,
            self.returns_mean,
            self.returns_std,
            self.volume_ratio,
            self.spread_ratio,
            self.momentum,
            self.correlation,
            self.skewness,
            self.kurtosis
        ])


@dataclass
class RegimeTransition:
    """Regime transition event"""
    from_regime: RegimeState
    to_regime: RegimeState
    confidence: float
    timestamp: datetime
    features: RegimeFeatures
    transition_probability: float


class BayesianChangePointDetector:
    """
    Bayesian Change-Point Detector for Market Regime Identification
    
    Features:
    - Bayesian change-point analysis
    - Gaussian mixture model for regime classification
    - Transition probability estimation
    - Confidence scoring for regime changes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 100)  # Observations
        self.min_regime_duration = config.get('min_regime_duration', 50)  # Observations
        self.change_threshold = config.get('change_threshold', 0.8)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Regime models
        self.gmm = GaussianMixture(
            n_components=len(RegimeState),
            covariance_type='full',
            random_state=42
        )
        
        # State tracking
        self.current_regime = RegimeState.RISK_ON
        self.regime_history: List[RegimeState] = []
        self.feature_history: List[RegimeFeatures] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Performance tracking
        self.detection_count = 0
        self.false_positives = 0
        self.avg_confidence = 0.0
        
        # Regime-specific parameters
        self.regime_parameters = self._initialize_regime_parameters()
        
        logger.info("Bayesian Change-Point Detector initialized")
    
    def _initialize_regime_parameters(self) -> Dict[RegimeState, Dict[str, Any]]:
        """Initialize regime-specific parameters"""
        return {
            RegimeState.RISK_ON: {
                'volatility_range': (0.01, 0.03),
                'returns_range': (0.001, 0.005),
                'volume_range': (0.8, 1.2),
                'momentum_range': (0.001, 0.01)
            },
            RegimeState.RISK_OFF: {
                'volatility_range': (0.03, 0.06),
                'returns_range': (-0.005, -0.001),
                'volume_range': (1.2, 2.0),
                'momentum_range': (-0.01, -0.001)
            },
            RegimeState.HIGH_VOL: {
                'volatility_range': (0.04, 0.08),
                'returns_range': (-0.01, 0.01),
                'volume_range': (1.5, 3.0),
                'momentum_range': (-0.02, 0.02)
            },
            RegimeState.LOW_VOL: {
                'volatility_range': (0.005, 0.015),
                'returns_range': (-0.002, 0.002),
                'volume_range': (0.5, 0.8),
                'momentum_range': (-0.001, 0.001)
            },
            RegimeState.TRENDING: {
                'volatility_range': (0.02, 0.04),
                'returns_range': (0.003, 0.008),
                'volume_range': (0.9, 1.1),
                'momentum_range': (0.005, 0.015)
            },
            RegimeState.MEAN_REVERTING: {
                'volatility_range': (0.015, 0.035),
                'returns_range': (-0.003, 0.003),
                'volume_range': (0.8, 1.2),
                'momentum_range': (-0.005, 0.005)
            },
            RegimeState.LIQUID: {
                'volatility_range': (0.01, 0.025),
                'returns_range': (-0.002, 0.002),
                'volume_range': (1.0, 1.5),
                'momentum_range': (-0.003, 0.003)
            },
            RegimeState.ILLIQUID: {
                'volatility_range': (0.02, 0.05),
                'returns_range': (-0.005, 0.005),
                'volume_range': (0.3, 0.7),
                'momentum_range': (-0.008, 0.008)
            }
        }
    
    async def detect_regime_change(self, market_data: Dict[str, Any], 
                                 trace_id: str) -> Optional[RegimeTransition]:
        """
        Detect regime change using Bayesian change-point analysis
        
        Args:
            market_data: Market data including prices, volumes, spreads
            trace_id: Trace ID for observability
            
        Returns:
            Regime transition event if detected, None otherwise
        """
        async with trace_operation("regime_detection", trace_id=trace_id):
            try:
                # Extract regime features
                features = self._extract_regime_features(market_data)
                self.feature_history.append(features)
                
                # Maintain window size
                if len(self.feature_history) > self.window_size:
                    self.feature_history.pop(0)
                
                # Check for change point
                if len(self.feature_history) >= self.min_regime_duration:
                    change_detected, confidence = await self._detect_change_point(trace_id)
                    
                    if change_detected and confidence > self.confidence_threshold:
                        # Classify new regime
                        new_regime = await self._classify_regime(features, trace_id)
                        
                        if new_regime != self.current_regime:
                            # Create transition event
                            transition = RegimeTransition(
                                from_regime=self.current_regime,
                                to_regime=new_regime,
                                confidence=confidence,
                                timestamp=datetime.utcnow(),
                                features=features,
                                transition_probability=self._calculate_transition_probability(
                                    self.current_regime, new_regime
                                )
                            )
                            
                            # Update state
                            self.current_regime = new_regime
                            self.regime_history.append(new_regime)
                            self.transition_history.append(transition)
                            self.detection_count += 1
                            
                            # Log transition
                            await self._log_regime_transition(transition, trace_id)
                            
                            return transition
                
                # Update current regime history
                self.regime_history.append(self.current_regime)
                
                return None
                
            except Exception as e:
                logger.error(f"Regime detection failed: {e}", extra={'trace_id': trace_id})
                return None
    
    def _extract_regime_features(self, market_data: Dict[str, Any]) -> RegimeFeatures:
        """Extract regime features from market data"""
        try:
            # Extract price data
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            spreads = market_data.get('spreads', [])
            
            if len(prices) < 2:
                # Use default features if insufficient data
                return RegimeFeatures(
                    volatility=0.02,
                    returns_mean=0.001,
                    returns_std=0.015,
                    volume_ratio=1.0,
                    spread_ratio=1.0,
                    momentum=0.001,
                    correlation=0.5,
                    skewness=0.0,
                    kurtosis=3.0,
                    timestamp=datetime.utcnow()
                )
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Volatility (rolling standard deviation)
            volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else np.std(returns) * np.sqrt(252)
            
            # Returns statistics
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            
            # Volume ratio (current vs average)
            if volumes:
                volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
            else:
                volume_ratio = 1.0
            
            # Spread ratio (current vs average)
            if spreads:
                spread_ratio = spreads[-1] / np.mean(spreads[-20:]) if len(spreads) >= 20 else 1.0
            else:
                spread_ratio = 1.0
            
            # Momentum (price trend)
            if len(prices) >= 10:
                momentum = (prices[-1] - prices[-10]) / prices[-10]
            else:
                momentum = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0.0
            
            # Correlation (simplified - would use multiple assets in production)
            correlation = 0.5  # Placeholder
            
            # Higher moments
            skewness = stats.skew(returns) if len(returns) > 2 else 0.0
            kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 3.0
            
            return RegimeFeatures(
                volatility=volatility,
                returns_mean=returns_mean,
                returns_std=returns_std,
                volume_ratio=volume_ratio,
                spread_ratio=spread_ratio,
                momentum=momentum,
                correlation=correlation,
                skewness=skewness,
                kurtosis=kurtosis,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return RegimeFeatures(
                volatility=0.02,
                returns_mean=0.001,
                returns_std=0.015,
                volume_ratio=1.0,
                spread_ratio=1.0,
                momentum=0.001,
                correlation=0.5,
                skewness=0.0,
                kurtosis=3.0,
                timestamp=datetime.utcnow()
            )
    
    async def _detect_change_point(self, trace_id: str) -> Tuple[bool, float]:
        """
        Detect change point using Bayesian analysis
        
        Returns:
            (change_detected, confidence)
        """
        try:
            if len(self.feature_history) < self.min_regime_duration:
                return False, 0.0
            
            # Extract feature arrays
            feature_arrays = [f.to_array() for f in self.feature_history]
            features_matrix = np.array(feature_arrays)
            
            # Split into two windows
            mid_point = len(feature_arrays) // 2
            window1 = features_matrix[:mid_point]
            window2 = features_matrix[mid_point:]
            
            if len(window1) < 10 or len(window2) < 10:
                return False, 0.0
            
            # Calculate distribution difference using KL divergence
            mean1, cov1 = np.mean(window1, axis=0), np.cov(window1.T)
            mean2, cov2 = np.mean(window2, axis=0), np.cov(window2.T)
            
            # KL divergence (simplified)
            try:
                # Use Mahalanobis distance as proxy for distribution difference
                diff_mean = mean2 - mean1
                pooled_cov = (cov1 + cov2) / 2
                
                # Calculate Mahalanobis distance
                mahal_distance = np.sqrt(diff_mean.T @ np.linalg.inv(pooled_cov) @ diff_mean)
                
                # Convert to confidence score
                confidence = min(1.0, mahal_distance / 10.0)  # Normalize
                
                # Detect change if confidence exceeds threshold
                change_detected = confidence > self.change_threshold
                
                return change_detected, confidence
                
            except np.linalg.LinAlgError:
                # Fallback: simple mean difference
                mean_diff = np.linalg.norm(mean2 - mean1)
                confidence = min(1.0, mean_diff / 2.0)
                change_detected = confidence > self.change_threshold
                
                return change_detected, confidence
                
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return False, 0.0
    
    async def _classify_regime(self, features: RegimeFeatures, 
                             trace_id: str) -> RegimeState:
        """Classify regime using Gaussian mixture model"""
        try:
            # Feature array
            feature_array = features.to_array().reshape(1, -1)
            
            # Predict regime using GMM
            if hasattr(self.gmm, 'means_'):
                # Model is fitted, use it
                regime_idx = self.gmm.predict(feature_array)[0]
                regime_states = list(RegimeState)
                return regime_states[regime_idx]
            else:
                # Model not fitted, use rule-based classification
                return self._rule_based_classification(features)
                
        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return self.current_regime
    
    def _rule_based_classification(self, features: RegimeFeatures) -> RegimeState:
        """Rule-based regime classification"""
        # Volatility-based classification
        if features.volatility > 0.04:
            if features.returns_mean < -0.002:
                return RegimeState.RISK_OFF
            else:
                return RegimeState.HIGH_VOL
        elif features.volatility < 0.015:
            return RegimeState.LOW_VOL
        
        # Momentum-based classification
        if abs(features.momentum) > 0.005:
            if features.momentum > 0:
                return RegimeState.TRENDING
            else:
                return RegimeState.MEAN_REVERTING
        
        # Volume-based classification
        if features.volume_ratio < 0.7:
            return RegimeState.ILLIQUID
        elif features.volume_ratio > 1.3:
            return RegimeState.LIQUID
        
        # Default classification
        if features.returns_mean > 0.001:
            return RegimeState.RISK_ON
        else:
            return RegimeState.RISK_OFF
    
    def _calculate_transition_probability(self, from_regime: RegimeState, 
                                        to_regime: RegimeState) -> float:
        """Calculate transition probability between regimes"""
        # Transition probability matrix (simplified)
        transition_matrix = {
            RegimeState.RISK_ON: {
                RegimeState.RISK_OFF: 0.3,
                RegimeState.HIGH_VOL: 0.2,
                RegimeState.LOW_VOL: 0.1,
                RegimeState.TRENDING: 0.2,
                RegimeState.MEAN_REVERTING: 0.1,
                RegimeState.LIQUID: 0.05,
                RegimeState.ILLIQUID: 0.05
            },
            RegimeState.RISK_OFF: {
                RegimeState.RISK_ON: 0.4,
                RegimeState.HIGH_VOL: 0.3,
                RegimeState.LOW_VOL: 0.1,
                RegimeState.TRENDING: 0.05,
                RegimeState.MEAN_REVERTING: 0.1,
                RegimeState.LIQUID: 0.02,
                RegimeState.ILLIQUID: 0.03
            },
            # Add other regime transitions...
        }
        
        # Get transition probability
        if from_regime in transition_matrix and to_regime in transition_matrix[from_regime]:
            return transition_matrix[from_regime][to_regime]
        else:
            return 0.1  # Default probability
    
    async def _log_regime_transition(self, transition: RegimeTransition, trace_id: str):
        """Log regime transition event"""
        try:
            await log_event("regime_transition_detected", {
                "trace_id": trace_id,
                "from_regime": transition.from_regime.value,
                "to_regime": transition.to_regime.value,
                "confidence": transition.confidence,
                "transition_probability": transition.transition_probability,
                "timestamp": transition.timestamp.isoformat(),
                "features": {
                    "volatility": transition.features.volatility,
                    "returns_mean": transition.features.returns_mean,
                    "volume_ratio": transition.features.volume_ratio,
                    "momentum": transition.features.momentum
                }
            })
            
        except Exception as e:
            logger.error(f"Regime transition logging failed: {e}")
    
    def get_current_regime(self) -> RegimeState:
        """Get current regime"""
        return self.current_regime
    
    def get_regime_history(self) -> List[RegimeState]:
        """Get regime history"""
        return self.regime_history.copy()
    
    def get_transition_history(self) -> List[RegimeTransition]:
        """Get transition history"""
        return self.transition_history.copy()
    
    def get_regime_parameters(self, regime: RegimeState) -> Dict[str, Any]:
        """Get regime-specific parameters"""
        return self.regime_parameters.get(regime, {})
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "current_regime": self.current_regime.value,
            "detection_count": self.detection_count,
            "false_positives": self.false_positives,
            "avg_confidence": self.avg_confidence,
            "regime_history_length": len(self.regime_history),
            "transition_history_length": len(self.transition_history)
        }
    
    async def fit_model(self, historical_features: List[RegimeFeatures]):
        """Fit the Gaussian mixture model with historical data"""
        try:
            if len(historical_features) < 50:
                logger.warning("Insufficient historical data for model fitting")
                return
            
            # Extract feature arrays
            feature_arrays = [f.to_array() for f in historical_features]
            features_matrix = np.array(feature_arrays)
            
            # Fit GMM
            self.gmm.fit(features_matrix)
            
            logger.info("Gaussian mixture model fitted successfully")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")


class RegimeConditionalPolicy:
    """
    Regime-Conditional Policy Manager
    
    Manages separate RL policies per regime with seamless transitions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_detector = BayesianChangePointDetector(config.get('detector_config', {}))
        
        # Regime-specific policies
        self.regime_policies: Dict[RegimeState, Any] = {}
        self.policy_switching_enabled = config.get('policy_switching_enabled', True)
        self.exploration_freeze_duration = config.get('exploration_freeze_duration', 10)
        
        # Transition tracking
        self.last_transition_time: Optional[datetime] = None
        self.decisions_since_transition = 0
        
        logger.info("Regime-Conditional Policy Manager initialized")
    
    async def get_policy_for_regime(self, regime: RegimeState) -> Any:
        """Get or create policy for specific regime"""
        if regime not in self.regime_policies:
            # Create new policy for regime
            self.regime_policies[regime] = await self._create_regime_policy(regime)
        
        return self.regime_policies[regime]
    
    async def _create_regime_policy(self, regime: RegimeState) -> Any:
        """Create new policy for specific regime"""
        # Simplified policy creation
        # In production, would create proper RL policies
        policy_config = {
            'regime': regime,
            'exploration_rate': self._get_regime_exploration_rate(regime),
            'risk_aversion': self._get_regime_risk_aversion(regime),
            'action_scale': self._get_regime_action_scale(regime)
        }
        
        # Return policy configuration (simplified)
        return policy_config
    
    def _get_regime_exploration_rate(self, regime: RegimeState) -> float:
        """Get exploration rate for regime"""
        exploration_rates = {
            RegimeState.RISK_ON: 0.1,
            RegimeState.RISK_OFF: 0.05,  # Conservative
            RegimeState.HIGH_VOL: 0.15,  # More exploration
            RegimeState.LOW_VOL: 0.08,
            RegimeState.TRENDING: 0.12,
            RegimeState.MEAN_REVERTING: 0.1,
            RegimeState.LIQUID: 0.1,
            RegimeState.ILLIQUID: 0.05  # Conservative
        }
        return exploration_rates.get(regime, 0.1)
    
    def _get_regime_risk_aversion(self, regime: RegimeState) -> float:
        """Get risk aversion for regime"""
        risk_aversions = {
            RegimeState.RISK_ON: 1.5,
            RegimeState.RISK_OFF: 3.0,  # High risk aversion
            RegimeState.HIGH_VOL: 2.5,
            RegimeState.LOW_VOL: 1.0,
            RegimeState.TRENDING: 1.2,
            RegimeState.MEAN_REVERTING: 1.8,
            RegimeState.LIQUID: 1.5,
            RegimeState.ILLIQUID: 2.5  # High risk aversion
        }
        return risk_aversions.get(regime, 2.0)
    
    def _get_regime_action_scale(self, regime: RegimeState) -> float:
        """Get action scale for regime"""
        action_scales = {
            RegimeState.RISK_ON: 1.0,
            RegimeState.RISK_OFF: 0.5,  # Smaller actions
            RegimeState.HIGH_VOL: 0.7,
            RegimeState.LOW_VOL: 1.2,
            RegimeState.TRENDING: 1.1,
            RegimeState.MEAN_REVERTING: 0.8,
            RegimeState.LIQUID: 1.0,
            RegimeState.ILLIQUID: 0.6  # Smaller actions
        }
        return action_scales.get(regime, 1.0)
    
    def should_freeze_exploration(self) -> bool:
        """Check if exploration should be frozen after regime transition"""
        if not self.last_transition_time:
            return False
        
        time_since_transition = datetime.utcnow() - self.last_transition_time
        return (time_since_transition < timedelta(minutes=self.exploration_freeze_duration) and
                self.decisions_since_transition < 10)
    
    def record_decision(self):
        """Record a decision made since last transition"""
        self.decisions_since_transition += 1
    
    def get_current_policy_config(self) -> Dict[str, Any]:
        """Get current policy configuration"""
        current_regime = self.regime_detector.get_current_regime()
        policy = self.regime_policies.get(current_regime, {})
        
        # Apply transition adjustments
        if self.should_freeze_exploration():
            policy['exploration_rate'] = 0.0  # Freeze exploration
            policy['action_scale'] *= 0.5  # Reduce action scale
        
        return policy
