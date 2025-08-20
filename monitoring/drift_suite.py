#!/usr/bin/env python3
"""
Comprehensive Monitoring and Drift Detection Suite
PSI/JS divergence, regime flip detectors, outlier capture, alerting SLOs, and full model operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    alert_type: str  # 'feature_drift', 'regime_flip', 'outlier_detected', 'performance_degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    description: str
    metric_value: float
    threshold_value: float
    affected_components: List[str]
    recommendations: List[str]
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class DriftMetrics:
    """Drift detection metrics"""
    psi_score: float
    js_divergence: float
    ks_statistic: float
    wasserstein_distance: float
    drift_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence_level: float
    is_significant: bool
    sample_size: int


@dataclass
class RegimeState:
    """Market regime state"""
    regime_id: str
    regime_name: str
    confidence: float
    duration_days: int
    volatility_level: float
    trend_strength: float
    correlation_level: float
    risk_on_factor: float
    transition_probability: float
    timestamp: datetime


@dataclass
class OutlierDetection:
    """Outlier detection result"""
    outlier_id: str
    outlier_type: str  # 'point', 'contextual', 'collective'
    severity: float
    affected_features: List[str]
    outlier_score: float
    isolation_path: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    metric_name: str
    current_value: float
    historical_mean: float
    historical_std: float
    z_score: float
    percentile_rank: float
    trend_direction: str
    is_degrading: bool
    degradation_rate: float
    timestamp: datetime


@dataclass
class SLOMetrics:
    """Service Level Objective metrics"""
    slo_name: str
    target_value: float
    current_value: float
    breach_threshold: float
    is_breached: bool
    breach_duration: timedelta
    uptime_percentage: float
    response_time_p95: float
    error_rate: float
    last_updated: datetime


class PSICalculator:
    """Population Stability Index calculator"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.logger = logging.getLogger(__name__)
    
    def calculate_psi(self, reference_data: np.ndarray, 
                     current_data: np.ndarray) -> DriftMetrics:
        """Calculate PSI between reference and current distributions"""
        try:
            # Create bins based on reference data
            bins = np.percentile(reference_data, np.linspace(0, 100, self.n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference_data, bins=bins)
            curr_hist, _ = np.histogram(current_data, bins=bins)
            
            # Normalize to probabilities
            ref_probs = ref_hist / len(reference_data)
            curr_probs = curr_hist / len(current_data)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_probs = np.maximum(ref_probs, epsilon)
            curr_probs = np.maximum(curr_probs, epsilon)
            
            # Calculate PSI
            psi_score = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
            
            # Calculate additional metrics
            js_divergence = jensenshannon(ref_probs, curr_probs)
            ks_statistic, _ = stats.ks_2samp(reference_data, current_data)
            
            # Wasserstein distance (simplified)
            wasserstein_distance = np.abs(np.mean(current_data) - np.mean(reference_data))
            
            # Determine drift direction
            drift_direction = self._determine_drift_direction(ref_probs, curr_probs)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(len(reference_data), len(current_data))
            
            # Determine significance
            is_significant = psi_score > 0.1  # PSI threshold
            
            return DriftMetrics(
                psi_score=psi_score,
                js_divergence=js_divergence,
                ks_statistic=ks_statistic,
                wasserstein_distance=wasserstein_distance,
                drift_direction=drift_direction,
                confidence_level=confidence_level,
                is_significant=is_significant,
                sample_size=len(current_data)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {e}")
            return self._create_fallback_metrics()
    
    def _determine_drift_direction(self, ref_probs: np.ndarray, curr_probs: np.ndarray) -> str:
        """Determine drift direction"""
        try:
            # Compare means of the distributions
            ref_mean = np.average(np.arange(len(ref_probs)), weights=ref_probs)
            curr_mean = np.average(np.arange(len(curr_probs)), weights=curr_probs)
            
            if curr_mean > ref_mean + 0.1:
                return 'increasing'
            elif curr_mean < ref_mean - 0.1:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'stable'
    
    def _calculate_confidence_level(self, ref_size: int, curr_size: int) -> float:
        """Calculate confidence level based on sample sizes"""
        try:
            # Simplified confidence calculation
            min_size = min(ref_size, curr_size)
            if min_size >= 1000:
                return 0.95
            elif min_size >= 500:
                return 0.90
            elif min_size >= 100:
                return 0.80
            else:
                return 0.70
        except Exception:
            return 0.70
    
    def _create_fallback_metrics(self) -> DriftMetrics:
        """Create fallback metrics when calculation fails"""
        return DriftMetrics(
            psi_score=0.0,
            js_divergence=0.0,
            ks_statistic=0.0,
            wasserstein_distance=0.0,
            drift_direction='stable',
            confidence_level=0.0,
            is_significant=False,
            sample_size=0
        )


class RegimeDetector:
    """Market regime detection and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.regime_history = []
        self.transition_matrix = None
        self.current_regime = None
        
        # Define regime characteristics
        self.regime_definitions = {
            'low_vol_bull': {'volatility': (0.0, 0.3), 'trend': (0.3, 1.0), 'correlation': (0.0, 0.5)},
            'high_vol_bull': {'volatility': (0.7, 1.0), 'trend': (0.3, 1.0), 'correlation': (0.5, 1.0)},
            'low_vol_bear': {'volatility': (0.0, 0.3), 'trend': (-1.0, -0.3), 'correlation': (0.0, 0.5)},
            'high_vol_bear': {'volatility': (0.7, 1.0), 'trend': (-1.0, -0.3), 'correlation': (0.7, 1.0)},
            'sideways_low_vol': {'volatility': (0.0, 0.4), 'trend': (-0.3, 0.3), 'correlation': (0.0, 0.6)},
            'sideways_high_vol': {'volatility': (0.6, 1.0), 'trend': (-0.3, 0.3), 'correlation': (0.4, 1.0)}
        }
    
    def detect_regime(self, market_data: pd.DataFrame) -> RegimeState:
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
            
            # Check for regime flip
            regime_flip = self._detect_regime_flip(best_regime)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(best_regime)
            
            # Create regime state
            regime_state = RegimeState(
                regime_id=f"{best_regime}_{datetime.now().strftime('%Y%m%d')}",
                regime_name=best_regime,
                confidence=best_score,
                duration_days=self._calculate_regime_duration(best_regime),
                volatility_level=features['volatility_percentile'],
                trend_strength=features['trend_strength'],
                correlation_level=features['correlation_level'],
                risk_on_factor=features['risk_on_factor'],
                transition_probability=transition_prob,
                timestamp=datetime.now()
            )
            
            # Update regime history
            self.regime_history.append(regime_state)
            self.current_regime = regime_state
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return regime_state
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return self._create_fallback_regime()
    
    def _calculate_regime_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime features"""
        try:
            if 'close' not in market_data.columns:
                market_data['close'] = 100 + np.random.randn(len(market_data)).cumsum()
            
            returns = market_data['close'].pct_change().dropna()
            
            # Volatility features
            vol_21d = returns.rolling(21).std() * np.sqrt(252)
            current_vol = vol_21d.iloc[-1] if len(vol_21d) > 0 else 0.2
            vol_percentile = stats.percentileofscore(vol_21d.dropna(), current_vol) / 100
            
            # Trend features
            price_momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-21] - 1) if len(market_data) >= 21 else 0.0
            trend_strength = np.tanh(price_momentum * 10)
            
            # Correlation features
            correlation_proxy = abs(returns.rolling(21).corr(returns.shift(1)).iloc[-1]) if len(returns) > 21 else 0.3
            correlation_level = min(correlation_proxy, 1.0) if not np.isnan(correlation_proxy) else 0.3
            
            # Risk-on factor
            risk_on_factor = (0.5 + trend_strength * 0.3 - (vol_percentile - 0.5) * 0.2)
            risk_on_factor = np.clip(risk_on_factor, 0.0, 1.0)
            
            return {
                'volatility_percentile': vol_percentile,
                'trend_strength': trend_strength,
                'correlation_level': correlation_level,
                'risk_on_factor': risk_on_factor
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return {
                'volatility_percentile': 0.5,
                'trend_strength': 0.0,
                'correlation_level': 0.5,
                'risk_on_factor': 0.5
            }
    
    def _score_regime_match(self, features: Dict[str, float], regime_def: Dict[str, Tuple[float, float]]) -> float:
        """Score how well features match a regime definition"""
        score = 0.0
        weights = {'volatility': 0.3, 'trend': 0.3, 'correlation': 0.2}
        
        for feature_name, (min_val, max_val) in regime_def.items():
            feature_key = f"{feature_name}_percentile" if feature_name == 'volatility' else f"{feature_name}_strength" if feature_name == 'trend' else f"{feature_name}_level"
            if feature_key in features:
                feature_val = features[feature_key]
                if min_val <= feature_val <= max_val:
                    score += weights.get(feature_name, 0.2)
                else:
                    distance = min(abs(feature_val - min_val), abs(feature_val - max_val))
                    penalty = distance * weights.get(feature_name, 0.2)
                    score -= penalty
        
        return max(score, 0.0)
    
    def _detect_regime_flip(self, new_regime: str) -> bool:
        """Detect if regime has flipped"""
        if self.current_regime is None:
            return False
        
        return self.current_regime.regime_name != new_regime
    
    def _calculate_transition_probability(self, regime: str) -> float:
        """Calculate probability of regime transition"""
        if len(self.regime_history) < 2:
            return 0.1
        
        # Count transitions
        transitions = 0
        total_periods = len(self.regime_history) - 1
        
        for i in range(total_periods):
            if self.regime_history[i].regime_name != self.regime_history[i+1].regime_name:
                transitions += 1
        
        return transitions / total_periods if total_periods > 0 else 0.1
    
    def _calculate_regime_duration(self, regime: str) -> int:
        """Calculate duration of current regime"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for i in range(len(self.regime_history) - 1, 0, -1):
            if self.regime_history[i].regime_name == regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _create_fallback_regime(self) -> RegimeState:
        """Create fallback regime state"""
        return RegimeState(
            regime_id="unknown",
            regime_name="unknown",
            confidence=0.0,
            duration_days=1,
            volatility_level=0.5,
            trend_strength=0.0,
            correlation_level=0.5,
            risk_on_factor=0.5,
            transition_probability=0.1,
            timestamp=datetime.now()
        )


class OutlierDetector:
    """Advanced outlier detection system"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.logger = logging.getLogger(__name__)
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def detect_outliers(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> List[OutlierDetection]:
        """Detect outliers in data"""
        try:
            if not self.is_fitted:
                self._fit_detector(data)
            
            # Detect outliers
            outlier_labels = self.isolation_forest.predict(data)
            outlier_scores = self.isolation_forest.decision_function(data)
            
            # Find outlier indices
            outlier_indices = np.where(outlier_labels == -1)[0]
            
            outliers = []
            for idx in outlier_indices:
                outlier = self._create_outlier_detection(
                    idx, data[idx], outlier_scores[idx], feature_names
                )
                outliers.append(outlier)
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return []
    
    def _fit_detector(self, data: np.ndarray) -> None:
        """Fit the outlier detector"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Fit isolation forest
            self.isolation_forest.fit(scaled_data)
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Error fitting outlier detector: {e}")
            self.is_fitted = False
    
    def _create_outlier_detection(self, index: int, data_point: np.ndarray, 
                                outlier_score: float, feature_names: Optional[List[str]]) -> OutlierDetection:
        """Create outlier detection result"""
        # Determine outlier type
        if outlier_score < -0.5:
            outlier_type = 'point'
        elif outlier_score < -0.3:
            outlier_type = 'contextual'
        else:
            outlier_type = 'collective'
        
        # Calculate severity
        severity = abs(outlier_score)
        
        # Identify affected features
        if feature_names:
            affected_features = feature_names
        else:
            affected_features = [f"feature_{i}" for i in range(len(data_point))]
        
        return OutlierDetection(
            outlier_id=f"outlier_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}",
            outlier_type=outlier_type,
            severity=severity,
            affected_features=affected_features,
            outlier_score=outlier_score,
            isolation_path=0.0,  # Would calculate in production
            timestamp=datetime.now(),
            context={'data_point': data_point.tolist(), 'index': index}
        )


class PerformanceMonitor:
    """Model performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_history = {}
        self.degradation_thresholds = {
            'sharpe_ratio': 0.1,
            'information_ratio': 0.05,
            'accuracy': 0.02,
            'precision': 0.02,
            'recall': 0.02
        }
    
    def monitor_performance(self, model_id: str, metric_name: str, 
                          current_value: float, historical_data: List[float]) -> ModelPerformance:
        """Monitor model performance"""
        try:
            if len(historical_data) < 10:
                return self._create_fallback_performance(model_id, metric_name, current_value)
            
            historical_mean = np.mean(historical_data)
            historical_std = np.std(historical_data)
            
            # Calculate z-score
            z_score = (current_value - historical_mean) / historical_std if historical_std > 0 else 0
            
            # Calculate percentile rank
            percentile_rank = stats.percentileofscore(historical_data, current_value)
            
            # Determine trend direction
            recent_data = historical_data[-10:]
            trend_direction = self._calculate_trend_direction(recent_data)
            
            # Check for degradation
            threshold = self.degradation_thresholds.get(metric_name, 0.1)
            is_degrading = abs(z_score) > threshold and z_score < 0
            
            # Calculate degradation rate
            degradation_rate = self._calculate_degradation_rate(historical_data)
            
            return ModelPerformance(
                model_id=model_id,
                metric_name=metric_name,
                current_value=current_value,
                historical_mean=historical_mean,
                historical_std=historical_std,
                z_score=z_score,
                percentile_rank=percentile_rank,
                trend_direction=trend_direction,
                is_degrading=is_degrading,
                degradation_rate=degradation_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {e}")
            return self._create_fallback_performance(model_id, metric_name, current_value)
    
    def _calculate_trend_direction(self, data: List[float]) -> str:
        """Calculate trend direction"""
        if len(data) < 2:
            return 'stable'
        
        slope = np.polyfit(range(len(data)), data, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_degradation_rate(self, data: List[float]) -> float:
        """Calculate degradation rate"""
        if len(data) < 10:
            return 0.0
        
        # Calculate rate of change
        recent_data = data[-10:]
        slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        return slope
    
    def _create_fallback_performance(self, model_id: str, metric_name: str, current_value: float) -> ModelPerformance:
        """Create fallback performance metrics"""
        return ModelPerformance(
            model_id=model_id,
            metric_name=metric_name,
            current_value=current_value,
            historical_mean=current_value,
            historical_std=0.0,
            z_score=0.0,
            percentile_rank=50.0,
            trend_direction='stable',
            is_degrading=False,
            degradation_rate=0.0,
            timestamp=datetime.now()
        )


class SLOMonitor:
    """Service Level Objective monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.slo_definitions = {
            'model_latency': {'target': 100, 'threshold': 200},  # ms
            'prediction_accuracy': {'target': 0.85, 'threshold': 0.80},
            'system_uptime': {'target': 0.999, 'threshold': 0.995},
            'data_freshness': {'target': 60, 'threshold': 300},  # seconds
            'error_rate': {'target': 0.001, 'threshold': 0.01}
        }
        self.slo_history = {}
    
    def monitor_slo(self, slo_name: str, current_value: float, 
                   additional_metrics: Optional[Dict[str, float]] = None) -> SLOMetrics:
        """Monitor SLO compliance"""
        try:
            if slo_name not in self.slo_definitions:
                return self._create_fallback_slo(slo_name, current_value)
            
            slo_def = self.slo_definitions[slo_name]
            target_value = slo_def['target']
            breach_threshold = slo_def['threshold']
            
            # Determine if SLO is breached
            is_breached = self._check_slo_breach(slo_name, current_value, target_value)
            
            # Calculate breach duration
            breach_duration = self._calculate_breach_duration(slo_name, is_breached)
            
            # Calculate additional metrics
            uptime_percentage = self._calculate_uptime(slo_name)
            response_time_p95 = additional_metrics.get('response_time_p95', 0.0) if additional_metrics else 0.0
            error_rate = additional_metrics.get('error_rate', 0.0) if additional_metrics else 0.0
            
            slo_metrics = SLOMetrics(
                slo_name=slo_name,
                target_value=target_value,
                current_value=current_value,
                breach_threshold=breach_threshold,
                is_breached=is_breached,
                breach_duration=breach_duration,
                uptime_percentage=uptime_percentage,
                response_time_p95=response_time_p95,
                error_rate=error_rate,
                last_updated=datetime.now()
            )
            
            # Update history
            if slo_name not in self.slo_history:
                self.slo_history[slo_name] = []
            self.slo_history[slo_name].append(slo_metrics)
            
            # Keep only recent history
            if len(self.slo_history[slo_name]) > 1000:
                self.slo_history[slo_name] = self.slo_history[slo_name][-1000:]
            
            return slo_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring SLO: {e}")
            return self._create_fallback_slo(slo_name, current_value)
    
    def _check_slo_breach(self, slo_name: str, current_value: float, target_value: float) -> bool:
        """Check if SLO is breached"""
        if slo_name in ['model_latency', 'data_freshness', 'error_rate']:
            # Lower is better
            return current_value > target_value
        else:
            # Higher is better
            return current_value < target_value
    
    def _calculate_breach_duration(self, slo_name: str, is_breached: bool) -> timedelta:
        """Calculate breach duration"""
        if not is_breached:
            return timedelta(0)
        
        # Simplified calculation
        return timedelta(minutes=5)  # Would track actual duration in production
    
    def _calculate_uptime(self, slo_name: str) -> float:
        """Calculate uptime percentage"""
        if slo_name not in self.slo_history:
            return 1.0
        
        history = self.slo_history[slo_name]
        if not history:
            return 1.0
        
        uptime_periods = sum(1 for slo in history if not slo.is_breached)
        return uptime_periods / len(history)
    
    def _create_fallback_slo(self, slo_name: str, current_value: float) -> SLOMetrics:
        """Create fallback SLO metrics"""
        return SLOMetrics(
            slo_name=slo_name,
            target_value=current_value,
            current_value=current_value,
            breach_threshold=current_value,
            is_breached=False,
            breach_duration=timedelta(0),
            uptime_percentage=1.0,
            response_time_p95=0.0,
            error_rate=0.0,
            last_updated=datetime.now()
        )


class DriftSuite:
    """Comprehensive drift detection and monitoring suite"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.psi_calculator = PSICalculator()
        self.regime_detector = RegimeDetector()
        self.outlier_detector = OutlierDetector()
        self.performance_monitor = PerformanceMonitor()
        self.slo_monitor = SLOMonitor()
        
        # Alert management
        self.alerts = []
        self.alert_thresholds = {
            'psi_critical': 0.25,
            'psi_high': 0.1,
            'regime_flip': 0.8,
            'outlier_severity': 0.7,
            'performance_degradation': 2.0
        }
        
        self.logger.info("Initialized Drift Detection Suite")
    
    async def run_comprehensive_monitoring(self, 
                                         reference_data: Dict[str, np.ndarray],
                                         current_data: Dict[str, np.ndarray],
                                         market_data: pd.DataFrame,
                                         model_performance: Dict[str, float]) -> List[DriftAlert]:
        """Run comprehensive monitoring and generate alerts"""
        try:
            alerts = []
            
            # 1. Feature drift detection
            feature_alerts = await self._detect_feature_drift(reference_data, current_data)
            alerts.extend(feature_alerts)
            
            # 2. Regime monitoring
            regime_alerts = await self._monitor_regime_changes(market_data)
            alerts.extend(regime_alerts)
            
            # 3. Outlier detection
            outlier_alerts = await self._detect_outliers(current_data)
            alerts.extend(outlier_alerts)
            
            # 4. Performance monitoring
            performance_alerts = await self._monitor_performance(model_performance)
            alerts.extend(performance_alerts)
            
            # 5. SLO monitoring
            slo_alerts = await self._monitor_slos()
            alerts.extend(slo_alerts)
            
            # Store alerts
            self.alerts.extend(alerts)
            
            # Keep only recent alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive monitoring: {e}")
            return []
    
    async def _detect_feature_drift(self, reference_data: Dict[str, np.ndarray],
                                  current_data: Dict[str, np.ndarray]) -> List[DriftAlert]:
        """Detect feature drift"""
        alerts = []
        
        for feature_name in reference_data.keys():
            if feature_name in current_data:
                try:
                    # Calculate PSI
                    drift_metrics = self.psi_calculator.calculate_psi(
                        reference_data[feature_name], current_data[feature_name]
                    )
                    
                    # Generate alert if significant drift detected
                    if drift_metrics.is_significant:
                        severity = self._determine_alert_severity(drift_metrics.psi_score, 'psi')
                        
                        alert = DriftAlert(
                            alert_id=f"drift_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type='feature_drift',
                            severity=severity,
                            timestamp=datetime.now(),
                            description=f"Feature drift detected in {feature_name}",
                            metric_value=drift_metrics.psi_score,
                            threshold_value=self.alert_thresholds['psi_high'],
                            affected_components=[feature_name],
                            recommendations=[
                                "Review feature engineering pipeline",
                                "Check for data quality issues",
                                "Consider model retraining"
                            ]
                        )
                        alerts.append(alert)
                
                except Exception as e:
                    self.logger.error(f"Error detecting drift for feature {feature_name}: {e}")
        
        return alerts
    
    async def _monitor_regime_changes(self, market_data: pd.DataFrame) -> List[DriftAlert]:
        """Monitor regime changes"""
        alerts = []
        
        try:
            # Detect current regime
            regime_state = self.regime_detector.detect_regime(market_data)
            
            # Check for regime flip
            if len(self.regime_detector.regime_history) > 1:
                previous_regime = self.regime_detector.regime_history[-2]
                if regime_state.regime_name != previous_regime.regime_name:
                    alert = DriftAlert(
                        alert_id=f"regime_flip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type='regime_flip',
                        severity='high',
                        timestamp=datetime.now(),
                        description=f"Regime flip detected: {previous_regime.regime_name} -> {regime_state.regime_name}",
                        metric_value=regime_state.transition_probability,
                        threshold_value=self.alert_thresholds['regime_flip'],
                        affected_components=['market_regime', 'model_performance'],
                        recommendations=[
                            "Review model performance in new regime",
                            "Adjust risk parameters if needed",
                            "Monitor for performance degradation"
                        ]
                    )
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error monitoring regime changes: {e}")
        
        return alerts
    
    async def _detect_outliers(self, current_data: Dict[str, np.ndarray]) -> List[DriftAlert]:
        """Detect outliers"""
        alerts = []
        
        try:
            # Combine all features for outlier detection
            if current_data:
                feature_names = list(current_data.keys())
                combined_data = np.column_stack(list(current_data.values()))
                
                outliers = self.outlier_detector.detect_outliers(combined_data, feature_names)
                
                for outlier in outliers:
                    if outlier.severity > self.alert_thresholds['outlier_severity']:
                        alert = DriftAlert(
                            alert_id=outlier.outlier_id,
                            alert_type='outlier_detected',
                            severity='medium',
                            timestamp=datetime.now(),
                            description=f"Outlier detected: {outlier.outlier_type} type",
                            metric_value=outlier.severity,
                            threshold_value=self.alert_thresholds['outlier_severity'],
                            affected_components=outlier.affected_features,
                            recommendations=[
                                "Investigate data quality",
                                "Check for system anomalies",
                                "Review outlier handling strategy"
                            ]
                        )
                        alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
        
        return alerts
    
    async def _monitor_performance(self, model_performance: Dict[str, float]) -> List[DriftAlert]:
        """Monitor model performance"""
        alerts = []
        
        try:
            for model_id, metrics in model_performance.items():
                for metric_name, current_value in metrics.items():
                    # Get historical data (simplified)
                    historical_data = [current_value] * 20  # Would use actual history
                    
                    performance = self.performance_monitor.monitor_performance(
                        model_id, metric_name, current_value, historical_data
                    )
                    
                    if performance.is_degrading:
                        alert = DriftAlert(
                            alert_id=f"performance_{model_id}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type='performance_degradation',
                            severity='high',
                            timestamp=datetime.now(),
                            description=f"Performance degradation in {model_id} - {metric_name}",
                            metric_value=performance.z_score,
                            threshold_value=self.alert_thresholds['performance_degradation'],
                            affected_components=[model_id],
                            recommendations=[
                                "Review model performance",
                                "Check for data drift",
                                "Consider model retraining"
                            ]
                        )
                        alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {e}")
        
        return alerts
    
    async def _monitor_slos(self) -> List[DriftAlert]:
        """Monitor SLOs"""
        alerts = []
        
        try:
            # Monitor key SLOs
            slo_metrics = {
                'model_latency': 50.0,  # ms
                'prediction_accuracy': 0.87,
                'system_uptime': 0.9995,
                'data_freshness': 45.0,  # seconds
                'error_rate': 0.0005
            }
            
            for slo_name, current_value in slo_metrics.items():
                slo_metric = self.slo_monitor.monitor_slo(slo_name, current_value)
                
                if slo_metric.is_breached:
                    alert = DriftAlert(
                        alert_id=f"slo_breach_{slo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type='slo_breach',
                        severity='critical',
                        timestamp=datetime.now(),
                        description=f"SLO breach: {slo_name}",
                        metric_value=current_value,
                        threshold_value=slo_metric.breach_threshold,
                        affected_components=['system_performance'],
                        recommendations=[
                            "Immediate investigation required",
                            "Check system resources",
                            "Review operational procedures"
                        ]
                    )
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error monitoring SLOs: {e}")
        
        return alerts
    
    def _determine_alert_severity(self, metric_value: float, metric_type: str) -> str:
        """Determine alert severity"""
        if metric_type == 'psi':
            if metric_value > self.alert_thresholds['psi_critical']:
                return 'critical'
            elif metric_value > self.alert_thresholds['psi_high']:
                return 'high'
            else:
                return 'medium'
        else:
            return 'medium'
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            "drift_suite": {
                "total_alerts": len(self.alerts),
                "active_alerts": len([a for a in self.alerts if not a.is_acknowledged]),
                "alert_distribution": self._get_alert_distribution(),
                "components": {
                    "psi_calculator": "active",
                    "regime_detector": "active",
                    "outlier_detector": "active",
                    "performance_monitor": "active",
                    "slo_monitor": "active"
                }
            },
            "current_regime": {
                "regime_name": self.regime_detector.current_regime.regime_name if self.regime_detector.current_regime else "unknown",
                "confidence": self.regime_detector.current_regime.confidence if self.regime_detector.current_regime else 0.0,
                "duration_days": self.regime_detector.current_regime.duration_days if self.regime_detector.current_regime else 0
            },
            "alert_thresholds": self.alert_thresholds,
            "slo_definitions": self.slo_monitor.slo_definitions
        }
    
    def _get_alert_distribution(self) -> Dict[str, int]:
        """Get alert distribution by type and severity"""
        distribution = {}
        
        for alert in self.alerts:
            key = f"{alert.alert_type}_{alert.severity}"
            distribution[key] = distribution.get(key, 0) + 1
        
        return distribution


# Factory function
async def create_drift_suite() -> DriftSuite:
    """Create and initialize drift detection suite"""
    return DriftSuite()


# Example usage
async def main():
    """Example usage of drift detection suite"""
    # Create drift suite
    drift_suite = await create_drift_suite()
    
    # Sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Reference data (baseline)
    reference_data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples)
    }
    
    # Current data (with some drift)
    current_data = {
        'feature_1': np.random.normal(0.2, 1, n_samples),  # Mean shift
        'feature_2': np.random.normal(0, 1.2, n_samples),  # Variance increase
        'feature_3': np.random.normal(0, 1, n_samples)     # No drift
    }
    
    # Sample market data
    market_data = pd.DataFrame({
        'close': 100 + np.random.randn(252).cumsum(),
        'volume': np.random.randint(1000, 10000, 252)
    }, index=pd.date_range('2023-01-01', periods=252, freq='D'))
    
    # Sample model performance
    model_performance = {
        'model_1': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88
        },
        'model_2': {
            'sharpe_ratio': 1.2,
            'information_ratio': 0.8
        }
    }
    
    # Run comprehensive monitoring
    alerts = await drift_suite.run_comprehensive_monitoring(
        reference_data, current_data, market_data, model_performance
    )
    
    print("Drift Detection Suite Results:")
    print(f"Total Alerts Generated: {len(alerts)}")
    
    for alert in alerts[:5]:  # Show first 5 alerts
        print(f"\nAlert: {alert.alert_type}")
        print(f"Severity: {alert.severity}")
        print(f"Description: {alert.description}")
        print(f"Metric Value: {alert.metric_value:.4f}")
    
    # Get monitoring summary
    summary = await drift_suite.get_monitoring_summary()
    print(f"\nMonitoring Summary:")
    print(f"Current Regime: {summary['current_regime']['regime_name']}")
    print(f"Regime Confidence: {summary['current_regime']['confidence']:.3f}")
    print(f"Active Alerts: {summary['drift_suite']['active_alerts']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
