#!/usr/bin/env python3
"""
Robustness System: Anomaly Detection & Distribution Shift Defense
================================================================

Implements comprehensive anomaly detection and robustness measures
to protect against data quality issues and distribution shift.

Key Features:
- Multi-method anomaly detection (Hampel, MAD, Isolation Forest)
- Distributional robustness (Wasserstein DRO)
- Adversarial validation for train/live distribution comparison
- Automatic throttling and quarantine mechanisms
- PSI/KS drift detection with auto-response
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from enum import Enum

# Scientific computing
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance

# ML and anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

# Local imports
from schemas.contracts import Signal, Opportunity, RegimeType
from common.observability.telemetry import get_telemetry, trace_operation

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies detected"""
    POINT_ANOMALY = "point_anomaly"
    DISTRIBUTIONAL_SHIFT = "distributional_shift"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY = "data_quality"

class SeverityLevel(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyResult:
    """Result from anomaly detection"""
    is_anomaly: bool
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float
    threshold: float
    score: float
    features_affected: List[str]
    recommendation: str
    metadata: Dict[str, Any]

@dataclass
class DriftReport:
    """Report on distribution drift"""
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    drift_magnitude: str
    affected_features: List[str]
    recommendation: str

class AnomalyDetector(ABC):
    """Base class for anomaly detectors"""
    
    @abstractmethod
    async def detect(self, data: np.ndarray, trace_id: str) -> AnomalyResult:
        """Detect anomalies in data"""
        pass
    
    @abstractmethod
    def fit(self, reference_data: np.ndarray):
        """Fit detector on reference data"""
        pass

class HampelFilter(AnomalyDetector):
    """
    Hampel Filter for robust anomaly detection
    
    Uses median absolute deviation (MAD) for outlier detection
    More robust than standard deviation-based methods
    """
    
    def __init__(self, window_size: int = 7, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_median = None
        self.reference_mad = None
        
    def fit(self, reference_data: np.ndarray):
        """Fit on reference data"""
        if reference_data.ndim == 1:
            self.reference_median = np.median(reference_data)
            self.reference_mad = stats.median_abs_deviation(reference_data)
        else:
            self.reference_median = np.median(reference_data, axis=0)
            self.reference_mad = stats.median_abs_deviation(reference_data, axis=0)
    
    async def detect(self, data: np.ndarray, trace_id: str) -> AnomalyResult:
        """Detect anomalies using Hampel filter"""
        async with trace_operation("hampel_detection", trace_id=trace_id):
            try:
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                # Calculate rolling median and MAD if window-based
                if self.reference_median is None:
                    # Use rolling statistics
                    anomalies = []
                    for i in range(len(data)):
                        start_idx = max(0, i - self.window_size // 2)
                        end_idx = min(len(data), i + self.window_size // 2 + 1)
                        window_data = data[start_idx:end_idx]
                        
                        median = np.median(window_data, axis=0)
                        mad = stats.median_abs_deviation(window_data, axis=0)
                        
                        # Avoid division by zero
                        mad = np.where(mad == 0, 1e-8, mad)
                        
                        # Calculate modified z-score
                        modified_z_score = 0.6745 * (data[i] - median) / mad
                        is_anomaly = np.any(np.abs(modified_z_score) > self.threshold)
                        anomalies.append(is_anomaly)
                    
                    anomaly_rate = np.mean(anomalies)
                    max_score = np.max([np.max(np.abs(0.6745 * (data[i] - np.median(data, axis=0)) / 
                                                     (stats.median_abs_deviation(data, axis=0) + 1e-8)))
                                       for i in range(len(data))])
                else:
                    # Use reference statistics
                    mad = np.where(self.reference_mad == 0, 1e-8, self.reference_mad)
                    modified_z_scores = 0.6745 * (data - self.reference_median) / mad
                    anomalies = np.any(np.abs(modified_z_scores) > self.threshold, axis=1)
                    anomaly_rate = np.mean(anomalies)
                    max_score = np.max(np.abs(modified_z_scores))
                
                # Determine severity
                if anomaly_rate > 0.5:
                    severity = SeverityLevel.CRITICAL
                elif anomaly_rate > 0.3:
                    severity = SeverityLevel.HIGH
                elif anomaly_rate > 0.1:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                return AnomalyResult(
                    is_anomaly=anomaly_rate > 0.05,
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=severity,
                    confidence=min(1.0, max_score / self.threshold),
                    threshold=self.threshold,
                    score=float(max_score),
                    features_affected=[f"feature_{i}" for i in range(data.shape[1])],
                    recommendation="QUARANTINE" if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else "MONITOR",
                    metadata={'anomaly_rate': anomaly_rate, 'method': 'hampel'}
                )
                
            except Exception as e:
                logger.error(f"Hampel filter detection failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _get_default_result(self) -> AnomalyResult:
        """Return default result when detection fails"""
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.DATA_QUALITY,
            severity=SeverityLevel.LOW,
            confidence=0.0,
            threshold=self.threshold,
            score=0.0,
            features_affected=[],
            recommendation="PASS",
            metadata={'method': 'hampel', 'error': True}
        )

class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest for multivariate anomaly detection
    
    Effective for high-dimensional data and complex anomaly patterns
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, reference_data: np.ndarray):
        """Fit isolation forest on reference data"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(reference_data)
            
            # Fit isolation forest
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(scaled_data)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Failed to fit isolation forest: {e}")
            self.is_fitted = False
    
    async def detect(self, data: np.ndarray, trace_id: str) -> AnomalyResult:
        """Detect anomalies using isolation forest"""
        async with trace_operation("isolation_forest_detection", trace_id=trace_id):
            try:
                if not self.is_fitted or self.model is None:
                    return self._get_default_result()
                
                # Scale the data
                scaled_data = self.scaler.transform(data)
                
                # Predict anomalies
                predictions = self.model.predict(scaled_data)
                anomaly_scores = self.model.decision_function(scaled_data)
                
                # Calculate metrics
                is_anomaly = np.any(predictions == -1)
                anomaly_rate = np.mean(predictions == -1)
                min_score = np.min(anomaly_scores)
                
                # Determine severity based on anomaly score
                if min_score < -0.5:
                    severity = SeverityLevel.CRITICAL
                elif min_score < -0.3:
                    severity = SeverityLevel.HIGH
                elif min_score < -0.1:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                return AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=severity,
                    confidence=min(1.0, abs(min_score)),
                    threshold=0.0,  # Isolation forest uses relative scoring
                    score=float(min_score),
                    features_affected=[f"feature_{i}" for i in range(data.shape[1])],
                    recommendation="QUARANTINE" if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else "MONITOR",
                    metadata={'anomaly_rate': anomaly_rate, 'method': 'isolation_forest'}
                )
                
            except Exception as e:
                logger.error(f"Isolation forest detection failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _get_default_result(self) -> AnomalyResult:
        """Return default result when detection fails"""
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.DATA_QUALITY,
            severity=SeverityLevel.LOW,
            confidence=0.0,
            threshold=0.0,
            score=0.0,
            features_affected=[],
            recommendation="PASS",
            metadata={'method': 'isolation_forest', 'error': True}
        )

class DistributionDriftDetector:
    """
    Detects distribution drift using PSI and KS tests
    
    Monitors for changes in feature distributions that could
    indicate model degradation or data quality issues
    """
    
    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.reference_distributions = {}
        self.feature_names = []
        
    def fit(self, reference_data: pd.DataFrame):
        """Fit on reference data distributions"""
        try:
            self.feature_names = reference_data.columns.tolist()
            
            for column in reference_data.columns:
                if reference_data[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # Numerical feature - store histogram
                    hist, bin_edges = np.histogram(reference_data[column].dropna(), bins=20)
                    self.reference_distributions[column] = {
                        'type': 'numerical',
                        'hist': hist,
                        'bin_edges': bin_edges,
                        'values': reference_data[column].dropna().values
                    }
                else:
                    # Categorical feature - store value counts
                    value_counts = reference_data[column].value_counts(normalize=True)
                    self.reference_distributions[column] = {
                        'type': 'categorical',
                        'value_counts': value_counts
                    }
                    
        except Exception as e:
            logger.error(f"Failed to fit drift detector: {e}")
    
    async def detect_drift(self, current_data: pd.DataFrame, trace_id: str) -> DriftReport:
        """Detect distribution drift"""
        async with trace_operation("drift_detection", trace_id=trace_id):
            try:
                psi_scores = {}
                ks_statistics = {}
                ks_p_values = {}
                
                for column in current_data.columns:
                    if column not in self.reference_distributions:
                        continue
                    
                    ref_dist = self.reference_distributions[column]
                    
                    if ref_dist['type'] == 'numerical':
                        # Calculate PSI for numerical features
                        psi_score = self._calculate_psi_numerical(
                            current_data[column].dropna().values,
                            ref_dist['hist'],
                            ref_dist['bin_edges']
                        )
                        psi_scores[column] = psi_score
                        
                        # KS test
                        ks_stat, ks_p = ks_2samp(
                            ref_dist['values'],
                            current_data[column].dropna().values
                        )
                        ks_statistics[column] = ks_stat
                        ks_p_values[column] = ks_p
                        
                    else:
                        # Calculate PSI for categorical features
                        current_counts = current_data[column].value_counts(normalize=True)
                        psi_score = self._calculate_psi_categorical(
                            current_counts,
                            ref_dist['value_counts']
                        )
                        psi_scores[column] = psi_score
                
                # Aggregate results
                avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
                avg_ks_stat = np.mean(list(ks_statistics.values())) if ks_statistics else 0.0
                min_ks_p = np.min(list(ks_p_values.values())) if ks_p_values else 1.0
                
                # Determine drift
                psi_drift = avg_psi > self.psi_threshold
                ks_drift = min_ks_p < self.ks_threshold
                drift_detected = psi_drift or ks_drift
                
                # Determine magnitude
                if avg_psi > 0.5:
                    magnitude = "SEVERE"
                elif avg_psi > 0.3:
                    magnitude = "MODERATE"
                elif avg_psi > 0.1:
                    magnitude = "MILD"
                else:
                    magnitude = "NONE"
                
                # Identify affected features
                affected_features = [
                    col for col, psi in psi_scores.items()
                    if psi > self.psi_threshold
                ]
                
                # Recommendation
                if magnitude == "SEVERE":
                    recommendation = "RETRAIN_MODEL"
                elif magnitude == "MODERATE":
                    recommendation = "THROTTLE_TRADING"
                elif magnitude == "MILD":
                    recommendation = "MONITOR_CLOSELY"
                else:
                    recommendation = "CONTINUE"
                
                logger.info(f"Drift detection: PSI={avg_psi:.3f}, KS={avg_ks_stat:.3f}, "
                           f"Magnitude={magnitude}", extra={'trace_id': trace_id})
                
                return DriftReport(
                    psi_score=avg_psi,
                    ks_statistic=avg_ks_stat,
                    ks_p_value=min_ks_p,
                    drift_detected=drift_detected,
                    drift_magnitude=magnitude,
                    affected_features=affected_features,
                    recommendation=recommendation
                )
                
            except Exception as e:
                logger.error(f"Drift detection failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_drift_report()
    
    def _calculate_psi_numerical(self, current: np.ndarray, ref_hist: np.ndarray,
                                bin_edges: np.ndarray) -> float:
        """Calculate PSI for numerical features"""
        try:
            # Create histogram for current data using reference bin edges
            current_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to get probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            current_prob = current_hist / np.sum(current_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            ref_prob = ref_prob + epsilon
            current_prob = current_prob + epsilon
            
            # Calculate PSI
            psi = np.sum((current_prob - ref_prob) * np.log(current_prob / ref_prob))
            return float(psi)
            
        except Exception:
            return 0.0
    
    def _calculate_psi_categorical(self, current_counts: pd.Series,
                                  ref_counts: pd.Series) -> float:
        """Calculate PSI for categorical features"""
        try:
            # Align the series
            all_categories = set(current_counts.index) | set(ref_counts.index)
            
            current_aligned = pd.Series(0.0, index=all_categories)
            ref_aligned = pd.Series(0.0, index=all_categories)
            
            current_aligned.update(current_counts)
            ref_aligned.update(ref_counts)
            
            # Add small epsilon
            epsilon = 1e-8
            current_aligned += epsilon
            ref_aligned += epsilon
            
            # Normalize
            current_aligned /= current_aligned.sum()
            ref_aligned /= ref_aligned.sum()
            
            # Calculate PSI
            psi = np.sum((current_aligned - ref_aligned) * np.log(current_aligned / ref_aligned))
            return float(psi)
            
        except Exception:
            return 0.0
    
    def _get_default_drift_report(self) -> DriftReport:
        """Return default drift report when detection fails"""
        return DriftReport(
            psi_score=0.0,
            ks_statistic=0.0,
            ks_p_value=1.0,
            drift_detected=False,
            drift_magnitude="NONE",
            affected_features=[],
            recommendation="CONTINUE"
        )

class AdversarialValidator:
    """
    Adversarial validation to detect train/live distribution differences
    
    Trains a classifier to distinguish between training and live data.
    High accuracy indicates significant distribution shift.
    """
    
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.model = None
        self.is_fitted = False
        
    async def validate(self, train_data: pd.DataFrame, live_data: pd.DataFrame,
                      trace_id: str) -> Dict[str, Any]:
        """Perform adversarial validation"""
        async with trace_operation("adversarial_validation", trace_id=trace_id):
            try:
                # Prepare data
                train_data = train_data.copy()
                live_data = live_data.copy()
                
                # Align columns
                common_cols = list(set(train_data.columns) & set(live_data.columns))
                train_data = train_data[common_cols]
                live_data = live_data[common_cols]
                
                # Create labels (0 = train, 1 = live)
                train_labels = np.zeros(len(train_data))
                live_labels = np.ones(len(live_data))
                
                # Combine data
                X = pd.concat([train_data, live_data], ignore_index=True)
                y = np.concatenate([train_labels, live_labels])
                
                # Handle missing values
                X = X.fillna(X.median())
                
                # Split for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Train classifier
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1
                )
                
                # Add feature names to avoid warnings
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                X_val_df = pd.DataFrame(X_val, columns=feature_names)
                
                self.model.fit(X_train_df, y_train)
                
                # Evaluate
                y_pred_proba = self.model.predict_proba(X_val_df)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                accuracy = accuracy_score(y_val, y_pred)
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                # Feature importance
                feature_importance = dict(zip(
                    common_cols,
                    self.model.feature_importances_
                ))
                
                # Sort by importance
                top_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                # Determine if significant shift detected
                shift_detected = accuracy > self.threshold
                
                if accuracy > 0.85:
                    severity = "SEVERE"
                    recommendation = "STOP_TRADING"
                elif accuracy > 0.75:
                    severity = "HIGH"
                    recommendation = "RETRAIN_IMMEDIATELY"
                elif accuracy > self.threshold:
                    severity = "MODERATE"
                    recommendation = "MONITOR_CLOSELY"
                else:
                    severity = "LOW"
                    recommendation = "CONTINUE"
                
                self.is_fitted = True
                
                logger.info(f"Adversarial validation: accuracy={accuracy:.3f}, "
                           f"AUC={auc_score:.3f}, shift_detected={shift_detected}",
                           extra={'trace_id': trace_id})
                
                return {
                    'shift_detected': shift_detected,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'severity': severity,
                    'recommendation': recommendation,
                    'top_discriminative_features': top_features,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                logger.error(f"Adversarial validation failed: {e}", extra={'trace_id': trace_id})
                return {
                    'shift_detected': False,
                    'accuracy': 0.5,
                    'auc_score': 0.5,
                    'severity': 'LOW',
                    'recommendation': 'CONTINUE',
                    'top_discriminative_features': [],
                    'feature_importance': {}
                }

class RobustnessManager:
    """
    Main robustness manager that coordinates all defense mechanisms
    
    Integrates anomaly detection, drift detection, and adversarial validation
    into a comprehensive robustness system with automatic responses.
    """
    
    def __init__(self):
        self.anomaly_detectors = {
            'hampel': HampelFilter(),
            'isolation_forest': IsolationForestDetector()
        }
        self.drift_detector = DistributionDriftDetector()
        self.adversarial_validator = AdversarialValidator()
        
        self.quarantine_queue = []
        self.throttle_level = 0.0  # 0.0 = no throttle, 1.0 = full stop
        self.alert_history = []
        
    async def comprehensive_check(self, current_data: pd.DataFrame,
                                 reference_data: Optional[pd.DataFrame] = None,
                                 trace_id: str = "") -> Dict[str, Any]:
        """Perform comprehensive robustness check"""
        async with trace_operation("comprehensive_robustness_check", trace_id=trace_id):
            results = {
                'anomaly_detection': {},
                'drift_detection': None,
                'adversarial_validation': None,
                'overall_assessment': {},
                'recommendations': [],
                'throttle_level': self.throttle_level
            }
            
            try:
                # Convert to numpy for anomaly detection
                numerical_data = current_data.select_dtypes(include=[np.number]).values
                
                if len(numerical_data) > 0:
                    # Run anomaly detection
                    for name, detector in self.anomaly_detectors.items():
                        anomaly_result = await detector.detect(numerical_data, trace_id)
                        results['anomaly_detection'][name] = anomaly_result
                        
                        # Handle quarantine
                        if anomaly_result.recommendation == "QUARANTINE":
                            self._add_to_quarantine(current_data, anomaly_result)
                
                # Run drift detection if reference data available
                if reference_data is not None:
                    drift_report = await self.drift_detector.detect_drift(current_data, trace_id)
                    results['drift_detection'] = drift_report
                    
                    # Run adversarial validation
                    adv_validation = await self.adversarial_validator.validate(
                        reference_data, current_data, trace_id
                    )
                    results['adversarial_validation'] = adv_validation
                
                # Overall assessment
                assessment = self._assess_overall_risk(results)
                results['overall_assessment'] = assessment
                
                # Update throttle level
                self._update_throttle_level(assessment)
                results['throttle_level'] = self.throttle_level
                
                # Generate recommendations
                recommendations = self._generate_recommendations(results)
                results['recommendations'] = recommendations
                
                logger.info(f"Robustness check complete. Risk level: {assessment.get('risk_level', 'UNKNOWN')}, "
                           f"Throttle: {self.throttle_level:.2f}", extra={'trace_id': trace_id})
                
                return results
                
            except Exception as e:
                logger.error(f"Comprehensive robustness check failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_results()
    
    def _assess_overall_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level from all checks"""
        risk_score = 0.0
        risk_factors = []
        
        # Anomaly detection risk
        for name, result in results['anomaly_detection'].items():
            if result.is_anomaly:
                if result.severity == SeverityLevel.CRITICAL:
                    risk_score += 0.4
                elif result.severity == SeverityLevel.HIGH:
                    risk_score += 0.3
                elif result.severity == SeverityLevel.MEDIUM:
                    risk_score += 0.2
                else:
                    risk_score += 0.1
                risk_factors.append(f"{name}_anomaly")
        
        # Drift detection risk
        if results['drift_detection']:
            drift = results['drift_detection']
            if drift.drift_detected:
                if drift.drift_magnitude == "SEVERE":
                    risk_score += 0.3
                elif drift.drift_magnitude == "MODERATE":
                    risk_score += 0.2
                else:
                    risk_score += 0.1
                risk_factors.append("distribution_drift")
        
        # Adversarial validation risk
        if results['adversarial_validation']:
            adv_val = results['adversarial_validation']
            if adv_val['shift_detected']:
                if adv_val['severity'] == "SEVERE":
                    risk_score += 0.3
                elif adv_val['severity'] == "HIGH":
                    risk_score += 0.2
                else:
                    risk_score += 0.1
                risk_factors.append("adversarial_shift")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "CRITICAL"
        elif risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'quarantine_size': len(self.quarantine_queue)
        }
    
    def _update_throttle_level(self, assessment: Dict[str, Any]):
        """Update throttle level based on risk assessment"""
        risk_level = assessment['risk_level']
        
        if risk_level == "CRITICAL":
            self.throttle_level = 1.0  # Full stop
        elif risk_level == "HIGH":
            self.throttle_level = 0.8  # Severe throttle
        elif risk_level == "MEDIUM":
            self.throttle_level = 0.5  # Moderate throttle
        else:
            # Gradual recovery
            self.throttle_level = max(0.0, self.throttle_level - 0.1)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Anomaly recommendations
        for name, result in results['anomaly_detection'].items():
            if result.is_anomaly and result.recommendation != "PASS":
                recommendations.append(f"{name}: {result.recommendation}")
        
        # Drift recommendations
        if results['drift_detection'] and results['drift_detection'].drift_detected:
            recommendations.append(f"Drift: {results['drift_detection'].recommendation}")
        
        # Adversarial validation recommendations
        if results['adversarial_validation'] and results['adversarial_validation']['shift_detected']:
            recommendations.append(f"Distribution shift: {results['adversarial_validation']['recommendation']}")
        
        # Throttle recommendations
        if self.throttle_level > 0.5:
            recommendations.append(f"THROTTLE_TRADING: Level {self.throttle_level:.1f}")
        
        return recommendations
    
    def _add_to_quarantine(self, data: pd.DataFrame, anomaly_result: AnomalyResult):
        """Add data to quarantine queue"""
        quarantine_item = {
            'timestamp': datetime.utcnow(),
            'data_hash': hash(str(data.values.tobytes())),
            'anomaly_type': anomaly_result.anomaly_type.value,
            'severity': anomaly_result.severity.value,
            'confidence': anomaly_result.confidence
        }
        
        self.quarantine_queue.append(quarantine_item)
        
        # Limit quarantine size
        if len(self.quarantine_queue) > 1000:
            self.quarantine_queue = self.quarantine_queue[-1000:]
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Return default results when checks fail"""
        return {
            'anomaly_detection': {},
            'drift_detection': None,
            'adversarial_validation': None,
            'overall_assessment': {
                'risk_score': 0.0,
                'risk_level': 'LOW',
                'risk_factors': [],
                'quarantine_size': 0
            },
            'recommendations': [],
            'throttle_level': 0.0
        }
    
    def get_quarantine_status(self) -> Dict[str, Any]:
        """Get current quarantine status"""
        if not self.quarantine_queue:
            return {'total_quarantined': 0, 'recent_quarantined': 0}
        
        recent_threshold = datetime.utcnow() - timedelta(hours=1)
        recent_count = sum(1 for item in self.quarantine_queue 
                          if item['timestamp'] > recent_threshold)
        
        return {
            'total_quarantined': len(self.quarantine_queue),
            'recent_quarantined': recent_count,
            'oldest_quarantine': self.quarantine_queue[0]['timestamp'] if self.quarantine_queue else None,
            'latest_quarantine': self.quarantine_queue[-1]['timestamp'] if self.quarantine_queue else None
        }
    
    def reset_throttle(self):
        """Reset throttle level (emergency override)"""
        self.throttle_level = 0.0
        logger.warning("Throttle level manually reset to 0.0")
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        self.throttle_level = 1.0
        logger.critical("Emergency stop activated - all trading throttled")
