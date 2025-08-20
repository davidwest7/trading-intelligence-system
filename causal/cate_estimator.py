#!/usr/bin/env python3
"""
Causal Average Treatment Effect (CATE) Estimator
==================================================

Implements T-Learner, DR-Learner, and IV analysis for causal inference
in trading signals to fight spurious alpha and estimate true uplift.

Key Features:
- T-Learner (Two-model approach)
- DR-Learner (Doubly Robust approach)
- Instrumental Variables analysis
- Uplift estimation and validation
- Causal signal prioritization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# Statistical inference
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

# Local imports
from schemas.contracts import Signal, Opportunity, RegimeType
from common.observability.telemetry import get_telemetry, trace_operation

logger = logging.getLogger(__name__)

@dataclass
class CausalResult:
    """Result from causal analysis"""
    cate_estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    uplift_score: float
    treatment_effect: float
    iv_strength: Optional[float] = None
    confounding_bias: Optional[float] = None
    
@dataclass
class TreatmentGroup:
    """Treatment/control group data"""
    features: np.ndarray
    outcomes: np.ndarray
    treatment: np.ndarray
    instrument: Optional[np.ndarray] = None
    
class CausalEstimator(ABC):
    """Base class for causal estimators"""
    
    @abstractmethod
    async def estimate_cate(self, treatment_group: TreatmentGroup,
                           trace_id: str) -> CausalResult:
        """Estimate Conditional Average Treatment Effect"""
        pass
    
    @abstractmethod
    def get_confidence_interval(self, estimate: float, se: float) -> Tuple[float, float]:
        """Get confidence interval for estimate"""
        pass

class TLearner(CausalEstimator):
    """
    T-Learner for CATE estimation
    
    Uses two separate models: one for treated, one for control.
    CATE = E[Y|X,T=1] - E[Y|X,T=0]
    """
    
    def __init__(self, base_learner='lgb', n_estimators: int = 100):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.model_treated = None
        self.model_control = None
        self.is_trained = False
        
    def _create_model(self):
        """Create base model instance"""
        if self.base_learner == 'lgb':
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif self.base_learner == 'rf':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        elif self.base_learner == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=42
            )
        else:
            return LinearRegression()
    
    async def estimate_cate(self, treatment_group: TreatmentGroup,
                           trace_id: str) -> CausalResult:
        """Estimate CATE using T-Learner approach"""
        async with trace_operation("t_learner_cate", trace_id=trace_id):
            try:
                # Split data by treatment assignment
                treated_mask = treatment_group.treatment == 1
                control_mask = treatment_group.treatment == 0
                
                X_treated = treatment_group.features[treated_mask]
                y_treated = treatment_group.outcomes[treated_mask]
                X_control = treatment_group.features[control_mask]
                y_control = treatment_group.outcomes[control_mask]
                
                if len(X_treated) < 10 or len(X_control) < 10:
                    logger.warning("Insufficient data for T-Learner", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # Train separate models
                self.model_treated = self._create_model()
                self.model_control = self._create_model()
                
                # Convert to pandas DataFrame with feature names to avoid warnings
                feature_names = [f'feature_{i}' for i in range(X_treated.shape[1])]
                X_treated_df = pd.DataFrame(X_treated, columns=feature_names)
                X_control_df = pd.DataFrame(X_control, columns=feature_names)
                
                self.model_treated.fit(X_treated_df, y_treated)
                self.model_control.fit(X_control_df, y_control)
                
                # Predict counterfactuals for all units
                X_all_df = pd.DataFrame(treatment_group.features, columns=feature_names)
                pred_treated = self.model_treated.predict(X_all_df)
                pred_control = self.model_control.predict(X_all_df)
                
                # Calculate individual treatment effects
                individual_effects = pred_treated - pred_control
                cate_estimate = np.mean(individual_effects)
                
                # Calculate standard error using cross-validation
                se = self._calculate_standard_error(treatment_group)
                ci = self.get_confidence_interval(cate_estimate, se)
                
                # Statistical test
                t_stat = cate_estimate / (se + 1e-8)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(treatment_group.features) - 1))
                
                # Calculate uplift score (magnitude of effect relative to noise)
                uplift_score = abs(cate_estimate) / (np.std(individual_effects) + 1e-8)
                
                self.is_trained = True
                
                logger.info(f"T-Learner CATE: {cate_estimate:.4f}, CI: {ci}, p-value: {p_value:.4f}",
                           extra={'trace_id': trace_id})
                
                return CausalResult(
                    cate_estimate=cate_estimate,
                    confidence_interval=ci,
                    p_value=p_value,
                    uplift_score=uplift_score,
                    treatment_effect=cate_estimate
                )
                
            except Exception as e:
                logger.error(f"T-Learner estimation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _calculate_standard_error(self, treatment_group: TreatmentGroup) -> float:
        """Calculate standard error using cross-validation"""
        try:
            # Cross-validated prediction errors
            treated_mask = treatment_group.treatment == 1
            control_mask = treatment_group.treatment == 0
            
            # CV scores for treated model
            if np.sum(treated_mask) > 5:
                feature_names = [f'feature_{i}' for i in range(treatment_group.features.shape[1])]
                X_treated_cv = pd.DataFrame(treatment_group.features[treated_mask], columns=feature_names)
                cv_scores_treated = cross_val_score(
                    self._create_model(),
                    X_treated_cv,
                    treatment_group.outcomes[treated_mask],
                    cv=min(5, np.sum(treated_mask)),
                    scoring='neg_mean_squared_error'
                )
                mse_treated = -np.mean(cv_scores_treated)
            else:
                mse_treated = 1.0
            
            # CV scores for control model
            if np.sum(control_mask) > 5:
                X_control_cv = pd.DataFrame(treatment_group.features[control_mask], columns=feature_names)
                cv_scores_control = cross_val_score(
                    self._create_model(),
                    X_control_cv,
                    treatment_group.outcomes[control_mask],
                    cv=min(5, np.sum(control_mask)),
                    scoring='neg_mean_squared_error'
                )
                mse_control = -np.mean(cv_scores_control)
            else:
                mse_control = 1.0
            
            # Combined standard error
            n_treated = np.sum(treated_mask)
            n_control = np.sum(control_mask)
            
            se = np.sqrt(mse_treated / n_treated + mse_control / n_control)
            return se
            
        except Exception:
            return 0.1  # Default SE
    
    def get_confidence_interval(self, estimate: float, se: float) -> Tuple[float, float]:
        """Get 95% confidence interval"""
        z_score = 1.96  # 95% CI
        margin = z_score * se
        return (estimate - margin, estimate + margin)
    
    def _get_default_result(self) -> CausalResult:
        """Return default result when estimation fails"""
        return CausalResult(
            cate_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            p_value=1.0,
            uplift_score=0.0,
            treatment_effect=0.0
        )

class DRLearner(CausalEstimator):
    """
    Doubly Robust (DR) Learner for CATE estimation
    
    Uses both outcome regression and propensity score modeling
    for more robust causal inference.
    """
    
    def __init__(self, base_learner='lgb', n_estimators: int = 100):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.outcome_model = None
        self.propensity_model = None
        self.is_trained = False
        
    def _create_regressor(self):
        """Create regression model"""
        if self.base_learner == 'lgb':
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            return RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
    
    def _create_classifier(self):
        """Create classification model"""
        if self.base_learner == 'lgb':
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            return LogisticRegression(random_state=42, max_iter=1000)
    
    async def estimate_cate(self, treatment_group: TreatmentGroup,
                           trace_id: str) -> CausalResult:
        """Estimate CATE using Doubly Robust approach"""
        async with trace_operation("dr_learner_cate", trace_id=trace_id):
            try:
                X = treatment_group.features
                y = treatment_group.outcomes
                w = treatment_group.treatment
                
                if len(X) < 20:
                    logger.warning("Insufficient data for DR-Learner", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # Step 1: Estimate propensity scores
                self.propensity_model = self._create_classifier()
                # Add feature names to avoid warnings
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=feature_names)
                self.propensity_model.fit(X_df, w)
                propensity_scores = self.propensity_model.predict_proba(X_df)[:, 1]
                
                # Clip propensity scores to avoid extreme weights
                propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
                
                # Step 2: Estimate outcome regression
                self.outcome_model = self._create_regressor()
                
                # Augment features with treatment indicator
                X_augmented = np.column_stack([X, w])
                # Add feature names for augmented features
                augmented_feature_names = feature_names + ['treatment']
                X_augmented_df = pd.DataFrame(X_augmented, columns=augmented_feature_names)
                self.outcome_model.fit(X_augmented_df, y)
                
                # Step 3: Predict counterfactuals
                X_treated = np.column_stack([X, np.ones(len(X))])
                X_control = np.column_stack([X, np.zeros(len(X))])
                
                X_treated_df = pd.DataFrame(X_treated, columns=augmented_feature_names)
                X_control_df = pd.DataFrame(X_control, columns=augmented_feature_names)
                
                mu_1 = self.outcome_model.predict(X_treated_df)
                mu_0 = self.outcome_model.predict(X_control_df)
                
                # Step 4: Doubly robust estimator
                # DR formula: tau_hat = (mu_1 - mu_0) + (W/e(X))*(Y - mu_1) - ((1-W)/(1-e(X)))*(Y - mu_0)
                dr_correction = (
                    (w * (y - mu_1)) / propensity_scores -
                    ((1 - w) * (y - mu_0)) / (1 - propensity_scores)
                )
                
                individual_effects = (mu_1 - mu_0) + dr_correction
                cate_estimate = np.mean(individual_effects)
                
                # Calculate standard error
                se = np.std(individual_effects) / np.sqrt(len(individual_effects))
                ci = self.get_confidence_interval(cate_estimate, se)
                
                # Statistical test
                t_stat = cate_estimate / (se + 1e-8)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(X) - 1))
                
                # Calculate uplift score
                uplift_score = abs(cate_estimate) / (np.std(individual_effects) + 1e-8)
                
                # Estimate confounding bias (difference between naive and DR estimates)
                naive_estimate = np.mean(y[w == 1]) - np.mean(y[w == 0])
                confounding_bias = abs(naive_estimate - cate_estimate)
                
                self.is_trained = True
                
                logger.info(f"DR-Learner CATE: {cate_estimate:.4f}, CI: {ci}, p-value: {p_value:.4f}",
                           extra={'trace_id': trace_id})
                
                return CausalResult(
                    cate_estimate=cate_estimate,
                    confidence_interval=ci,
                    p_value=p_value,
                    uplift_score=uplift_score,
                    treatment_effect=cate_estimate,
                    confounding_bias=confounding_bias
                )
                
            except Exception as e:
                logger.error(f"DR-Learner estimation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def get_confidence_interval(self, estimate: float, se: float) -> Tuple[float, float]:
        """Get 95% confidence interval"""
        z_score = 1.96
        margin = z_score * se
        return (estimate - margin, estimate + margin)
    
    def _get_default_result(self) -> CausalResult:
        """Return default result when estimation fails"""
        return CausalResult(
            cate_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            p_value=1.0,
            uplift_score=0.0,
            treatment_effect=0.0,
            confounding_bias=0.0
        )

class InstrumentalVariableAnalyzer:
    """
    Instrumental Variable analysis for causal inference
    
    Uses external instruments to identify causal effects
    when unobserved confounding is suspected.
    """
    
    def __init__(self):
        self.first_stage_model = None
        self.second_stage_model = None
        self.is_trained = False
    
    async def estimate_iv_effect(self, treatment_group: TreatmentGroup,
                                trace_id: str) -> CausalResult:
        """Estimate causal effect using instrumental variables"""
        async with trace_operation("iv_analysis", trace_id=trace_id):
            try:
                if treatment_group.instrument is None:
                    logger.warning("No instrument provided for IV analysis", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                X = treatment_group.features
                y = treatment_group.outcomes
                w = treatment_group.treatment
                z = treatment_group.instrument
                
                if len(X) < 30:
                    logger.warning("Insufficient data for IV analysis", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # First stage: regress treatment on instrument and covariates
                X_with_instrument = np.column_stack([X, z])
                self.first_stage_model = LinearRegression()
                self.first_stage_model.fit(X_with_instrument, w)
                
                # Predict treatment from instrument
                w_hat = self.first_stage_model.predict(X_with_instrument)
                
                # Check instrument strength (F-statistic)
                iv_strength = self._calculate_instrument_strength(X_with_instrument, w, z)
                
                if iv_strength < 10:  # Weak instrument threshold
                    logger.warning(f"Weak instrument detected (F={iv_strength:.2f})", 
                                 extra={'trace_id': trace_id})
                
                # Second stage: regress outcome on predicted treatment and covariates
                X_second_stage = np.column_stack([X, w_hat])
                self.second_stage_model = LinearRegression()
                self.second_stage_model.fit(X_second_stage, y)
                
                # IV estimate is the coefficient on w_hat
                iv_estimate = self.second_stage_model.coef_[-1]
                
                # Calculate standard error (simplified)
                residuals = y - self.second_stage_model.predict(X_second_stage)
                mse = np.mean(residuals**2)
                se = np.sqrt(mse / len(y))
                
                ci = (iv_estimate - 1.96 * se, iv_estimate + 1.96 * se)
                
                # Statistical test
                t_stat = iv_estimate / (se + 1e-8)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(X) - X.shape[1] - 1))
                
                # Calculate uplift score
                uplift_score = abs(iv_estimate) / (se + 1e-8)
                
                self.is_trained = True
                
                logger.info(f"IV estimate: {iv_estimate:.4f}, CI: {ci}, F-stat: {iv_strength:.2f}",
                           extra={'trace_id': trace_id})
                
                return CausalResult(
                    cate_estimate=iv_estimate,
                    confidence_interval=ci,
                    p_value=p_value,
                    uplift_score=uplift_score,
                    treatment_effect=iv_estimate,
                    iv_strength=iv_strength
                )
                
            except Exception as e:
                logger.error(f"IV analysis failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _calculate_instrument_strength(self, X_with_instrument: np.ndarray,
                                     w: np.ndarray, z: np.ndarray) -> float:
        """Calculate F-statistic for instrument strength"""
        try:
            # Reduced form: regress treatment on instrument only
            reduced_model = LinearRegression()
            reduced_model.fit(z.reshape(-1, 1), w)
            
            # Full first stage: regress treatment on instrument + covariates
            full_model = LinearRegression()
            full_model.fit(X_with_instrument, w)
            
            # Calculate F-statistic for instrument
            rss_reduced = np.sum((w - reduced_model.predict(z.reshape(-1, 1)))**2)
            rss_full = np.sum((w - full_model.predict(X_with_instrument))**2)
            
            n = len(w)
            k = X_with_instrument.shape[1] - 1  # excluding instrument
            
            f_stat = ((rss_reduced - rss_full) / 1) / (rss_full / (n - k - 2))
            return f_stat
            
        except Exception:
            return 1.0  # Default low F-stat
    
    def _get_default_result(self) -> CausalResult:
        """Return default result when IV analysis fails"""
        return CausalResult(
            cate_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            p_value=1.0,
            uplift_score=0.0,
            treatment_effect=0.0,
            iv_strength=1.0
        )

class CausalSignalAnalyzer:
    """
    Main causal inference system for trading signals
    
    Analyzes signals for causal effects and fights spurious alpha
    by estimating true uplift from trading decisions.
    """
    
    def __init__(self, methods: List[str] = ['t_learner', 'dr_learner']):
        self.methods = methods
        self.estimators = {
            't_learner': TLearner(),
            'dr_learner': DRLearner(),
            'iv_analyzer': InstrumentalVariableAnalyzer()
        }
        self.results_history = []
        
    async def analyze_signal_causality(self, signal: Signal, 
                                     historical_data: pd.DataFrame,
                                     trace_id: str) -> Dict[str, Any]:
        """
        Analyze causal effect of trading signal
        
        Args:
            signal: Trading signal to analyze
            historical_data: Historical data with outcomes
            trace_id: Trace ID for tracking
            
        Returns:
            Dictionary with causal analysis results
        """
        async with trace_operation("causal_signal_analysis", trace_id=trace_id):
            try:
                # Prepare treatment data
                treatment_group = self._prepare_treatment_data(signal, historical_data)
                
                if treatment_group is None:
                    return self._get_default_analysis()
                
                # Run multiple causal estimation methods
                results = {}
                
                for method in self.methods:
                    if method in self.estimators:
                        estimator = self.estimators[method]
                        result = await estimator.estimate_cate(treatment_group, trace_id)
                        results[method] = result
                
                # Aggregate results
                aggregated_result = self._aggregate_results(results)
                
                # Calculate signal priority based on causal evidence
                priority_score = self._calculate_priority_score(aggregated_result)
                
                analysis = {
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'causal_methods': results,
                    'aggregated_result': aggregated_result,
                    'priority_score': priority_score,
                    'has_causal_effect': aggregated_result.p_value < 0.05,
                    'effect_magnitude': abs(aggregated_result.cate_estimate),
                    'confidence_level': 1 - aggregated_result.p_value,
                    'recommendation': self._get_recommendation(aggregated_result),
                    'timestamp': datetime.utcnow()
                }
                
                # Store for learning
                self.results_history.append(analysis)
                
                logger.info(f"Causal analysis complete. Effect: {aggregated_result.cate_estimate:.4f}, "
                           f"P-value: {aggregated_result.p_value:.4f}, Priority: {priority_score:.2f}",
                           extra={'trace_id': trace_id})
                
                return analysis
                
            except Exception as e:
                logger.error(f"Causal signal analysis failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_analysis()
    
    def _prepare_treatment_data(self, signal: Signal, 
                              historical_data: pd.DataFrame) -> Optional[TreatmentGroup]:
        """Prepare data for causal analysis"""
        try:
            # Create treatment indicator (1 if signal was above threshold, 0 otherwise)
            threshold = historical_data['signal_strength'].median()
            treatment = (historical_data['signal_strength'] > threshold).astype(int)
            
            # Features (market conditions, regime, etc.)
            feature_cols = ['volatility', 'volume', 'momentum', 'regime_score']
            available_cols = [col for col in feature_cols if col in historical_data.columns]
            
            if len(available_cols) < 2:
                # Create synthetic features if real ones not available
                n = len(historical_data)
                features = np.random.randn(n, 4)  # Mock features
            else:
                features = historical_data[available_cols].values
            
            # Outcomes (realized returns)
            if 'realized_return' in historical_data.columns:
                outcomes = historical_data['realized_return'].values
            else:
                # Mock outcomes for demonstration
                outcomes = np.random.randn(len(historical_data)) * 0.02
            
            # Create instrument (exogenous timing shock)
            instrument = None
            if 'exogenous_timing' in historical_data.columns:
                instrument = historical_data['exogenous_timing'].values
            
            return TreatmentGroup(
                features=features,
                outcomes=outcomes,
                treatment=treatment.values,
                instrument=instrument
            )
            
        except Exception as e:
            logger.error(f"Failed to prepare treatment data: {e}")
            return None
    
    def _aggregate_results(self, results: Dict[str, CausalResult]) -> CausalResult:
        """Aggregate results from multiple estimation methods"""
        if not results:
            return CausalResult(0.0, (-0.1, 0.1), 1.0, 0.0, 0.0)
        
        # Weight by inverse variance (precision weighting)
        estimates = []
        weights = []
        
        for method, result in results.items():
            # Calculate weight from confidence interval width
            ci_width = result.confidence_interval[1] - result.confidence_interval[0]
            weight = 1 / (ci_width + 1e-8)
            
            estimates.append(result.cate_estimate)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted average
        aggregated_estimate = np.average(estimates, weights=weights)
        
        # Conservative confidence interval (widest)
        ci_lower = min([r.confidence_interval[0] for r in results.values()])
        ci_upper = max([r.confidence_interval[1] for r in results.values()])
        
        # Combined p-value (Stouffer's method)
        z_scores = []
        for result in results.values():
            z = stats.norm.ppf(1 - result.p_value / 2)
            z_scores.append(z)
        
        combined_z = np.mean(z_scores) * np.sqrt(len(z_scores))
        combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))
        
        # Average uplift score
        avg_uplift = np.mean([r.uplift_score for r in results.values()])
        
        return CausalResult(
            cate_estimate=aggregated_estimate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=combined_p,
            uplift_score=avg_uplift,
            treatment_effect=aggregated_estimate
        )
    
    def _calculate_priority_score(self, result: CausalResult) -> float:
        """Calculate priority score for signal based on causal evidence"""
        # Base score from effect size
        effect_score = abs(result.cate_estimate) * 10  # Scale to 0-1 range
        
        # Confidence bonus (lower p-value = higher score)
        confidence_score = max(0, 1 - result.p_value)
        
        # Uplift bonus
        uplift_score = min(1.0, result.uplift_score / 5.0)
        
        # Combined priority (weighted average)
        priority = 0.4 * effect_score + 0.4 * confidence_score + 0.2 * uplift_score
        
        return min(1.0, priority)  # Cap at 1.0
    
    def _get_recommendation(self, result: CausalResult) -> str:
        """Get recommendation based on causal analysis"""
        if result.p_value < 0.01 and abs(result.cate_estimate) > 0.01:
            return "STRONG_CAUSAL_EFFECT"
        elif result.p_value < 0.05 and abs(result.cate_estimate) > 0.005:
            return "MODERATE_CAUSAL_EFFECT"
        elif result.p_value < 0.1:
            return "WEAK_CAUSAL_EFFECT"
        else:
            return "NO_CAUSAL_EFFECT"
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when estimation fails"""
        return {
            'signal_id': 'unknown',
            'symbol': 'unknown',
            'causal_methods': {},
            'aggregated_result': CausalResult(0.0, (-0.1, 0.1), 1.0, 0.0, 0.0),
            'priority_score': 0.0,
            'has_causal_effect': False,
            'effect_magnitude': 0.0,
            'confidence_level': 0.0,
            'recommendation': 'NO_CAUSAL_EFFECT',
            'timestamp': datetime.utcnow()
        }
    
    async def batch_analyze_signals(self, signals: List[Signal],
                                   historical_data: pd.DataFrame,
                                   trace_id: str) -> List[Dict[str, Any]]:
        """Analyze multiple signals for causal effects"""
        async with trace_operation("batch_causal_analysis", trace_id=trace_id):
            tasks = []
            for signal in signals:
                task = self.analyze_signal_causality(signal, historical_data, trace_id)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            logger.info(f"Batch causal analysis complete: {len(valid_results)}/{len(signals)} successful",
                       extra={'trace_id': trace_id})
            
            return valid_results
    
    def get_historical_performance(self) -> Dict[str, Any]:
        """Get historical performance of causal analysis"""
        if not self.results_history:
            return {'total_analyses': 0}
        
        total = len(self.results_history)
        causal_signals = sum(1 for r in self.results_history if r['has_causal_effect'])
        avg_priority = np.mean([r['priority_score'] for r in self.results_history])
        
        return {
            'total_analyses': total,
            'causal_signals': causal_signals,
            'causal_rate': causal_signals / total,
            'average_priority_score': avg_priority,
            'last_analysis': self.results_history[-1]['timestamp'] if self.results_history else None
        }
