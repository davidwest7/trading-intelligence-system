#!/usr/bin/env python3
"""
Advanced Off-Policy Evaluation (OPE) System
==========================================

Implements state-of-the-art off-policy evaluation methods for trading policies:
- Doubly Robust (DR-OPE) 
- Self-Normalized Importance Sampling (SNIPS)
- Fitted Q Evaluation (FQE)
- Live counterfactuals with safety filters
- Policy gradient off-policy evaluation

Key Features:
- Multiple OPE estimators with uncertainty quantification
- Online learning with live exploration budget
- Safety filters for exploration
- Policy performance tracking and comparison
- Automated A/B testing framework
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from enum import Enum

# ML and RL imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# Statistical and mathematical
from scipy import stats
from scipy.optimize import minimize
import cvxpy as cp

# Local imports
from schemas.contracts import Signal, Opportunity, Intent, DecisionLog, RegimeType
from common.observability.telemetry import get_telemetry, trace_operation

logger = logging.getLogger(__name__)

class OPEMethod(Enum):
    """Off-policy evaluation methods"""
    DIRECT_METHOD = "direct_method"
    IMPORTANCE_SAMPLING = "importance_sampling"
    DOUBLY_ROBUST = "doubly_robust"
    SNIPS = "snips"
    FITTED_Q = "fitted_q"
    REGRESSION_IS = "regression_is"

@dataclass
class PolicyExperience:
    """Single experience tuple for policy evaluation"""
    state: np.ndarray
    action: Any
    reward: float
    next_state: np.ndarray
    done: bool
    behavior_prob: float
    target_prob: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class OPEResult:
    """Result from off-policy evaluation"""
    method: OPEMethod
    value_estimate: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    bias_estimate: float
    variance_estimate: float
    sample_size: int
    convergence_achieved: bool
    metadata: Dict[str, Any]

@dataclass
class PolicyComparison:
    """Comparison between two policies"""
    policy_a_value: float
    policy_b_value: float
    difference: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significant: bool
    recommendation: str

class BaseOPEEstimator(ABC):
    """Base class for off-policy evaluation estimators"""
    
    @abstractmethod
    async def evaluate(self, experiences: List[PolicyExperience], 
                      trace_id: str) -> OPEResult:
        """Evaluate policy using off-policy data"""
        pass
    
    @abstractmethod
    def get_method(self) -> OPEMethod:
        """Get the OPE method type"""
        pass

class DoublyRobustEstimator(BaseOPEEstimator):
    """
    Doubly Robust Off-Policy Evaluation
    
    Combines direct method and importance sampling for more robust estimates.
    Less biased than either method alone.
    """
    
    def __init__(self, q_model=None, pi_model=None):
        self.q_model = q_model or lgb.LGBMRegressor(n_estimators=100, verbose=-1)
        self.pi_model = pi_model or lgb.LGBMRegressor(n_estimators=100, verbose=-1)
        self.is_fitted = False
        
    def get_method(self) -> OPEMethod:
        return OPEMethod.DOUBLY_ROBUST
    
    async def evaluate(self, experiences: List[PolicyExperience], 
                      trace_id: str) -> OPEResult:
        """Evaluate using doubly robust estimation"""
        async with trace_operation("dr_ope_evaluation", trace_id=trace_id):
            try:
                if len(experiences) < 50:
                    logger.warning("Insufficient data for DR-OPE", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # Prepare data
                states = np.array([exp.state for exp in experiences])
                actions = np.array([exp.action for exp in experiences])
                rewards = np.array([exp.reward for exp in experiences])
                next_states = np.array([exp.next_state for exp in experiences])
                behavior_probs = np.array([exp.behavior_prob for exp in experiences])
                target_probs = np.array([exp.target_prob for exp in experiences])
                
                # Train Q-function model
                # Augment states with actions for Q(s,a)
                if actions.ndim == 1:
                    actions_reshaped = actions.reshape(-1, 1)
                else:
                    actions_reshaped = actions
                
                state_action_features = np.column_stack([states, actions_reshaped])
                
                # Split data for cross-validation
                train_idx, val_idx = train_test_split(
                    range(len(experiences)), test_size=0.3, random_state=42
                )
                
                # Train Q-function with feature names
                feature_names = [f'feature_{i}' for i in range(state_action_features.shape[1])]
                state_action_df = pd.DataFrame(state_action_features, columns=feature_names)
                
                self.q_model.fit(
                    state_action_df.iloc[train_idx], 
                    rewards[train_idx]
                )
                
                # Train policy probability model (for propensity scores)
                # This is simplified - in practice you'd have a more sophisticated model
                self.pi_model.fit(states[train_idx], target_probs[train_idx])
                
                # Predict Q-values for validation set
                q_predictions = self.q_model.predict(state_action_df.iloc[val_idx])
                
                # Predict target policy probabilities
                state_feature_names = [f'state_feature_{i}' for i in range(states.shape[1])]
                states_df = pd.DataFrame(states, columns=state_feature_names)
                pi_predictions = self.pi_model.predict(states_df.iloc[val_idx])
                pi_predictions = np.clip(pi_predictions, 1e-8, 1.0)  # Avoid division by zero
                
                # Calculate importance weights
                importance_weights = pi_predictions / (behavior_probs[val_idx] + 1e-8)
                importance_weights = np.clip(importance_weights, 0, 10)  # Cap weights
                
                # Doubly robust estimation
                # V^{DR} = Q(s,a) + w(s,a) * (r - Q(s,a))
                dr_estimates = (
                    q_predictions + 
                    importance_weights * (rewards[val_idx] - q_predictions)
                )
                
                # Calculate final estimate
                value_estimate = np.mean(dr_estimates)
                standard_error = np.std(dr_estimates) / np.sqrt(len(dr_estimates))
                
                # Confidence interval
                ci_lower = value_estimate - 1.96 * standard_error
                ci_upper = value_estimate + 1.96 * standard_error
                
                # Bias and variance estimates
                dm_estimate = np.mean(q_predictions)  # Direct method
                is_estimate = np.mean(importance_weights * rewards[val_idx])  # IS
                
                bias_estimate = abs(value_estimate - dm_estimate)
                variance_estimate = np.var(dr_estimates)
                
                # Check convergence (simplified)
                convergence_achieved = standard_error < 0.01
                
                self.is_fitted = True
                
                logger.info(f"DR-OPE estimate: {value_estimate:.4f} ± {standard_error:.4f}",
                           extra={'trace_id': trace_id})
                
                return OPEResult(
                    method=self.get_method(),
                    value_estimate=value_estimate,
                    confidence_interval=(ci_lower, ci_upper),
                    standard_error=standard_error,
                    bias_estimate=bias_estimate,
                    variance_estimate=variance_estimate,
                    sample_size=len(val_idx),
                    convergence_achieved=convergence_achieved,
                    metadata={
                        'direct_method_estimate': dm_estimate,
                        'importance_sampling_estimate': is_estimate,
                        'mean_importance_weight': np.mean(importance_weights),
                        'max_importance_weight': np.max(importance_weights)
                    }
                )
                
            except Exception as e:
                logger.error(f"DR-OPE evaluation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _get_default_result(self) -> OPEResult:
        """Return default result when evaluation fails"""
        return OPEResult(
            method=self.get_method(),
            value_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            standard_error=0.1,
            bias_estimate=0.0,
            variance_estimate=0.01,
            sample_size=0,
            convergence_achieved=False,
            metadata={'error': True}
        )

class SNIPSEstimator(BaseOPEEstimator):
    """
    Self-Normalized Importance Sampling (SNIPS)
    
    More stable than standard importance sampling by normalizing weights.
    Reduces variance at the cost of some bias.
    """
    
    def __init__(self):
        self.is_fitted = False
        
    def get_method(self) -> OPEMethod:
        return OPEMethod.SNIPS
    
    async def evaluate(self, experiences: List[PolicyExperience], 
                      trace_id: str) -> OPEResult:
        """Evaluate using SNIPS"""
        async with trace_operation("snips_evaluation", trace_id=trace_id):
            try:
                if len(experiences) < 20:
                    logger.warning("Insufficient data for SNIPS", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # Extract data
                rewards = np.array([exp.reward for exp in experiences])
                behavior_probs = np.array([exp.behavior_prob for exp in experiences])
                target_probs = np.array([exp.target_prob for exp in experiences])
                
                # Calculate importance weights
                importance_weights = target_probs / (behavior_probs + 1e-8)
                
                # Cap extreme weights to reduce variance
                weight_cap = np.percentile(importance_weights, 95)
                importance_weights = np.clip(importance_weights, 0, weight_cap)
                
                # Self-normalized importance sampling
                # V^{SNIPS} = (Σ w_i * r_i) / (Σ w_i)
                weighted_rewards = importance_weights * rewards
                weight_sum = np.sum(importance_weights)
                
                if weight_sum < 1e-8:
                    logger.warning("Sum of importance weights too small", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                value_estimate = np.sum(weighted_rewards) / weight_sum
                
                # Standard error estimation (approximate)
                # Using delta method approximation
                n = len(experiences)
                w_bar = weight_sum / n
                
                if w_bar > 0:
                    # Variance estimation for SNIPS
                    numerator_var = np.var(weighted_rewards)
                    denominator_var = np.var(importance_weights)
                    covariance = np.cov(weighted_rewards, importance_weights)[0, 1]
                    
                    # Delta method variance
                    variance = (1 / (w_bar**2)) * (
                        numerator_var / n - 
                        2 * value_estimate * covariance / n + 
                        (value_estimate**2) * denominator_var / n
                    )
                    
                    standard_error = np.sqrt(max(0, variance))
                else:
                    standard_error = 0.1
                
                # Confidence interval
                ci_lower = value_estimate - 1.96 * standard_error
                ci_upper = value_estimate + 1.96 * standard_error
                
                # Bias estimate (difference from standard IS)
                is_estimate = np.mean(importance_weights * rewards) if np.mean(importance_weights) > 0 else 0.0
                bias_estimate = abs(value_estimate - is_estimate)
                
                variance_estimate = variance if 'variance' in locals() else standard_error**2
                
                # Check convergence
                convergence_achieved = standard_error < 0.01 and n > 100
                
                self.is_fitted = True
                
                logger.info(f"SNIPS estimate: {value_estimate:.4f} ± {standard_error:.4f}",
                           extra={'trace_id': trace_id})
                
                return OPEResult(
                    method=self.get_method(),
                    value_estimate=value_estimate,
                    confidence_interval=(ci_lower, ci_upper),
                    standard_error=standard_error,
                    bias_estimate=bias_estimate,
                    variance_estimate=variance_estimate,
                    sample_size=n,
                    convergence_achieved=convergence_achieved,
                    metadata={
                        'mean_importance_weight': np.mean(importance_weights),
                        'max_importance_weight': np.max(importance_weights),
                        'weight_cap': weight_cap,
                        'effective_sample_size': weight_sum**2 / np.sum(importance_weights**2)
                    }
                )
                
            except Exception as e:
                logger.error(f"SNIPS evaluation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _get_default_result(self) -> OPEResult:
        """Return default result when evaluation fails"""
        return OPEResult(
            method=self.get_method(),
            value_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            standard_error=0.1,
            bias_estimate=0.0,
            variance_estimate=0.01,
            sample_size=0,
            convergence_achieved=False,
            metadata={'error': True}
        )

class FittedQEstimator(BaseOPEEstimator):
    """
    Fitted Q Evaluation (FQE)
    
    Iteratively fits Q-function using Bellman equation.
    Good for sequential decision making problems.
    """
    
    def __init__(self, gamma: float = 0.99, max_iterations: int = 50):
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.q_model = None
        self.is_fitted = False
        
    def get_method(self) -> OPEMethod:
        return OPEMethod.FITTED_Q
    
    async def evaluate(self, experiences: List[PolicyExperience], 
                      trace_id: str) -> OPEResult:
        """Evaluate using Fitted Q Evaluation"""
        async with trace_operation("fqe_evaluation", trace_id=trace_id):
            try:
                if len(experiences) < 100:
                    logger.warning("Insufficient data for FQE", extra={'trace_id': trace_id})
                    return self._get_default_result()
                
                # Prepare data
                states = np.array([exp.state for exp in experiences])
                actions = np.array([exp.action for exp in experiences])
                rewards = np.array([exp.reward for exp in experiences])
                next_states = np.array([exp.next_state for exp in experiences])
                dones = np.array([exp.done for exp in experiences])
                target_probs = np.array([exp.target_prob for exp in experiences])
                
                # Handle action dimensions
                if actions.ndim == 1:
                    actions = actions.reshape(-1, 1)
                
                # Create state-action features
                state_action_features = np.column_stack([states, actions])
                
                # Initialize Q-function
                self.q_model = lgb.LGBMRegressor(
                    n_estimators=100, 
                    learning_rate=0.1,
                    verbose=-1,
                    random_state=42
                )
                
                # Fitted Q Iteration
                q_values = np.zeros(len(experiences))
                prev_q_values = None
                
                for iteration in range(self.max_iterations):
                    # Compute target Q-values
                    if iteration == 0:
                        # Initialize with immediate rewards
                        targets = rewards.copy()
                    else:
                        # Bellman backup: r + γ * max_a Q(s', a)
                        # For simplicity, assume next action follows target policy
                        next_q_values = self.q_model.predict(
                            np.column_stack([next_states, actions])  # Simplified
                        )
                        targets = rewards + self.gamma * (1 - dones) * next_q_values
                    
                    # Fit Q-function with feature names
                    state_action_df = pd.DataFrame(state_action_features, columns=feature_names)
                    self.q_model.fit(state_action_df, targets)
                    
                    # Check convergence
                    current_q_values = self.q_model.predict(state_action_df)
                    
                    if prev_q_values is not None:
                        q_diff = np.mean(np.abs(current_q_values - prev_q_values))
                        if q_diff < 1e-4:
                            logger.info(f"FQE converged at iteration {iteration}", 
                                       extra={'trace_id': trace_id})
                            break
                    
                    prev_q_values = current_q_values.copy()
                
                # Final Q-values
                final_q_values = self.q_model.predict(state_action_df)
                
                # Policy value estimate
                value_estimate = np.mean(final_q_values)
                standard_error = np.std(final_q_values) / np.sqrt(len(final_q_values))
                
                # Confidence interval
                ci_lower = value_estimate - 1.96 * standard_error
                ci_upper = value_estimate + 1.96 * standard_error
                
                # Bias and variance estimates
                variance_estimate = np.var(final_q_values)
                
                # Estimate bias using cross-validation
                bias_estimate = self._estimate_bias_cv(
                    state_action_features, rewards, next_states, dones
                )
                
                convergence_achieved = iteration < self.max_iterations - 1
                
                self.is_fitted = True
                
                logger.info(f"FQE estimate: {value_estimate:.4f} ± {standard_error:.4f}, "
                           f"iterations: {iteration + 1}", extra={'trace_id': trace_id})
                
                return OPEResult(
                    method=self.get_method(),
                    value_estimate=value_estimate,
                    confidence_interval=(ci_lower, ci_upper),
                    standard_error=standard_error,
                    bias_estimate=bias_estimate,
                    variance_estimate=variance_estimate,
                    sample_size=len(experiences),
                    convergence_achieved=convergence_achieved,
                    metadata={
                        'iterations': iteration + 1,
                        'gamma': self.gamma,
                        'final_q_range': (np.min(final_q_values), np.max(final_q_values))
                    }
                )
                
            except Exception as e:
                logger.error(f"FQE evaluation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_result()
    
    def _estimate_bias_cv(self, state_action_features: np.ndarray, 
                         rewards: np.ndarray, next_states: np.ndarray,
                         dones: np.ndarray) -> float:
        """Estimate bias using cross-validation"""
        try:
            # Simple cross-validation to estimate bias
            train_idx, val_idx = train_test_split(
                range(len(rewards)), test_size=0.3, random_state=42
            )
            
            # Train on subset
            temp_model = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
            temp_model.fit(state_action_features[train_idx], rewards[train_idx])
            
            # Predict on validation set
            val_predictions = temp_model.predict(state_action_features[val_idx])
            val_actual = rewards[val_idx]
            
            # Bias is systematic error
            bias = np.mean(val_predictions - val_actual)
            return abs(bias)
            
        except Exception:
            return 0.0
    
    def _get_default_result(self) -> OPEResult:
        """Return default result when evaluation fails"""
        return OPEResult(
            method=self.get_method(),
            value_estimate=0.0,
            confidence_interval=(-0.1, 0.1),
            standard_error=0.1,
            bias_estimate=0.0,
            variance_estimate=0.01,
            sample_size=0,
            convergence_achieved=False,
            metadata={'error': True}
        )

class LiveCounterfactualSystem:
    """
    Live counterfactual evaluation system
    
    Manages live exploration budget and safety filters
    for online policy evaluation and improvement.
    """
    
    def __init__(self, exploration_budget: float = 0.05, safety_threshold: float = 0.1):
        self.exploration_budget = exploration_budget  # Fraction of decisions for exploration
        self.safety_threshold = safety_threshold  # Maximum allowed loss
        self.exploration_history = []
        self.safety_violations = 0
        self.current_exploration_rate = 0.0
        
    async def should_explore(self, state: np.ndarray, trace_id: str) -> bool:
        """Decide whether to explore or exploit"""
        async with trace_operation("exploration_decision", trace_id=trace_id):
            try:
                # Check safety constraints
                if self.safety_violations > 10:  # Too many safety violations
                    logger.warning("Exploration paused due to safety violations", 
                                 extra={'trace_id': trace_id})
                    return False
                
                # Check exploration budget
                recent_exploration_rate = self._get_recent_exploration_rate()
                if recent_exploration_rate >= self.exploration_budget:
                    return False
                
                # Safety filter: only explore in "safe" states
                if not self._is_safe_state(state):
                    return False
                
                # Adaptive exploration based on uncertainty
                uncertainty = self._estimate_state_uncertainty(state)
                exploration_prob = min(self.exploration_budget, uncertainty)
                
                should_explore = np.random.random() < exploration_prob
                
                if should_explore:
                    self.current_exploration_rate = recent_exploration_rate + 1/100  # Approximate
                
                logger.debug(f"Exploration decision: {should_explore}, rate: {recent_exploration_rate:.3f}",
                           extra={'trace_id': trace_id})
                
                return should_explore
                
            except Exception as e:
                logger.error(f"Exploration decision failed: {e}", extra={'trace_id': trace_id})
                return False
    
    def record_exploration_result(self, state: np.ndarray, action: Any, 
                                reward: float, trace_id: str):
        """Record result of exploration action"""
        exploration_record = {
            'timestamp': datetime.utcnow(),
            'state_hash': hash(str(state.tobytes())),
            'action': action,
            'reward': reward,
            'safe': reward > -self.safety_threshold,
            'trace_id': trace_id
        }
        
        self.exploration_history.append(exploration_record)
        
        # Check safety violation
        if reward < -self.safety_threshold:
            self.safety_violations += 1
            logger.warning(f"Safety violation detected: reward={reward:.4f}", 
                         extra={'trace_id': trace_id})
        
        # Limit history size
        if len(self.exploration_history) > 10000:
            self.exploration_history = self.exploration_history[-10000:]
    
    def _get_recent_exploration_rate(self, hours: int = 24) -> float:
        """Get exploration rate in recent hours"""
        if not self.exploration_history:
            return 0.0
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_explorations = [
            exp for exp in self.exploration_history 
            if exp['timestamp'] > cutoff_time
        ]
        
        # Approximate total decisions (this would be tracked separately in practice)
        total_decisions = len(recent_explorations) / self.exploration_budget if self.exploration_budget > 0 else 1000
        
        return len(recent_explorations) / max(1, total_decisions)
    
    def _is_safe_state(self, state: np.ndarray) -> bool:
        """Determine if state is safe for exploration"""
        # Simplified safety check - in practice this would be more sophisticated
        # E.g., check volatility, market conditions, position size, etc.
        
        # Assume first feature is volatility indicator
        if len(state) > 0:
            volatility_indicator = state[0]
            return volatility_indicator < 2.0  # Arbitrary threshold
        
        return True
    
    def _estimate_state_uncertainty(self, state: np.ndarray) -> float:
        """Estimate uncertainty in current state"""
        # Simplified uncertainty estimation
        # In practice, this could use ensemble disagreement, epistemic uncertainty, etc.
        
        # Use state magnitude as proxy for uncertainty
        state_magnitude = np.linalg.norm(state)
        uncertainty = min(1.0, state_magnitude / 10.0)  # Normalize
        
        return uncertainty
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        if not self.exploration_history:
            return {
                'total_explorations': 0,
                'recent_exploration_rate': 0.0,
                'safety_violations': self.safety_violations,
                'average_exploration_reward': 0.0
            }
        
        recent_rate = self._get_recent_exploration_rate()
        avg_reward = np.mean([exp['reward'] for exp in self.exploration_history])
        safe_rate = np.mean([exp['safe'] for exp in self.exploration_history])
        
        return {
            'total_explorations': len(self.exploration_history),
            'recent_exploration_rate': recent_rate,
            'safety_violations': self.safety_violations,
            'average_exploration_reward': avg_reward,
            'safety_rate': safe_rate,
            'current_budget_utilization': recent_rate / self.exploration_budget
        }

class AdvancedOPEManager:
    """
    Main manager for advanced off-policy evaluation
    
    Coordinates multiple OPE methods, manages live counterfactuals,
    and provides policy comparison and A/B testing capabilities.
    """
    
    def __init__(self):
        self.estimators = {
            'doubly_robust': DoublyRobustEstimator(),
            'snips': SNIPSEstimator(),
            'fitted_q': FittedQEstimator()
        }
        
        self.counterfactual_system = LiveCounterfactualSystem()
        self.evaluation_history = []
        self.policy_comparisons = []
        
    async def comprehensive_evaluation(self, experiences: List[PolicyExperience],
                                     trace_id: str) -> Dict[str, Any]:
        """Run comprehensive OPE evaluation using multiple methods"""
        async with trace_operation("comprehensive_ope", trace_id=trace_id):
            try:
                results = {}
                
                # Run all OPE methods
                for name, estimator in self.estimators.items():
                    result = await estimator.evaluate(experiences, trace_id)
                    results[name] = result
                
                # Aggregate results
                aggregated = self._aggregate_ope_results(results)
                
                # Calculate ensemble estimate
                ensemble_estimate = self._calculate_ensemble_estimate(results)
                
                evaluation = {
                    'individual_methods': results,
                    'aggregated_result': aggregated,
                    'ensemble_estimate': ensemble_estimate,
                    'sample_size': len(experiences),
                    'evaluation_timestamp': datetime.utcnow(),
                    'convergence_status': self._check_convergence(results),
                    'recommendation': self._get_evaluation_recommendation(aggregated)
                }
                
                # Store for history
                self.evaluation_history.append(evaluation)
                
                logger.info(f"Comprehensive OPE complete. Ensemble estimate: {ensemble_estimate:.4f}",
                           extra={'trace_id': trace_id})
                
                return evaluation
                
            except Exception as e:
                logger.error(f"Comprehensive OPE evaluation failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_evaluation()
    
    async def compare_policies(self, policy_a_experiences: List[PolicyExperience],
                              policy_b_experiences: List[PolicyExperience],
                              trace_id: str) -> PolicyComparison:
        """Compare two policies using OPE"""
        async with trace_operation("policy_comparison", trace_id=trace_id):
            try:
                # Evaluate both policies
                eval_a = await self.comprehensive_evaluation(policy_a_experiences, trace_id)
                eval_b = await self.comprehensive_evaluation(policy_b_experiences, trace_id)
                
                value_a = eval_a['ensemble_estimate']
                value_b = eval_b['ensemble_estimate']
                
                # Statistical significance test
                difference = value_a - value_b
                
                # Approximate standard error of difference
                se_a = eval_a['aggregated_result'].standard_error
                se_b = eval_b['aggregated_result'].standard_error
                se_diff = np.sqrt(se_a**2 + se_b**2)
                
                # T-test
                t_stat = difference / (se_diff + 1e-8)
                df = len(policy_a_experiences) + len(policy_b_experiences) - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                # Confidence interval for difference
                margin = 1.96 * se_diff
                ci_lower = difference - margin
                ci_upper = difference + margin
                
                significant = p_value < 0.05
                
                # Recommendation
                if significant:
                    if difference > 0:
                        recommendation = "PREFER_POLICY_A"
                    else:
                        recommendation = "PREFER_POLICY_B"
                else:
                    recommendation = "NO_SIGNIFICANT_DIFFERENCE"
                
                comparison = PolicyComparison(
                    policy_a_value=value_a,
                    policy_b_value=value_b,
                    difference=difference,
                    confidence_interval=(ci_lower, ci_upper),
                    p_value=p_value,
                    significant=significant,
                    recommendation=recommendation
                )
                
                self.policy_comparisons.append({
                    'comparison': comparison,
                    'timestamp': datetime.utcnow(),
                    'trace_id': trace_id
                })
                
                logger.info(f"Policy comparison: A={value_a:.4f}, B={value_b:.4f}, "
                           f"diff={difference:.4f}, p={p_value:.4f}",
                           extra={'trace_id': trace_id})
                
                return comparison
                
            except Exception as e:
                logger.error(f"Policy comparison failed: {e}", extra={'trace_id': trace_id})
                return PolicyComparison(0.0, 0.0, 0.0, (-0.1, 0.1), 1.0, False, "ERROR")
    
    def _aggregate_ope_results(self, results: Dict[str, OPEResult]) -> OPEResult:
        """Aggregate results from multiple OPE methods"""
        if not results:
            return OPEResult(
                OPEMethod.DOUBLY_ROBUST, 0.0, (-0.1, 0.1), 0.1, 0.0, 0.01, 0, False, {}
            )
        
        # Weight by inverse variance (precision weighting)
        weights = []
        estimates = []
        
        for method, result in results.items():
            if result.variance_estimate > 0:
                weight = 1 / result.variance_estimate
            else:
                weight = 1.0
            
            weights.append(weight)
            estimates.append(result.value_estimate)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted average
        aggregated_estimate = np.average(estimates, weights=weights)
        
        # Conservative confidence interval (widest)
        ci_lower = min([r.confidence_interval[0] for r in results.values()])
        ci_upper = max([r.confidence_interval[1] for r in results.values()])
        
        # Average standard error
        avg_se = np.mean([r.standard_error for r in results.values()])
        
        # Average bias and variance
        avg_bias = np.mean([r.bias_estimate for r in results.values()])
        avg_variance = np.mean([r.variance_estimate for r in results.values()])
        
        # Total sample size
        total_samples = max([r.sample_size for r in results.values()])
        
        # Convergence if all methods converged
        all_converged = all([r.convergence_achieved for r in results.values()])
        
        return OPEResult(
            method=OPEMethod.DOUBLY_ROBUST,  # Representative method
            value_estimate=aggregated_estimate,
            confidence_interval=(ci_lower, ci_upper),
            standard_error=avg_se,
            bias_estimate=avg_bias,
            variance_estimate=avg_variance,
            sample_size=total_samples,
            convergence_achieved=all_converged,
            metadata={'aggregation_weights': weights.tolist(), 'methods_used': list(results.keys())}
        )
    
    def _calculate_ensemble_estimate(self, results: Dict[str, OPEResult]) -> float:
        """Calculate ensemble estimate with method-specific weights"""
        if not results:
            return 0.0
        
        # Method-specific weights based on theoretical properties
        method_weights = {
            'doubly_robust': 0.4,  # Most robust
            'snips': 0.3,          # Good variance reduction
            'fitted_q': 0.3        # Good for sequential problems
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, result in results.items():
            weight = method_weights.get(method, 0.1)
            
            # Adjust weight by convergence and sample size
            if result.convergence_achieved:
                weight *= 1.2
            if result.sample_size > 100:
                weight *= 1.1
            
            weighted_sum += weight * result.value_estimate
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _check_convergence(self, results: Dict[str, OPEResult]) -> Dict[str, Any]:
        """Check convergence status across methods"""
        converged_methods = [
            method for method, result in results.items() 
            if result.convergence_achieved
        ]
        
        convergence_rate = len(converged_methods) / len(results) if results else 0.0
        
        return {
            'overall_converged': convergence_rate >= 0.5,
            'convergence_rate': convergence_rate,
            'converged_methods': converged_methods,
            'min_sample_size': min([r.sample_size for r in results.values()]) if results else 0
        }
    
    def _get_evaluation_recommendation(self, aggregated_result: OPEResult) -> str:
        """Get recommendation based on evaluation results"""
        if not aggregated_result.convergence_achieved:
            return "COLLECT_MORE_DATA"
        elif aggregated_result.standard_error > 0.05:
            return "HIGH_UNCERTAINTY"
        elif aggregated_result.value_estimate > 0.01:
            return "POLICY_PERFORMING_WELL"
        elif aggregated_result.value_estimate < -0.01:
            return "POLICY_UNDERPERFORMING"
        else:
            return "POLICY_NEUTRAL"
    
    def _get_default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation when evaluation fails"""
        return {
            'individual_methods': {},
            'aggregated_result': OPEResult(
                OPEMethod.DOUBLY_ROBUST, 0.0, (-0.1, 0.1), 0.1, 0.0, 0.01, 0, False, {}
            ),
            'ensemble_estimate': 0.0,
            'sample_size': 0,
            'evaluation_timestamp': datetime.utcnow(),
            'convergence_status': {'overall_converged': False, 'convergence_rate': 0.0},
            'recommendation': 'EVALUATION_FAILED'
        }
    
    def get_historical_performance(self) -> Dict[str, Any]:
        """Get historical OPE performance"""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        recent_evaluations = self.evaluation_history[-10:]  # Last 10
        
        avg_ensemble_estimate = np.mean([
            eval['ensemble_estimate'] for eval in recent_evaluations
        ])
        
        convergence_rate = np.mean([
            eval['convergence_status']['overall_converged'] for eval in recent_evaluations
        ])
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'recent_average_estimate': avg_ensemble_estimate,
            'convergence_rate': convergence_rate,
            'total_policy_comparisons': len(self.policy_comparisons),
            'exploration_stats': self.counterfactual_system.get_exploration_stats()
        }
