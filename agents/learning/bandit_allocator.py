#!/usr/bin/env python3
"""
Contextual Bandit Allocator
LinUCB, Thompson Sampling, and Bayesian budget allocators for gating agents by regime/state
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json
from scipy import stats
from scipy.linalg import inv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BanditConfig:
    """Configuration for bandit allocators"""
    n_agents: int = 8  # Number of agents to allocate across
    alpha: float = 0.1  # Confidence parameter for LinUCB
    lambda_reg: float = 1.0  # Regularization parameter
    exploration_bonus: float = 0.1  # Exploration bonus
    context_dim: int = 10  # Dimensionality of context features
    learning_rate: float = 0.01  # Learning rate for gradient-based methods
    budget_constraint: float = 1.0  # Total budget constraint
    min_allocation: float = 0.01  # Minimum allocation per agent
    decay_factor: float = 0.99  # Decay factor for historical weights
    update_frequency: int = 100  # Update frequency in iterations


@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent_id: str
    agent_name: str
    allocation: float
    context: np.ndarray
    timestamp: datetime
    expected_reward: float
    confidence: float


@dataclass
class BanditUpdate:
    """Update for bandit algorithm"""
    agent_id: str
    context: np.ndarray
    reward: float
    timestamp: datetime
    additional_info: Dict[str, Any]


@dataclass
class AllocationResult:
    """Result of budget allocation"""
    allocations: Dict[str, float]
    expected_rewards: Dict[str, float]
    uncertainties: Dict[str, float]
    total_expected_reward: float
    exploration_bonus: float
    timestamp: datetime


class ContextFeatureExtractor:
    """Extract context features for bandit algorithms"""
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, market_data: pd.DataFrame, 
                        regime_info: Dict[str, Any],
                        portfolio_state: Dict[str, Any]) -> np.ndarray:
        """Extract context features for bandit decision making"""
        try:
            features = []
            
            # Market features
            if len(market_data) > 0:
                returns = market_data['close'].pct_change().dropna()
                
                # Volatility features
                vol_5d = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0.2
                vol_21d = returns.tail(21).std() * np.sqrt(252) if len(returns) >= 21 else 0.2
                vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.0
                
                # Momentum features  
                mom_5d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5] - 1) if len(market_data) >= 5 else 0.0
                mom_21d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-21] - 1) if len(market_data) >= 21 else 0.0
                
                # Trend features
                price_trend = np.polyfit(range(min(21, len(market_data))), 
                                       market_data['close'].tail(min(21, len(market_data))), 1)[0] if len(market_data) > 1 else 0.0
                
                features.extend([vol_5d, vol_21d, vol_ratio, mom_5d, mom_21d, price_trend])
            else:
                features.extend([0.2, 0.2, 1.0, 0.0, 0.0, 0.0])
            
            # Regime features
            regime_features = [
                regime_info.get('volatility_percentile', 0.5),
                regime_info.get('trend_strength', 0.0),
                regime_info.get('correlation_level', 0.5),
                regime_info.get('risk_on_factor', 0.5)
            ]
            features.extend(regime_features)
            
            # Portfolio state features
            portfolio_features = [
                portfolio_state.get('total_pnl', 0.0),
                portfolio_state.get('sharpe_ratio', 0.0),
                portfolio_state.get('max_drawdown', 0.0),
                portfolio_state.get('positions_count', 0.0),
                portfolio_state.get('cash_ratio', 1.0)
            ]
            features.extend(portfolio_features)
            
            # Pad or truncate to target dimension
            features = np.array(features, dtype=float)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            elif len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            
            # Scale features if fitted
            if self.is_fitted:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            else:
                # Fit scaler on first use (with some synthetic data for stability)
                synthetic_features = np.random.randn(100, self.feature_dim)
                synthetic_features[0] = features
                self.scaler.fit(synthetic_features)
                self.is_fitted = True
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim)


class BanditAllocator(ABC):
    """Base class for bandit allocators"""
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.agent_ids = [f"agent_{i}" for i in range(config.n_agents)]
        self.agent_names = ["technical", "flow", "sentiment", "macro", "undervalued", "insider", "learning", "causal"][:config.n_agents]
        self.logger = logging.getLogger(__name__)
        self.iteration = 0
        self.history = []
    
    @abstractmethod
    async def select_allocations(self, context: np.ndarray) -> AllocationResult:
        """Select allocations for agents"""
        pass
    
    @abstractmethod
    async def update(self, update: BanditUpdate) -> None:
        """Update bandit with reward feedback"""
        pass
    
    def _normalize_allocations(self, raw_allocations: Dict[str, float]) -> Dict[str, float]:
        """Normalize allocations to satisfy budget constraint"""
        total = sum(raw_allocations.values())
        if total <= 0:
            # Equal allocation fallback
            equal_alloc = self.config.budget_constraint / len(raw_allocations)
            return {agent_id: equal_alloc for agent_id in raw_allocations.keys()}
        
        # Scale to budget constraint
        normalized = {}
        for agent_id, allocation in raw_allocations.items():
            normalized[agent_id] = max(
                self.config.min_allocation,
                (allocation / total) * self.config.budget_constraint
            )
        
        # Renormalize after applying minimum constraints
        total_after_min = sum(normalized.values())
        if total_after_min > self.config.budget_constraint:
            excess = total_after_min - self.config.budget_constraint
            # Proportionally reduce allocations above minimum
            reducible_total = sum(max(0, alloc - self.config.min_allocation) for alloc in normalized.values())
            if reducible_total > 0:
                for agent_id in normalized:
                    reducible_amount = max(0, normalized[agent_id] - self.config.min_allocation)
                    reduction = (reducible_amount / reducible_total) * excess
                    normalized[agent_id] -= reduction
        
        return normalized


class LinUCBAllocator(BanditAllocator):
    """Linear Upper Confidence Bound allocator"""
    
    def __init__(self, config: BanditConfig):
        super().__init__(config)
        
        # Initialize parameters for each agent
        self.A = {}  # Design matrices
        self.b = {}  # Reward vectors
        self.theta = {}  # Parameter estimates
        
        for agent_id in self.agent_ids:
            self.A[agent_id] = np.eye(config.context_dim) * config.lambda_reg
            self.b[agent_id] = np.zeros(config.context_dim)
            self.theta[agent_id] = np.zeros(config.context_dim)
        
        self.logger.info(f"Initialized LinUCB allocator with {config.n_agents} agents")
    
    async def select_allocations(self, context: np.ndarray) -> AllocationResult:
        """Select allocations using LinUCB algorithm"""
        try:
            raw_allocations = {}
            expected_rewards = {}
            uncertainties = {}
            
            for i, agent_id in enumerate(self.agent_ids):
                agent_name = self.agent_names[i] if i < len(self.agent_names) else f"agent_{i}"
                
                # Update parameter estimate
                try:
                    A_inv = inv(self.A[agent_id])
                    self.theta[agent_id] = A_inv @ self.b[agent_id]
                except np.linalg.LinAlgError:
                    # Fallback to pseudoinverse
                    self.theta[agent_id] = np.linalg.pinv(self.A[agent_id]) @ self.b[agent_id]
                    A_inv = np.linalg.pinv(self.A[agent_id])
                
                # Calculate expected reward and confidence interval
                expected_reward = context @ self.theta[agent_id]
                confidence_radius = self.config.alpha * np.sqrt(context @ A_inv @ context)
                
                # UCB score (upper confidence bound)
                ucb_score = expected_reward + confidence_radius
                
                expected_rewards[agent_id] = float(expected_reward)
                uncertainties[agent_id] = float(confidence_radius)
                raw_allocations[agent_id] = max(0, ucb_score)  # Ensure non-negative
            
            # Normalize allocations
            normalized_allocations = self._normalize_allocations(raw_allocations)
            
            # Calculate total expected reward
            total_expected_reward = sum(
                normalized_allocations[agent_id] * expected_rewards[agent_id]
                for agent_id in self.agent_ids
            )
            
            # Calculate exploration bonus
            exploration_bonus = sum(
                normalized_allocations[agent_id] * uncertainties[agent_id]
                for agent_id in self.agent_ids
            )
            
            self.iteration += 1
            
            return AllocationResult(
                allocations=normalized_allocations,
                expected_rewards=expected_rewards,
                uncertainties=uncertainties,
                total_expected_reward=total_expected_reward,
                exploration_bonus=exploration_bonus,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in LinUCB allocation: {e}")
            # Fallback to equal allocation
            equal_alloc = self.config.budget_constraint / len(self.agent_ids)
            return AllocationResult(
                allocations={agent_id: equal_alloc for agent_id in self.agent_ids},
                expected_rewards={agent_id: 0.0 for agent_id in self.agent_ids},
                uncertainties={agent_id: 1.0 for agent_id in self.agent_ids},
                total_expected_reward=0.0,
                exploration_bonus=0.0,
                timestamp=datetime.now()
            )
    
    async def update(self, update: BanditUpdate) -> None:
        """Update LinUCB parameters with reward feedback"""
        try:
            agent_id = update.agent_id
            if agent_id in self.A:
                # Update design matrix and reward vector
                self.A[agent_id] += np.outer(update.context, update.context)
                self.b[agent_id] += update.reward * update.context
                
                self.logger.debug(f"Updated LinUCB for agent {agent_id} with reward {update.reward:.4f}")
            else:
                self.logger.warning(f"Unknown agent ID for update: {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Error updating LinUCB: {e}")


class ThompsonSamplingAllocator(BanditAllocator):
    """Thompson Sampling allocator with Bayesian updates"""
    
    def __init__(self, config: BanditConfig):
        super().__init__(config)
        
        # Initialize Bayesian parameters for each agent
        self.prior_precision = config.lambda_reg
        self.noise_precision = 1.0
        
        self.S = {}  # Precision matrices
        self.mu = {}  # Mean vectors
        
        for agent_id in self.agent_ids:
            self.S[agent_id] = np.eye(config.context_dim) * self.prior_precision
            self.mu[agent_id] = np.zeros(config.context_dim)
        
        self.logger.info(f"Initialized Thompson Sampling allocator with {config.n_agents} agents")
    
    async def select_allocations(self, context: np.ndarray) -> AllocationResult:
        """Select allocations using Thompson Sampling"""
        try:
            raw_allocations = {}
            expected_rewards = {}
            uncertainties = {}
            
            for i, agent_id in enumerate(self.agent_ids):
                agent_name = self.agent_names[i] if i < len(self.agent_names) else f"agent_{i}"
                
                # Sample theta from posterior distribution
                try:
                    S_inv = inv(self.S[agent_id])
                    posterior_cov = S_inv / self.noise_precision
                    
                    # Sample from multivariate normal
                    theta_sample = np.random.multivariate_normal(self.mu[agent_id], posterior_cov)
                    
                    # Calculate expected reward for this sample
                    sampled_reward = context @ theta_sample
                    
                    # Calculate posterior mean and uncertainty
                    expected_reward = context @ self.mu[agent_id]
                    uncertainty = np.sqrt(context @ posterior_cov @ context)
                    
                except np.linalg.LinAlgError:
                    # Fallback
                    sampled_reward = np.random.normal(0, 1)
                    expected_reward = 0.0
                    uncertainty = 1.0
                
                expected_rewards[agent_id] = float(expected_reward)
                uncertainties[agent_id] = float(uncertainty)
                raw_allocations[agent_id] = max(0, sampled_reward)  # Ensure non-negative
            
            # Normalize allocations
            normalized_allocations = self._normalize_allocations(raw_allocations)
            
            # Calculate total expected reward
            total_expected_reward = sum(
                normalized_allocations[agent_id] * expected_rewards[agent_id]
                for agent_id in self.agent_ids
            )
            
            # Calculate exploration bonus (average uncertainty)
            exploration_bonus = sum(
                normalized_allocations[agent_id] * uncertainties[agent_id]
                for agent_id in self.agent_ids
            )
            
            self.iteration += 1
            
            return AllocationResult(
                allocations=normalized_allocations,
                expected_rewards=expected_rewards,
                uncertainties=uncertainties,
                total_expected_reward=total_expected_reward,
                exploration_bonus=exploration_bonus,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Thompson Sampling allocation: {e}")
            # Fallback to equal allocation
            equal_alloc = self.config.budget_constraint / len(self.agent_ids)
            return AllocationResult(
                allocations={agent_id: equal_alloc for agent_id in self.agent_ids},
                expected_rewards={agent_id: 0.0 for agent_id in self.agent_ids},
                uncertainties={agent_id: 1.0 for agent_id in self.agent_ids},
                total_expected_reward=0.0,
                exploration_bonus=0.0,
                timestamp=datetime.now()
            )
    
    async def update(self, update: BanditUpdate) -> None:
        """Update Thompson Sampling parameters with reward feedback"""
        try:
            agent_id = update.agent_id
            if agent_id in self.S:
                # Bayesian update
                context = update.context
                reward = update.reward
                
                # Update precision matrix and mean
                self.S[agent_id] += self.noise_precision * np.outer(context, context)
                
                try:
                    S_inv = inv(self.S[agent_id])
                    self.mu[agent_id] = S_inv @ (
                        self.S[agent_id] @ self.mu[agent_id] + 
                        self.noise_precision * reward * context
                    )
                except np.linalg.LinAlgError:
                    # Fallback update
                    learning_rate = self.config.learning_rate
                    prediction_error = reward - context @ self.mu[agent_id]
                    self.mu[agent_id] += learning_rate * prediction_error * context
                
                self.logger.debug(f"Updated Thompson Sampling for agent {agent_id} with reward {reward:.4f}")
            else:
                self.logger.warning(f"Unknown agent ID for update: {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Error updating Thompson Sampling: {e}")


class BayesianBudgetAllocator(BanditAllocator):
    """Bayesian budget allocator with portfolio optimization"""
    
    def __init__(self, config: BanditConfig):
        super().__init__(config)
        
        # Portfolio optimization parameters
        self.risk_aversion = 1.0
        self.transaction_cost = 0.001
        self.max_position_size = 0.3
        
        # Bayesian parameters
        self.alpha_prior = 1.0  # Prior for beta distribution
        self.beta_prior = 1.0
        
        # Agent performance tracking
        self.agent_alphas = {agent_id: self.alpha_prior for agent_id in self.agent_ids}
        self.agent_betas = {agent_id: self.beta_prior for agent_id in self.agent_ids}
        self.agent_performance = {agent_id: [] for agent_id in self.agent_ids}
        
        # Covariance estimation
        self.returns_history = {agent_id: [] for agent_id in self.agent_ids}
        self.correlation_matrix = np.eye(len(self.agent_ids))
        
        self.logger.info(f"Initialized Bayesian Budget allocator with {config.n_agents} agents")
    
    async def select_allocations(self, context: np.ndarray) -> AllocationResult:
        """Select allocations using Bayesian portfolio optimization"""
        try:
            # Sample expected returns from Beta distributions
            expected_returns = {}
            uncertainties = {}
            
            for agent_id in self.agent_ids:
                # Sample from Beta distribution
                alpha = self.agent_alphas[agent_id]
                beta = self.agent_betas[agent_id]
                
                # Expected return and uncertainty
                expected_return = alpha / (alpha + beta)
                variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                uncertainty = np.sqrt(variance)
                
                expected_returns[agent_id] = float(expected_return)
                uncertainties[agent_id] = float(uncertainty)
            
            # Portfolio optimization
            raw_allocations = self._optimize_portfolio(expected_returns, uncertainties, context)
            
            # Normalize allocations
            normalized_allocations = self._normalize_allocations(raw_allocations)
            
            # Calculate total expected reward
            total_expected_reward = sum(
                normalized_allocations[agent_id] * expected_returns[agent_id]
                for agent_id in self.agent_ids
            )
            
            # Calculate exploration bonus (average uncertainty weighted by allocation)
            exploration_bonus = sum(
                normalized_allocations[agent_id] * uncertainties[agent_id]
                for agent_id in self.agent_ids
            )
            
            self.iteration += 1
            
            return AllocationResult(
                allocations=normalized_allocations,
                expected_rewards=expected_returns,
                uncertainties=uncertainties,
                total_expected_reward=total_expected_reward,
                exploration_bonus=exploration_bonus,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian Budget allocation: {e}")
            # Fallback to equal allocation
            equal_alloc = self.config.budget_constraint / len(self.agent_ids)
            return AllocationResult(
                allocations={agent_id: equal_alloc for agent_id in self.agent_ids},
                expected_rewards={agent_id: 0.0 for agent_id in self.agent_ids},
                uncertainties={agent_id: 1.0 for agent_id in self.agent_ids},
                total_expected_reward=0.0,
                exploration_bonus=0.0,
                timestamp=datetime.now()
            )
    
    def _optimize_portfolio(self, expected_returns: Dict[str, float], 
                          uncertainties: Dict[str, float], context: np.ndarray) -> Dict[str, float]:
        """Optimize portfolio using mean-variance framework"""
        try:
            n_agents = len(self.agent_ids)
            
            # Convert to arrays
            mu = np.array([expected_returns[agent_id] for agent_id in self.agent_ids])
            sigma = np.array([uncertainties[agent_id] for agent_id in self.agent_ids])
            
            # Create covariance matrix
            cov_matrix = np.outer(sigma, sigma) * self.correlation_matrix
            
            # Add regularization to ensure positive definiteness
            cov_matrix += np.eye(n_agents) * 1e-6
            
            # Mean-variance optimization with risk aversion
            try:
                inv_cov = inv(cov_matrix)
                ones = np.ones(n_agents)
                
                # Calculate optimal weights (mean-variance efficient portfolio)
                numerator = inv_cov @ (mu - self.risk_aversion * np.diag(cov_matrix))
                denominator = ones @ inv_cov @ ones
                
                weights = numerator / denominator if denominator > 0 else ones / n_agents
                
                # Apply constraints
                weights = np.clip(weights, 0, self.max_position_size)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else ones / n_agents
                
            except np.linalg.LinAlgError:
                # Fallback to simple allocation based on expected returns
                weights = np.maximum(mu, 0)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else ones / n_agents
            
            # Convert back to dictionary
            allocations = {}
            for i, agent_id in enumerate(self.agent_ids):
                allocations[agent_id] = float(weights[i] * self.config.budget_constraint)
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            # Equal allocation fallback
            equal_weight = self.config.budget_constraint / len(self.agent_ids)
            return {agent_id: equal_weight for agent_id in self.agent_ids}
    
    async def update(self, update: BanditUpdate) -> None:
        """Update Bayesian parameters with reward feedback"""
        try:
            agent_id = update.agent_id
            reward = update.reward
            
            if agent_id in self.agent_alphas:
                # Convert reward to binary outcome (success/failure)
                # Assume rewards are in [-1, 1] range
                success = 1 if reward > 0 else 0
                
                # Update Beta distribution parameters
                if success:
                    self.agent_alphas[agent_id] += 1
                else:
                    self.agent_betas[agent_id] += 1
                
                # Track performance history
                self.agent_performance[agent_id].append(reward)
                self.returns_history[agent_id].append(reward)
                
                # Keep only recent history
                max_history = 1000
                if len(self.agent_performance[agent_id]) > max_history:
                    self.agent_performance[agent_id] = self.agent_performance[agent_id][-max_history:]
                    self.returns_history[agent_id] = self.returns_history[agent_id][-max_history:]
                
                # Update correlation matrix periodically
                if self.iteration % self.config.update_frequency == 0:
                    self._update_correlation_matrix()
                
                self.logger.debug(f"Updated Bayesian Budget for agent {agent_id}: α={self.agent_alphas[agent_id]:.2f}, β={self.agent_betas[agent_id]:.2f}")
            else:
                self.logger.warning(f"Unknown agent ID for update: {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Error updating Bayesian Budget: {e}")
    
    def _update_correlation_matrix(self):
        """Update correlation matrix based on returns history"""
        try:
            # Create returns matrix
            min_length = min(len(history) for history in self.returns_history.values() if len(history) > 0)
            
            if min_length >= 30:  # Minimum samples for correlation
                returns_matrix = np.array([
                    self.returns_history[agent_id][-min_length:] 
                    for agent_id in self.agent_ids
                ])
                
                # Calculate correlation matrix
                correlation_matrix = np.corrcoef(returns_matrix)
                
                # Handle NaN values
                correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                
                # Ensure positive semi-definite
                eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
                eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
                correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                # Normalize diagonal to 1
                diag_sqrt = np.sqrt(np.diag(correlation_matrix))
                correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
                
                self.correlation_matrix = correlation_matrix
                self.logger.debug("Updated correlation matrix")
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")


class BanditEnsemble:
    """Ensemble of bandit allocators"""
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize allocators
        self.allocators = {
            'linucb': LinUCBAllocator(config),
            'thompson': ThompsonSamplingAllocator(config),
            'bayesian': BayesianBudgetAllocator(config)
        }
        
        # Meta-learning weights
        self.meta_weights = {name: 1.0 / len(self.allocators) for name in self.allocators.keys()}
        self.meta_performance = {name: [] for name in self.allocators.keys()}
        
        # Feature extractor
        self.feature_extractor = ContextFeatureExtractor(config.context_dim)
        
        self.logger.info(f"Initialized Bandit Ensemble with {len(self.allocators)} allocators")
    
    async def select_allocations(self, market_data: pd.DataFrame,
                               regime_info: Dict[str, Any],
                               portfolio_state: Dict[str, Any]) -> AllocationResult:
        """Select allocations using ensemble of bandits"""
        try:
            # Extract context features
            context = self.feature_extractor.extract_features(market_data, regime_info, portfolio_state)
            
            # Get allocations from each bandit
            allocator_results = {}
            for name, allocator in self.allocators.items():
                result = await allocator.select_allocations(context)
                allocator_results[name] = result
            
            # Ensemble allocation using meta-weights
            ensemble_allocations = {}
            ensemble_expected_rewards = {}
            ensemble_uncertainties = {}
            
            for agent_id in self.config.agent_ids if hasattr(self.config, 'agent_ids') else allocator_results['linucb'].allocations.keys():
                weighted_allocation = 0.0
                weighted_expected_reward = 0.0
                weighted_uncertainty = 0.0
                
                for allocator_name, result in allocator_results.items():
                    weight = self.meta_weights[allocator_name]
                    weighted_allocation += weight * result.allocations.get(agent_id, 0.0)
                    weighted_expected_reward += weight * result.expected_rewards.get(agent_id, 0.0)
                    weighted_uncertainty += weight * result.uncertainties.get(agent_id, 0.0)
                
                ensemble_allocations[agent_id] = weighted_allocation
                ensemble_expected_rewards[agent_id] = weighted_expected_reward
                ensemble_uncertainties[agent_id] = weighted_uncertainty
            
            # Normalize ensemble allocations
            total_allocation = sum(ensemble_allocations.values())
            if total_allocation > 0:
                for agent_id in ensemble_allocations:
                    ensemble_allocations[agent_id] = (
                        ensemble_allocations[agent_id] / total_allocation * self.config.budget_constraint
                    )
            
            # Calculate ensemble metrics
            total_expected_reward = sum(
                ensemble_allocations[agent_id] * ensemble_expected_rewards[agent_id]
                for agent_id in ensemble_allocations
            )
            
            exploration_bonus = sum(
                ensemble_allocations[agent_id] * ensemble_uncertainties[agent_id]
                for agent_id in ensemble_allocations
            )
            
            return AllocationResult(
                allocations=ensemble_allocations,
                expected_rewards=ensemble_expected_rewards,
                uncertainties=ensemble_uncertainties,
                total_expected_reward=total_expected_reward,
                exploration_bonus=exploration_bonus,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in ensemble allocation: {e}")
            # Fallback to equal allocation
            agent_ids = list(self.allocators['linucb'].agent_ids)
            equal_alloc = self.config.budget_constraint / len(agent_ids)
            return AllocationResult(
                allocations={agent_id: equal_alloc for agent_id in agent_ids},
                expected_rewards={agent_id: 0.0 for agent_id in agent_ids},
                uncertainties={agent_id: 1.0 for agent_id in agent_ids},
                total_expected_reward=0.0,
                exploration_bonus=0.0,
                timestamp=datetime.now()
            )
    
    async def update_all(self, updates: List[BanditUpdate]) -> None:
        """Update all bandit allocators"""
        for update in updates:
            for allocator in self.allocators.values():
                await allocator.update(update)
    
    async def update_meta_weights(self, allocator_rewards: Dict[str, float]) -> None:
        """Update meta-weights based on allocator performance"""
        try:
            # Update performance history
            for allocator_name, reward in allocator_rewards.items():
                if allocator_name in self.meta_performance:
                    self.meta_performance[allocator_name].append(reward)
                    # Keep only recent history
                    if len(self.meta_performance[allocator_name]) > 100:
                        self.meta_performance[allocator_name] = self.meta_performance[allocator_name][-100:]
            
            # Update meta-weights using exponential moving average of performance
            for allocator_name in self.meta_weights:
                if allocator_name in self.meta_performance and len(self.meta_performance[allocator_name]) > 0:
                    recent_performance = np.mean(self.meta_performance[allocator_name][-10:])
                    # Use softmax for weight updates
                    self.meta_weights[allocator_name] = np.exp(recent_performance)
            
            # Normalize weights
            total_weight = sum(self.meta_weights.values())
            if total_weight > 0:
                for allocator_name in self.meta_weights:
                    self.meta_weights[allocator_name] /= total_weight
            
            self.logger.debug(f"Updated meta-weights: {self.meta_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating meta-weights: {e}")
    
    async def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of allocation system"""
        return {
            "ensemble": {
                "allocators": list(self.allocators.keys()),
                "meta_weights": self.meta_weights,
                "total_iterations": sum(allocator.iteration for allocator in self.allocators.values())
            },
            "config": {
                "n_agents": self.config.n_agents,
                "budget_constraint": self.config.budget_constraint,
                "context_dim": self.config.context_dim,
                "exploration_bonus": self.config.exploration_bonus
            },
            "performance": {
                allocator_name: {
                    "recent_performance": np.mean(perf[-10:]) if len(perf) >= 10 else 0.0,
                    "total_updates": len(perf)
                }
                for allocator_name, perf in self.meta_performance.items()
            }
        }


# Factory functions
async def create_linucb_allocator(config: Optional[BanditConfig] = None) -> LinUCBAllocator:
    """Create LinUCB allocator"""
    return LinUCBAllocator(config or BanditConfig())


async def create_thompson_allocator(config: Optional[BanditConfig] = None) -> ThompsonSamplingAllocator:
    """Create Thompson Sampling allocator"""
    return ThompsonSamplingAllocator(config or BanditConfig())


async def create_bayesian_allocator(config: Optional[BanditConfig] = None) -> BayesianBudgetAllocator:
    """Create Bayesian Budget allocator"""
    return BayesianBudgetAllocator(config or BanditConfig())


async def create_bandit_ensemble(config: Optional[BanditConfig] = None) -> BanditEnsemble:
    """Create Bandit Ensemble"""
    return BanditEnsemble(config or BanditConfig())


# Example usage
async def main():
    """Example usage of bandit allocators"""
    # Create configuration
    config = BanditConfig(
        n_agents=8,
        alpha=0.1,
        context_dim=15,
        budget_constraint=1.0
    )
    
    # Create ensemble
    ensemble = await create_bandit_ensemble(config)
    
    # Sample market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    market_data = pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Sample regime and portfolio state
    regime_info = {
        'volatility_percentile': 0.7,
        'trend_strength': 0.3,
        'correlation_level': 0.6,
        'risk_on_factor': 0.4
    }
    
    portfolio_state = {
        'total_pnl': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.03,
        'positions_count': 5,
        'cash_ratio': 0.1
    }
    
    # Get allocations
    result = await ensemble.select_allocations(market_data, regime_info, portfolio_state)
    
    print("Bandit Allocation Results:")
    print(f"Total Expected Reward: {result.total_expected_reward:.4f}")
    print(f"Exploration Bonus: {result.exploration_bonus:.4f}")
    print("\nAgent Allocations:")
    for agent_id, allocation in result.allocations.items():
        expected_reward = result.expected_rewards[agent_id]
        uncertainty = result.uncertainties[agent_id]
        print(f"  {agent_id}: {allocation:.3f} (reward: {expected_reward:.3f}, uncertainty: {uncertainty:.3f})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
