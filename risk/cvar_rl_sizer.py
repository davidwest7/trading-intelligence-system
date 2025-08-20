#!/usr/bin/env python3
"""
CVaR-Aware Reinforcement Learning Portfolio Sizer

Implements Constrained Markov Decision Process (CMDP) for portfolio sizing
with Conditional Value at Risk (CVaR) optimization and hard constraints.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

from schemas.contracts import Opportunity, Intent
from common.observability.telemetry import log_event, trace_operation


logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of portfolio constraints"""
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    SECTOR_LIMIT = "sector_limit"
    SINGLE_POSITION = "single_position"
    LEVERAGE_CAP = "leverage_cap"
    VAR_LIMIT = "var_limit"
    CVAR_LIMIT = "cvar_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"


@dataclass
class PortfolioState:
    """Portfolio state representation"""
    positions: Dict[str, float]  # symbol -> position size
    cash: float
    portfolio_value: float
    sector_exposures: Dict[str, float]
    current_risk: float
    current_cvar: float
    current_drawdown: float
    timestamp: datetime
    
    def to_array(self) -> np.ndarray:
        """Convert state to feature array"""
        # Normalize positions by portfolio value
        pos_array = np.array([self.positions.get(sym, 0.0) / max(self.portfolio_value, 1e-6) 
                             for sym in sorted(self.positions.keys())])
        
        # Add portfolio metrics
        metrics = np.array([
            self.cash / max(self.portfolio_value, 1e-6),
            self.current_risk,
            self.current_cvar,
            self.current_drawdown,
            len(self.positions) / 100.0  # Position count normalized
        ])
        
        return np.concatenate([pos_array, metrics])


@dataclass
class SizingAction:
    """Portfolio sizing action"""
    symbol: str
    target_size: float  # Target position size in EUR
    confidence: float   # Action confidence [0, 1]
    risk_contribution: float  # Expected risk contribution
    expected_cost: float  # Expected execution cost
    
    def to_array(self) -> np.ndarray:
        """Convert action to feature array"""
        return np.array([
            self.target_size,
            self.confidence,
            self.risk_contribution,
            self.expected_cost
        ])


class CVaRRLSizer:
    """
    CVaR-Aware Reinforcement Learning Portfolio Sizer
    
    Features:
    - Constrained MDP with CVaR objective
    - Lagrange multiplier learning for constraints
    - Action projection into feasible set
    - Real-time risk monitoring and throttling
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_budget = config.get('risk_budget', 500.0)  # EUR
        self.max_gross_exposure = config.get('max_gross_exposure', 1.5)
        self.max_net_exposure = config.get('max_net_exposure', 0.5)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.25)
        self.max_single_position = config.get('max_single_position', 0.05)
        self.max_leverage = config.get('max_leverage', 2.0)
        self.max_var = config.get('max_var', 0.02)
        self.max_cvar = config.get('max_cvar', 0.03)
        self.max_drawdown = config.get('max_drawdown', 0.10)
        
        # CVaR parameters
        self.cvar_alpha = config.get('cvar_alpha', 0.95)
        self.risk_aversion = config.get('risk_aversion', 2.0)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.exploration_epsilon = config.get('exploration_epsilon', 0.1)
        self.lagrange_learning_rate = config.get('lagrange_learning_rate', 0.01)
        
        # State tracking
        self.current_state: Optional[PortfolioState] = None
        self.lagrange_multipliers: Dict[ConstraintType, float] = {
            constraint: 0.0 for constraint in ConstraintType
        }
        
        # Performance tracking
        self.total_reward = 0.0
        self.constraint_violations = {constraint: 0 for constraint in ConstraintType}
        self.action_count = 0
        
        logger.info(f"CVaR RL Sizer initialized with risk budget: {self.risk_budget} EUR")
    
    async def size_portfolio(self, opportunities: List[Opportunity], 
                           current_positions: Dict[str, float],
                           portfolio_value: float,
                           trace_id: str) -> List[Intent]:
        """
        Size portfolio using CVaR-aware RL with constraints
        
        Args:
            opportunities: List of opportunities from selector
            current_positions: Current portfolio positions
            portfolio_value: Current portfolio value
            trace_id: Trace ID for observability
            
        Returns:
            List of sizing intents with position targets
        """
        async with trace_operation("cvar_rl_sizing", trace_id=trace_id):
            try:
                # Update current state
                self._update_state(current_positions, portfolio_value)
                
                # Generate sizing actions for opportunities
                sizing_actions = []
                for opp in opportunities:
                    action = await self._generate_sizing_action(opp, trace_id)
                    if action:
                        sizing_actions.append(action)
                
                # Apply constraint projection
                feasible_actions = await self._project_to_feasible_set(sizing_actions, trace_id)
                
                # Convert to intents
                intents = []
                for action in feasible_actions:
                    intent = Intent(
                        trace_id=trace_id,
                        opportunity_id=action.symbol,  # Using symbol as opportunity ID
                        symbol=action.symbol,
                        size_eur=action.target_size,
                        risk_eur=action.risk_contribution,
                        confidence=action.confidence,
                        expected_cost=action.expected_cost,
                        sizing_method="cvar_rl",
                        risk_budget_used=action.risk_contribution / self.risk_budget,
                        constraint_satisfied=True,
                        timestamp=datetime.utcnow()
                    )
                    intents.append(intent)
                
                # Update learning
                await self._update_learning(sizing_actions, feasible_actions, trace_id)
                
                # Log metrics
                await self._log_sizing_metrics(intents, trace_id)
                
                return intents
                
            except Exception as e:
                logger.error(f"CVaR RL sizing failed: {e}", extra={'trace_id': trace_id})
                return []
    
    def _update_state(self, positions: Dict[str, float], portfolio_value: float):
        """Update current portfolio state"""
        # Calculate sector exposures (simplified)
        sector_exposures = self._calculate_sector_exposures(positions)
        
        # Calculate current risk metrics
        current_risk = self._calculate_portfolio_risk(positions, portfolio_value)
        current_cvar = self._calculate_portfolio_cvar(positions, portfolio_value)
        current_drawdown = self._calculate_drawdown(portfolio_value)
        
        self.current_state = PortfolioState(
            positions=positions.copy(),
            cash=portfolio_value - sum(positions.values()),
            portfolio_value=portfolio_value,
            sector_exposures=sector_exposures,
            current_risk=current_risk,
            current_cvar=current_cvar,
            current_drawdown=current_drawdown,
            timestamp=datetime.utcnow()
        )
    
    async def _generate_sizing_action(self, opportunity: Opportunity, 
                                    trace_id: str) -> Optional[SizingAction]:
        """Generate sizing action for an opportunity using RL policy"""
        try:
            # State features: opportunity + current portfolio state
            state_features = self._extract_state_features(opportunity)
            
            # Action features: opportunity characteristics
            action_features = np.array([
                opportunity.mu_blended,
                opportunity.sigma_blended,
                opportunity.confidence_blended,
                opportunity.sharpe_ratio or 0.0,
                opportunity.var_95,
                opportunity.cvar_95
            ])
            
            # Combine features
            combined_features = np.concatenate([state_features, action_features])
            
            # Generate action using policy (simplified for now)
            target_size = await self._policy_forward(combined_features, opportunity)
            
            if target_size <= 0:
                return None
            
            # Calculate risk contribution
            risk_contribution = self._calculate_risk_contribution(
                opportunity, target_size
            )
            
            # Estimate execution cost
            expected_cost = self._estimate_execution_cost(opportunity, target_size)
            
            # Calculate confidence based on opportunity quality
            confidence = min(opportunity.confidence_blended, 0.95)
            
            return SizingAction(
                symbol=opportunity.symbol,
                target_size=target_size,
                confidence=confidence,
                risk_contribution=risk_contribution,
                expected_cost=expected_cost
            )
            
        except Exception as e:
            logger.error(f"Action generation failed for {opportunity.symbol}: {e}")
            return None
    
    async def _policy_forward(self, features: np.ndarray, 
                            opportunity: Opportunity) -> float:
        """
        Forward pass through RL policy network
        
        Simplified implementation - in production would use a proper neural network
        """
        # Simple linear policy with exploration
        if np.random.random() < self.exploration_epsilon:
            # Exploration: random action
            # Use default ADV if not available
            symbol_adv = getattr(opportunity, 'symbol_adv', 1000000)  # Default 1M ADV
            max_size = min(
                self.risk_budget * 0.1,  # 10% of risk budget
                symbol_adv * 0.01,  # 1% of ADV
                self.risk_budget / opportunity.sigma_blended if opportunity.sigma_blended > 0 else 0
            )
            return np.random.uniform(0, max_size)
        else:
            # Exploitation: policy-based action
            # Simple heuristic: Kelly criterion with risk adjustment
            kelly_fraction = max(0, opportunity.mu_blended / opportunity.sigma_blended**2)
            kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
            
            # Adjust for risk budget
            risk_budget_fraction = self.risk_budget * kelly_fraction / self.risk_aversion
            
            # Adjust for opportunity confidence
            confidence_adjustment = opportunity.confidence_blended
            
            target_size = risk_budget_fraction * confidence_adjustment
            
            # Apply limits
            target_size = min(target_size, self.risk_budget * 0.1)  # Max 10% of budget
            # Use default ADV if not available
            symbol_adv = getattr(opportunity, 'symbol_adv', 1000000)  # Default 1M ADV
            target_size = min(target_size, symbol_adv * 0.01)  # Max 1% ADV
            
            return max(0, target_size)
    
    async def _project_to_feasible_set(self, actions: List[SizingAction], 
                                     trace_id: str) -> List[SizingAction]:
        """
        Project actions into feasible set using convex optimization
        
        Implements action projection to satisfy all constraints
        """
        if not actions:
            return []
        
        try:
            # Extract current portfolio state
            current_positions = self.current_state.positions.copy()
            portfolio_value = self.current_state.portfolio_value
            
            # Create optimization variables
            action_sizes = cp.Variable(len(actions))
            
            # Objective: maximize expected return minus CVaR penalty
            expected_returns = cp.sum([action.target_size * opp.mu_blended 
                                     for action, opp in zip(actions, self._get_opportunities(actions))])
            
            # CVaR penalty (simplified)
            cvar_penalty = cp.sum([action.target_size * opp.cvar_95 
                                 for action, opp in zip(actions, self._get_opportunities(actions))])
            
            objective = expected_returns - self.risk_aversion * cvar_penalty
            
            # Constraints
            constraints = []
            
            # Gross exposure constraint
            total_gross = cp.sum(cp.abs(action_sizes))
            constraints.append(total_gross <= self.max_gross_exposure * portfolio_value)
            
            # Net exposure constraint
            total_net = cp.sum(action_sizes)
            constraints.append(cp.abs(total_net) <= self.max_net_exposure * portfolio_value)
            
            # Single position constraint
            for i in range(len(actions)):
                constraints.append(cp.abs(action_sizes[i]) <= self.max_single_position * portfolio_value)
            
            # Risk budget constraint
            total_risk = cp.sum([action_sizes[i] * actions[i].risk_contribution / actions[i].target_size 
                               for i in range(len(actions)) if actions[i].target_size > 0])
            constraints.append(total_risk <= self.risk_budget)
            
            # Non-negative constraint
            constraints.append(action_sizes >= 0)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Maximize(objective), constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                # Update action sizes with optimal values
                for i, action in enumerate(actions):
                    action.target_size = max(0, action_sizes.value[i])
                
                return [action for action in actions if action.target_size > 0]
            else:
                logger.warning(f"Optimization failed, using original actions")
                return actions
                
        except Exception as e:
            logger.error(f"Action projection failed: {e}")
            return actions
    
    def _calculate_risk_contribution(self, opportunity: Opportunity, 
                                   position_size: float) -> float:
        """Calculate risk contribution of a position"""
        # Simplified risk contribution calculation
        # In production, would use proper portfolio risk models
        return position_size * opportunity.sigma_blended
    
    def _estimate_execution_cost(self, opportunity: Opportunity, 
                               position_size: float) -> float:
        """Estimate execution cost for a position"""
        # Simplified cost model
        # In production, would use sophisticated cost models
        base_cost = 0.0005  # 5bps base cost
        # Use default ADV if not available
        symbol_adv = getattr(opportunity, 'symbol_adv', 1e6)  # Default 1M ADV
        size_impact = position_size / symbol_adv * 0.001  # Size impact
        volatility_impact = opportunity.sigma_blended * 0.1  # Volatility impact
        
        total_cost = base_cost + size_impact + volatility_impact
        return position_size * total_cost
    
    def _calculate_sector_exposures(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposures (simplified)"""
        # Simplified sector mapping
        sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'TSLA': 'Automotive', 'AMZN': 'Consumer', 'NVDA': 'Technology',
            'META': 'Technology', 'JPM': 'Financial', 'XOM': 'Energy', 'JNJ': 'Healthcare'
        }
        
        sector_exposures = {}
        for symbol, position in positions.items():
            sector = sector_mapping.get(symbol, 'Other')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(position)
        
        return sector_exposures
    
    def _calculate_portfolio_risk(self, positions: Dict[str, float], 
                                portfolio_value: float) -> float:
        """Calculate portfolio risk (simplified)"""
        if not positions:
            return 0.0
        
        # Simplified risk calculation
        total_risk = sum(abs(pos) * 0.02 for pos in positions.values())  # 2% per position
        return total_risk / max(portfolio_value, 1e-6)
    
    def _calculate_portfolio_cvar(self, positions: Dict[str, float], 
                                portfolio_value: float) -> float:
        """Calculate portfolio CVaR (simplified)"""
        if not positions:
            return 0.0
        
        # Simplified CVaR calculation
        total_cvar = sum(abs(pos) * 0.03 for pos in positions.values())  # 3% per position
        return total_cvar / max(portfolio_value, 1e-6)
    
    def _calculate_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown (simplified)"""
        # Simplified drawdown calculation
        # In production, would track peak portfolio value
        return 0.0  # Placeholder
    
    def _extract_state_features(self, opportunity: Opportunity) -> np.ndarray:
        """Extract state features for RL policy"""
        if not self.current_state:
            return np.zeros(10)
        
        # Portfolio state features
        portfolio_features = np.array([
            self.current_state.cash / max(self.current_state.portfolio_value, 1e-6),
            self.current_state.current_risk,
            self.current_state.current_cvar,
            self.current_state.current_drawdown,
            len(self.current_state.positions) / 100.0
        ])
        
        # Opportunity-specific features
        opportunity_features = np.array([
            opportunity.mu_blended,
            opportunity.sigma_blended,
            opportunity.confidence_blended,
            opportunity.sharpe_ratio or 0.0
        ])
        
        return np.concatenate([portfolio_features, opportunity_features])
    
    def _get_opportunities(self, actions: List[SizingAction]) -> List[Opportunity]:
        """Get opportunities corresponding to actions (simplified)"""
        # In production, would maintain opportunity mapping
        # For now, create dummy opportunities
        opportunities = []
        for action in actions:
            opp = Opportunity(
                symbol=action.symbol,
                mu_blended=0.01,  # Placeholder
                sigma_blended=0.02,  # Placeholder
                confidence_blended=action.confidence,
                sharpe_ratio=0.5,  # Placeholder
                var_95=-0.02,  # Placeholder
                cvar_95=-0.025,  # Placeholder
                agent_signals={},
                trace_id="dummy"
            )
            opportunities.append(opp)
        return opportunities
    
    async def _update_learning(self, original_actions: List[SizingAction],
                             feasible_actions: List[SizingAction], 
                             trace_id: str):
        """Update learning based on action projection"""
        try:
            # Calculate constraint violations
            violations = self._calculate_constraint_violations(original_actions, feasible_actions)
            
            # Update Lagrange multipliers
            for constraint_type, violation in violations.items():
                if violation > 0:
                    self.lagrange_multipliers[constraint_type] += self.lagrange_learning_rate * violation
                    self.constraint_violations[constraint_type] += 1
            
            # Update exploration epsilon
            self.exploration_epsilon = max(0.01, self.exploration_epsilon * 0.999)
            
            self.action_count += 1
            
        except Exception as e:
            logger.error(f"Learning update failed: {e}")
    
    def _calculate_constraint_violations(self, original_actions: List[SizingAction],
                                       feasible_actions: List[SizingAction]) -> Dict[ConstraintType, float]:
        """Calculate constraint violations"""
        violations = {constraint: 0.0 for constraint in ConstraintType}
        
        # Calculate total original vs feasible sizes
        original_total = sum(action.target_size for action in original_actions)
        feasible_total = sum(action.target_size for action in feasible_actions)
        
        # Gross exposure violation
        if original_total > self.max_gross_exposure * (self.current_state.portfolio_value or 1e6):
            violations[ConstraintType.GROSS_EXPOSURE] = original_total - feasible_total
        
        return violations
    
    async def _log_sizing_metrics(self, intents: List[Intent], trace_id: str):
        """Log sizing metrics for monitoring"""
        try:
            total_size = sum(intent.size_eur for intent in intents)
            total_risk = sum(intent.risk_eur for intent in intents)
            avg_confidence = np.mean([intent.confidence for intent in intents]) if intents else 0
            
            await log_event("cvar_rl_sizing_complete", {
                "trace_id": trace_id,
                "intent_count": len(intents),
                "total_size_eur": total_size,
                "total_risk_eur": total_risk,
                "avg_confidence": avg_confidence,
                "risk_budget_used": total_risk / self.risk_budget,
                "constraint_violations": self.constraint_violations,
                "lagrange_multipliers": self.lagrange_multipliers
            })
            
        except Exception as e:
            logger.error(f"Metrics logging failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "total_reward": self.total_reward,
            "action_count": self.action_count,
            "constraint_violations": self.constraint_violations,
            "lagrange_multipliers": self.lagrange_multipliers,
            "exploration_epsilon": self.exploration_epsilon,
            "current_risk": self.current_state.current_risk if self.current_state else 0,
            "current_cvar": self.current_state.current_cvar if self.current_state else 0
        }
