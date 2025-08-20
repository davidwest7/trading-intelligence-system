#!/usr/bin/env python3
"""
RL Sizer/Hedger: Constrained, CVaR-aware Position Sizing

This component implements reinforcement learning-based position sizing with
Conditional Value at Risk (CVaR) constraints and hedging strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    symbol: str
    action: str
    position_size: float
    hedge_size: float
    hedge_instruments: List[str]
    cvar: float
    expected_return: float
    risk_budget: float
    constraints_satisfied: bool
    metadata: Dict[str, Any]

@dataclass
class HedgingStrategy:
    """Hedging strategy configuration"""
    primary_hedge: str  # e.g., 'SPY' for market beta
    secondary_hedges: List[str]  # e.g., ['VIX', 'TLT'] for volatility and rates
    hedge_ratios: Dict[str, float]
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly'
    max_hedge_cost: float

class RLSizerHedger:
    """
    RL Sizer/Hedger: Constrained, CVaR-aware position sizing
    
    Features:
    - Reinforcement learning for optimal position sizing
    - CVaR-based risk constraints
    - Dynamic hedging strategies
    - Portfolio-level risk budgeting
    - Multi-instrument hedging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Risk parameters
        self.cvar_confidence = self.config.get('cvar_confidence', 0.95)
        self.max_cvar = self.config.get('max_cvar', 0.02)  # 2% max CVaR
        self.risk_budget = self.config.get('risk_budget', 0.15)  # 15% total risk budget
        
        # Position sizing constraints
        self.max_position_size = self.config.get('max_position_size', 0.10)
        self.min_position_size = self.config.get('min_position_size', 0.01)
        self.max_leverage = self.config.get('max_leverage', 2.0)
        
        # Hedging parameters
        self.hedge_threshold = self.config.get('hedge_threshold', 0.01)  # 1% risk threshold
        self.max_hedge_ratio = self.config.get('max_hedge_ratio', 0.5)  # 50% max hedge
        self.hedge_cost_limit = self.config.get('hedge_cost_limit', 0.005)  # 0.5% max hedge cost
        
        # RL parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        
        # Hedging strategies
        self.hedging_strategies = {
            'market_beta': HedgingStrategy(
                primary_hedge='SPY',
                secondary_hedges=['QQQ', 'IWM'],
                hedge_ratios={'SPY': 0.7, 'QQQ': 0.2, 'IWM': 0.1},
                rebalance_frequency='daily',
                max_hedge_cost=0.003
            ),
            'volatility': HedgingStrategy(
                primary_hedge='VIX',
                secondary_hedges=['UVXY', 'VXX'],
                hedge_ratios={'VIX': 0.8, 'UVXY': 0.2},
                rebalance_frequency='weekly',
                max_hedge_cost=0.005
            ),
            'interest_rates': HedgingStrategy(
                primary_hedge='TLT',
                secondary_hedges=['IEF', 'SHY'],
                hedge_ratios={'TLT': 0.6, 'IEF': 0.3, 'SHY': 0.1},
                rebalance_frequency='weekly',
                max_hedge_cost=0.004
            )
        }
        
        # Performance tracking
        self.sizing_history = []
        self.hedging_history = []
        self.risk_budget_usage = 0.0
        
        # RL state
        self.q_table = {}  # State-action value function
        self.state_history = []
        
        logger.info("RL Sizer/Hedger initialized with CVaR constraints and hedging strategies")
    
    def calculate_position_size(self, opportunity: Dict[str, Any],
                              portfolio_state: Dict[str, Any],
                              market_data: pd.DataFrame) -> PositionSizingResult:
        """Calculate optimal position size using RL and CVaR constraints"""
        try:
            symbol = opportunity['symbol']
            signal_strength = opportunity['signal_strength']
            expected_return = opportunity['expected_return']
            risk_score = opportunity['risk_score']
            
            # Get current state
            current_state = self._get_current_state(portfolio_state, market_data)
            
            # Calculate CVaR
            cvar = self._calculate_cvar(symbol, market_data, risk_score)
            
            # Check if CVaR constraint is satisfied
            if cvar > self.max_cvar:
                logger.warning(f"CVaR constraint violated for {symbol}: {cvar:.4f} > {self.max_cvar}")
                return self._create_zero_position(symbol)
            
            # Calculate risk budget allocation
            risk_budget_allocation = self._allocate_risk_budget(
                opportunity, portfolio_state, cvar
            )
            
            # Use RL to determine position size
            position_size = self._rl_position_sizing(
                current_state, opportunity, risk_budget_allocation
            )
            
            # Apply constraints
            position_size = self._apply_position_constraints(
                position_size, portfolio_state
            )
            
            # Calculate hedging requirements
            hedge_size, hedge_instruments = self._calculate_hedging(
                symbol, position_size, market_data, portfolio_state
            )
            
            # Update risk budget usage
            self._update_risk_budget(cvar, position_size)
            
            # Create result
            result = PositionSizingResult(
                symbol=symbol,
                action='BUY' if signal_strength > 0 else 'SELL',
                position_size=position_size,
                hedge_size=hedge_size,
                hedge_instruments=hedge_instruments,
                cvar=cvar,
                expected_return=expected_return,
                risk_budget=risk_budget_allocation,
                constraints_satisfied=True,
                metadata={
                    'rl_state': current_state,
                    'hedge_cost': self._calculate_hedge_cost(hedge_size, hedge_instruments),
                    'risk_budget_remaining': self.risk_budget - self.risk_budget_usage
                }
            )
            
            self.sizing_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self._create_zero_position(opportunity.get('symbol', 'UNKNOWN'))
    
    def _get_current_state(self, portfolio_state: Dict[str, Any],
                          market_data: pd.DataFrame) -> str:
        """Get current state for RL agent"""
        try:
            # Create state representation
            portfolio_value = portfolio_state.get('total_value', 1000000)
            cash_ratio = portfolio_state.get('cash', 0) / portfolio_value
            current_risk = portfolio_state.get('current_risk', 0)
            
            # Market conditions
            if market_data is not None and len(market_data) > 20:
                volatility = market_data['close'].pct_change().std()
                momentum = market_data['close'].pct_change(20).mean()
            else:
                volatility = 0.02
                momentum = 0.0
            
            # Discretize state
            cash_bin = 'low' if cash_ratio < 0.2 else 'high' if cash_ratio > 0.8 else 'medium'
            risk_bin = 'low' if current_risk < 0.05 else 'high' if current_risk > 0.15 else 'medium'
            vol_bin = 'low' if volatility < 0.015 else 'high' if volatility > 0.03 else 'medium'
            mom_bin = 'up' if momentum > 0.001 else 'down' if momentum < -0.001 else 'flat'
            
            state = f"{cash_bin}_{risk_bin}_{vol_bin}_{mom_bin}"
            return state
            
        except Exception as e:
            logger.warning(f"Error getting current state: {e}")
            return "medium_medium_medium_flat"
    
    def _calculate_cvar(self, symbol: str, market_data: pd.DataFrame,
                       risk_score: float) -> float:
        """Calculate Conditional Value at Risk"""
        try:
            if market_data is None or len(market_data) < 50:
                return risk_score * 1.5  # Conservative estimate
            
            # Get symbol-specific data
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) < 50:
                return risk_score * 1.5
            
            # Calculate returns
            returns = symbol_data['close'].pct_change().dropna()
            
            # Calculate VaR
            var = np.percentile(returns, (1 - self.cvar_confidence) * 100)
            
            # Calculate CVaR (expected loss beyond VaR)
            tail_returns = returns[returns <= var]
            if len(tail_returns) > 0:
                cvar = np.mean(tail_returns)
            else:
                cvar = var
            
            # Adjust for risk score
            adjusted_cvar = abs(cvar) * (1 + risk_score)
            
            return min(adjusted_cvar, 0.1)  # Cap at 10%
            
        except Exception as e:
            logger.warning(f"Error calculating CVaR: {e}")
            return risk_score * 1.5
    
    def _allocate_risk_budget(self, opportunity: Dict[str, Any],
                            portfolio_state: Dict[str, Any],
                            cvar: float) -> float:
        """Allocate risk budget for the opportunity"""
        try:
            # Calculate available risk budget
            used_budget = self.risk_budget_usage
            available_budget = self.risk_budget - used_budget
            
            if available_budget <= 0:
                return 0.0
            
            # Allocate based on signal strength and confidence
            signal_strength = abs(opportunity['signal_strength'])
            confidence = opportunity['confidence']
            
            # Risk allocation score
            allocation_score = (signal_strength * 0.6 + confidence * 0.4)
            
            # Allocate proportionally to available budget
            risk_allocation = min(
                available_budget * allocation_score,
                cvar * 2  # Don't allocate more than 2x CVaR
            )
            
            return risk_allocation
            
        except Exception as e:
            logger.warning(f"Error allocating risk budget: {e}")
            return 0.0
    
    def _rl_position_sizing(self, state: str, opportunity: Dict[str, Any],
                          risk_budget: float) -> float:
        """Use reinforcement learning to determine position size"""
        try:
            # Initialize Q-table if needed
            if state not in self.q_table:
                self.q_table[state] = {}
            
            # Define action space (position sizes)
            actions = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]  # 1% to 10%
            
            # Epsilon-greedy action selection
            if np.random.random() < self.exploration_rate:
                # Exploration: random action
                position_size = np.random.choice(actions)
            else:
                # Exploitation: best action
                q_values = [self.q_table[state].get(action, 0) for action in actions]
                best_action_idx = np.argmax(q_values)
                position_size = actions[best_action_idx]
            
            # Adjust based on risk budget
            max_size_from_budget = risk_budget / opportunity.get('risk_score', 0.02)
            position_size = min(position_size, max_size_from_budget)
            
            return position_size
            
        except Exception as e:
            logger.warning(f"Error in RL position sizing: {e}")
            return 0.02  # Default 2%
    
    def _apply_position_constraints(self, position_size: float,
                                  portfolio_state: Dict[str, Any]) -> float:
        """Apply position size constraints"""
        try:
            # Basic constraints
            position_size = max(self.min_position_size, 
                              min(self.max_position_size, position_size))
            
            # Leverage constraint
            portfolio_value = portfolio_state.get('total_value', 1000000)
            current_exposure = portfolio_state.get('current_exposure', 0)
            
            max_additional_exposure = portfolio_value * (self.max_leverage - 1)
            if current_exposure + (position_size * portfolio_value) > max_additional_exposure:
                position_size = max(0, (max_additional_exposure - current_exposure) / portfolio_value)
            
            return position_size
            
        except Exception as e:
            logger.warning(f"Error applying position constraints: {e}")
            return self.min_position_size
    
    def _calculate_hedging(self, symbol: str, position_size: float,
                          market_data: pd.DataFrame,
                          portfolio_state: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate hedging requirements"""
        try:
            if position_size < self.hedge_threshold:
                return 0.0, []
            
            # Determine hedging strategy based on position characteristics
            strategy = self._select_hedging_strategy(symbol, position_size, market_data)
            
            if not strategy:
                return 0.0, []
            
            # Calculate hedge size
            hedge_size = position_size * strategy.hedge_ratios.get(strategy.primary_hedge, 0.5)
            
            # Apply hedge cost constraint
            hedge_cost = self._calculate_hedge_cost(hedge_size, [strategy.primary_hedge])
            if hedge_cost > strategy.max_hedge_cost:
                hedge_size *= strategy.max_hedge_cost / hedge_cost
            
            hedge_instruments = [strategy.primary_hedge] + strategy.secondary_hedges[:2]
            
            return hedge_size, hedge_instruments
            
        except Exception as e:
            logger.warning(f"Error calculating hedging: {e}")
            return 0.0, []
    
    def _select_hedging_strategy(self, symbol: str, position_size: float,
                               market_data: pd.DataFrame) -> Optional[HedgingStrategy]:
        """Select appropriate hedging strategy"""
        try:
            # Simple strategy selection based on position size and market conditions
            if position_size > 0.05:  # Large position
                if market_data is not None and len(market_data) > 20:
                    volatility = market_data['close'].pct_change().std()
                    if volatility > 0.025:  # High volatility
                        return self.hedging_strategies['volatility']
                    else:
                        return self.hedging_strategies['market_beta']
                else:
                    return self.hedging_strategies['market_beta']
            else:
                return None  # No hedging for small positions
                
        except Exception as e:
            logger.warning(f"Error selecting hedging strategy: {e}")
            return None
    
    def _calculate_hedge_cost(self, hedge_size: float, hedge_instruments: List[str]) -> float:
        """Calculate cost of hedging"""
        try:
            # Simple cost model
            base_cost = 0.001  # 0.1% base cost
            instrument_cost = 0.0005  # 0.05% per instrument
            
            total_cost = base_cost + (len(hedge_instruments) * instrument_cost)
            return hedge_size * total_cost
            
        except Exception as e:
            logger.warning(f"Error calculating hedge cost: {e}")
            return 0.0
    
    def _update_risk_budget(self, cvar: float, position_size: float):
        """Update risk budget usage"""
        try:
            self.risk_budget_usage += cvar * position_size
            
            # Reset if over budget
            if self.risk_budget_usage > self.risk_budget:
                self.risk_budget_usage = self.risk_budget
                
        except Exception as e:
            logger.warning(f"Error updating risk budget: {e}")
    
    def _create_zero_position(self, symbol: str) -> PositionSizingResult:
        """Create zero position result"""
        return PositionSizingResult(
            symbol=symbol,
            action='HOLD',
            position_size=0.0,
            hedge_size=0.0,
            hedge_instruments=[],
            cvar=0.0,
            expected_return=0.0,
            risk_budget=0.0,
            constraints_satisfied=False,
            metadata={'reason': 'constraints_not_satisfied'}
        )
    
    def update_rewards(self, symbol: str, actual_return: float, position_size: float):
        """Update RL rewards based on actual performance"""
        try:
            # Find the most recent sizing decision for this symbol
            for sizing in reversed(self.sizing_history):
                if sizing.symbol == symbol:
                    # Calculate reward
                    reward = actual_return * position_size
                    
                    # Update Q-table
                    state = sizing.metadata.get('rl_state', 'medium_medium_medium_flat')
                    action = sizing.position_size
                    
                    if state not in self.q_table:
                        self.q_table[state] = {}
                    
                    # Q-learning update
                    current_q = self.q_table[state].get(action, 0)
                    max_future_q = max(self.q_table[state].values()) if self.q_table[state] else 0
                    
                    new_q = current_q + self.learning_rate * (
                        reward + self.discount_factor * max_future_q - current_q
                    )
                    
                    self.q_table[state][action] = new_q
                    break
                    
        except Exception as e:
            logger.warning(f"Error updating rewards: {e}")
    
    def reset_risk_budget(self):
        """Reset risk budget (e.g., at end of day)"""
        self.risk_budget_usage = 0.0
        logger.info("Risk budget reset")
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing performance"""
        try:
            if not self.sizing_history:
                return {}
            
            total_sizings = len(self.sizing_history)
            successful_sizings = sum(1 for s in self.sizing_history if s.constraints_satisfied)
            
            avg_position_size = np.mean([s.position_size for s in self.sizing_history])
            avg_cvar = np.mean([s.cvar for s in self.sizing_history])
            avg_hedge_size = np.mean([s.hedge_size for s in self.sizing_history])
            
            return {
                'total_sizings': total_sizings,
                'successful_sizings': successful_sizings,
                'success_rate': successful_sizings / total_sizings if total_sizings > 0 else 0,
                'avg_position_size': avg_position_size,
                'avg_cvar': avg_cvar,
                'avg_hedge_size': avg_hedge_size,
                'risk_budget_usage': self.risk_budget_usage,
                'risk_budget_remaining': self.risk_budget - self.risk_budget_usage,
                'q_table_size': len(self.q_table)
            }
            
        except Exception as e:
            logger.warning(f"Error getting sizing summary: {e}")
            return {}
