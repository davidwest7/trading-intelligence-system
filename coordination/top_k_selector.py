#!/usr/bin/env python3
"""
Top-K Selector: Diversified Bandit for Opportunity Selection

This component implements a multi-armed bandit approach to select the top K
opportunities while maintaining diversification and exploration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class Opportunity:
    """Trading opportunity with metadata"""
    symbol: str
    signal_strength: float
    confidence: float
    expected_return: float
    risk_score: float
    agent_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    horizon: str
    sector: str = ""
    market_cap: str = ""

@dataclass
class SelectionResult:
    """Result of top-K selection"""
    selected_opportunities: List[Opportunity]
    selection_scores: Dict[str, float]
    diversification_metrics: Dict[str, float]
    exploration_bonus: Dict[str, float]
    timestamp: datetime

class TopKSelector:
    """
    Top-K Selector: Diversified bandit for opportunity selection
    
    Features:
    - Multi-armed bandit with UCB1 exploration
    - Diversification constraints (sector, market cap, agent)
    - Dynamic K selection based on market conditions
    - Exploration vs exploitation balance
    - Risk-adjusted scoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Bandit parameters
        self.exploration_constant = self.config.get('exploration_constant', 2.0)
        self.min_exploration_rate = self.config.get('min_exploration_rate', 0.1)
        self.max_exploration_rate = self.config.get('max_exploration_rate', 0.3)
        
        # Diversification constraints
        self.max_sector_weight = self.config.get('max_sector_weight', 0.25)
        self.max_agent_weight = self.config.get('max_agent_weight', 0.20)
        self.max_market_cap_weight = self.config.get('max_market_cap_weight', 0.30)
        
        # Dynamic K selection
        self.base_k = self.config.get('base_k', 10)
        self.max_k = self.config.get('max_k', 20)
        self.min_k = self.config.get('min_k', 5)
        
        # Performance tracking
        self.opportunity_history = defaultdict(list)
        self.selection_history = []
        self.performance_tracking = defaultdict(lambda: {'rewards': [], 'count': 0})
        
        # Current state
        self.current_k = self.base_k
        self.exploration_rate = 0.2
        self.last_update = datetime.now()
        
        logger.info("Top-K Selector initialized with bandit algorithm and diversification")
    
    def select_top_k(self, opportunities: List[Opportunity], 
                    market_data: pd.DataFrame = None,
                    portfolio_state: Dict[str, Any] = None) -> SelectionResult:
        """Select top K opportunities using diversified bandit"""
        try:
            if not opportunities:
                return SelectionResult([], {}, {}, {}, datetime.now())
            
            # Determine K based on market conditions
            k = self._determine_k(market_data, portfolio_state)
            
            # Calculate UCB scores for all opportunities
            ucb_scores = self._calculate_ucb_scores(opportunities)
            
            # Apply diversification constraints
            constrained_scores = self._apply_diversification_constraints(
                opportunities, ucb_scores, portfolio_state
            )
            
            # Select top K with exploration
            selected_opportunities = self._select_with_exploration(
                opportunities, constrained_scores, k
            )
            
            # Calculate diversification metrics
            diversification_metrics = self._calculate_diversification_metrics(selected_opportunities)
            
            # Calculate exploration bonus
            exploration_bonus = self._calculate_exploration_bonus(selected_opportunities)
            
            # Update performance tracking
            self._update_performance_tracking(opportunities, selected_opportunities)
            
            result = SelectionResult(
                selected_opportunities=selected_opportunities,
                selection_scores=constrained_scores,
                diversification_metrics=diversification_metrics,
                exploration_bonus=exploration_bonus,
                timestamp=datetime.now()
            )
            
            self.selection_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in top-K selection: {e}")
            return SelectionResult([], {}, {}, {}, datetime.now())
    
    def _determine_k(self, market_data: pd.DataFrame, 
                    portfolio_state: Dict[str, Any]) -> int:
        """Dynamically determine K based on market conditions"""
        try:
            base_k = self.base_k
            
            # Market volatility adjustment
            if market_data is not None and len(market_data) > 20:
                volatility = market_data['close'].pct_change().std()
                if volatility > 0.03:  # High volatility
                    base_k = min(self.max_k, base_k + 3)
                elif volatility < 0.01:  # Low volatility
                    base_k = max(self.min_k, base_k - 2)
            
            # Portfolio concentration adjustment
            if portfolio_state:
                current_concentration = portfolio_state.get('concentration_score', 0.5)
                if current_concentration > 0.7:  # High concentration
                    base_k = min(self.max_k, base_k + 2)
                elif current_concentration < 0.3:  # Low concentration
                    base_k = max(self.min_k, base_k - 1)
            
            # Market regime adjustment
            if market_data is not None and len(market_data) > 50:
                returns = market_data['close'].pct_change().dropna()
                momentum = returns.rolling(20).mean().iloc[-1]
                
                if momentum > 0.001:  # Bullish
                    base_k = min(self.max_k, base_k + 1)
                elif momentum < -0.001:  # Bearish
                    base_k = max(self.min_k, base_k - 1)
            
            return max(self.min_k, min(self.max_k, base_k))
            
        except Exception as e:
            logger.warning(f"Error determining K: {e}")
            return self.base_k
    
    def _calculate_ucb_scores(self, opportunities: List[Opportunity]) -> Dict[str, float]:
        """Calculate UCB1 scores for all opportunities"""
        try:
            ucb_scores = {}
            total_plays = sum(self.performance_tracking[opp.symbol]['count'] for opp in opportunities)
            
            for opportunity in opportunities:
                symbol = opportunity.symbol
                tracking = self.performance_tracking[symbol]
                
                if tracking['count'] == 0:
                    # New opportunity - high exploration value
                    ucb_score = opportunity.signal_strength + self.exploration_constant
                else:
                    # Calculate average reward
                    avg_reward = np.mean(tracking['rewards']) if tracking['rewards'] else 0
                    
                    # Calculate exploration bonus
                    exploration_bonus = self.exploration_constant * np.sqrt(
                        np.log(total_plays) / tracking['count']
                    )
                    
                    ucb_score = avg_reward + exploration_bonus
                
                # Combine with signal strength and confidence
                combined_score = (
                    ucb_score * 0.4 +
                    opportunity.signal_strength * 0.4 +
                    opportunity.confidence * 0.2
                )
                
                ucb_scores[symbol] = combined_score
            
            return ucb_scores
            
        except Exception as e:
            logger.warning(f"Error calculating UCB scores: {e}")
            return {opp.symbol: opp.signal_strength for opp in opportunities}
    
    def _apply_diversification_constraints(self, opportunities: List[Opportunity],
                                         ucb_scores: Dict[str, float],
                                         portfolio_state: Dict[str, Any]) -> Dict[str, float]:
        """Apply diversification constraints to scores"""
        try:
            constrained_scores = ucb_scores.copy()
            
            # Get current portfolio weights
            portfolio_weights = portfolio_state.get('sector_weights', {}) if portfolio_state else {}
            
            # Calculate sector weights in opportunities
            sector_weights = defaultdict(float)
            for opp in opportunities:
                if opp.sector:
                    sector_weights[opp.sector] += ucb_scores[opp.symbol]
            
            # Apply sector diversification penalty
            for opportunity in opportunities:
                if opportunity.sector:
                    sector_weight = sector_weights[opportunity.sector]
                    if sector_weight > self.max_sector_weight:
                        penalty = (sector_weight - self.max_sector_weight) * 0.5
                        constrained_scores[opportunity.symbol] -= penalty
            
            # Apply agent diversification penalty
            agent_weights = defaultdict(float)
            for opp in opportunities:
                agent_weights[opp.agent_id] += ucb_scores[opp.symbol]
            
            for opportunity in opportunities:
                agent_weight = agent_weights[opportunity.agent_id]
                if agent_weight > self.max_agent_weight:
                    penalty = (agent_weight - self.max_agent_weight) * 0.3
                    constrained_scores[opportunity.symbol] -= penalty
            
            # Apply market cap diversification
            market_cap_weights = defaultdict(float)
            for opp in opportunities:
                if opp.market_cap:
                    market_cap_weights[opp.market_cap] += ucb_scores[opp.symbol]
            
            for opportunity in opportunities:
                if opportunity.market_cap:
                    cap_weight = market_cap_weights[opportunity.market_cap]
                    if cap_weight > self.max_market_cap_weight:
                        penalty = (cap_weight - self.max_market_cap_weight) * 0.4
                        constrained_scores[opportunity.symbol] -= penalty
            
            return constrained_scores
            
        except Exception as e:
            logger.warning(f"Error applying diversification constraints: {e}")
            return ucb_scores
    
    def _select_with_exploration(self, opportunities: List[Opportunity],
                               scores: Dict[str, float], k: int) -> List[Opportunity]:
        """Select opportunities with exploration vs exploitation balance"""
        try:
            # Sort opportunities by score
            sorted_opportunities = sorted(
                opportunities, 
                key=lambda x: scores.get(x.symbol, x.signal_strength), 
                reverse=True
            )
            
            # Determine exploration vs exploitation
            if random.random() < self.exploration_rate:
                # Exploration: select some random opportunities
                num_explore = max(1, int(k * 0.3))  # 30% exploration
                num_exploit = k - num_explore
                
                # Select top opportunities for exploitation
                selected = sorted_opportunities[:num_exploit]
                
                # Add random opportunities for exploration
                remaining = sorted_opportunities[num_exploit:]
                if remaining:
                    explore_selection = random.sample(remaining, min(num_explore, len(remaining)))
                    selected.extend(explore_selection)
                
                # Ensure we have exactly k opportunities
                if len(selected) < k and len(sorted_opportunities) >= k:
                    additional = [opp for opp in sorted_opportunities if opp not in selected]
                    selected.extend(additional[:k - len(selected)])
                
            else:
                # Pure exploitation: select top k
                selected = sorted_opportunities[:k]
            
            return selected
            
        except Exception as e:
            logger.warning(f"Error in exploration selection: {e}")
            return sorted(opportunities, key=lambda x: scores.get(x.symbol, x.signal_strength), reverse=True)[:k]
    
    def _calculate_diversification_metrics(self, opportunities: List[Opportunity]) -> Dict[str, float]:
        """Calculate diversification metrics for selected opportunities"""
        try:
            if not opportunities:
                return {}
            
            # Sector diversification
            sectors = [opp.sector for opp in opportunities if opp.sector]
            sector_diversity = len(set(sectors)) / len(sectors) if sectors else 0
            
            # Agent diversification
            agents = [opp.agent_id for opp in opportunities]
            agent_diversity = len(set(agents)) / len(agents)
            
            # Market cap diversification
            market_caps = [opp.market_cap for opp in opportunities if opp.market_cap]
            cap_diversity = len(set(market_caps)) / len(market_caps) if market_caps else 0
            
            # Signal strength diversity
            signal_strengths = [opp.signal_strength for opp in opportunities]
            signal_diversity = 1.0 - np.std(signal_strengths) if len(signal_strengths) > 1 else 0
            
            return {
                'sector_diversity': sector_diversity,
                'agent_diversity': agent_diversity,
                'market_cap_diversity': cap_diversity,
                'signal_diversity': signal_diversity,
                'overall_diversity': (sector_diversity + agent_diversity + cap_diversity + signal_diversity) / 4
            }
            
        except Exception as e:
            logger.warning(f"Error calculating diversification metrics: {e}")
            return {}
    
    def _calculate_exploration_bonus(self, opportunities: List[Opportunity]) -> Dict[str, float]:
        """Calculate exploration bonus for selected opportunities"""
        try:
            exploration_bonus = {}
            
            for opportunity in opportunities:
                symbol = opportunity.symbol
                tracking = self.performance_tracking[symbol]
                
                if tracking['count'] == 0:
                    # New opportunity - high exploration value
                    exploration_bonus[symbol] = 1.0
                else:
                    # Calculate exploration value based on play count
                    exploration_value = 1.0 / (1.0 + tracking['count'])
                    exploration_bonus[symbol] = exploration_value
            
            return exploration_bonus
            
        except Exception as e:
            logger.warning(f"Error calculating exploration bonus: {e}")
            return {}
    
    def _update_performance_tracking(self, all_opportunities: List[Opportunity],
                                   selected_opportunities: List[Opportunity]):
        """Update performance tracking for opportunities"""
        try:
            # Update play counts
            for opportunity in selected_opportunities:
                self.performance_tracking[opportunity.symbol]['count'] += 1
            
            # Store all opportunities for potential reward updates
            for opportunity in all_opportunities:
                self.opportunity_history[opportunity.symbol].append({
                    'timestamp': opportunity.timestamp,
                    'signal_strength': opportunity.signal_strength,
                    'confidence': opportunity.confidence,
                    'expected_return': opportunity.expected_return,
                    'risk_score': opportunity.risk_score
                })
            
            # Keep only recent history
            max_history = 100
            for symbol in self.opportunity_history:
                if len(self.opportunity_history[symbol]) > max_history:
                    self.opportunity_history[symbol] = self.opportunity_history[symbol][-max_history:]
            
        except Exception as e:
            logger.warning(f"Error updating performance tracking: {e}")
    
    def update_rewards(self, symbol: str, reward: float):
        """Update reward for a specific symbol"""
        try:
            if symbol in self.performance_tracking:
                self.performance_tracking[symbol]['rewards'].append(reward)
                
                # Keep only recent rewards
                max_rewards = 50
                if len(self.performance_tracking[symbol]['rewards']) > max_rewards:
                    self.performance_tracking[symbol]['rewards'] = self.performance_tracking[symbol]['rewards'][-max_rewards:]
            
        except Exception as e:
            logger.warning(f"Error updating reward: {e}")
    
    def adjust_exploration_rate(self, market_conditions: Dict[str, Any]):
        """Adjust exploration rate based on market conditions"""
        try:
            # Increase exploration in uncertain markets
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.03:
                self.exploration_rate = min(self.max_exploration_rate, self.exploration_rate + 0.05)
            elif volatility < 0.01:
                self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - 0.03)
            
            # Adjust based on recent performance
            recent_performance = self._calculate_recent_performance()
            if recent_performance < 0.5:  # Poor performance
                self.exploration_rate = min(self.max_exploration_rate, self.exploration_rate + 0.1)
            elif recent_performance > 0.7:  # Good performance
                self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - 0.05)
            
        except Exception as e:
            logger.warning(f"Error adjusting exploration rate: {e}")
    
    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance across all opportunities"""
        try:
            recent_rewards = []
            for symbol, tracking in self.performance_tracking.items():
                if tracking['rewards']:
                    recent_rewards.extend(tracking['rewards'][-10:])  # Last 10 rewards
            
            if recent_rewards:
                return np.mean(recent_rewards)
            return 0.5  # Default performance
            
        except Exception as e:
            logger.warning(f"Error calculating recent performance: {e}")
            return 0.5
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of selection performance"""
        try:
            total_selections = len(self.selection_history)
            if total_selections == 0:
                return {}
            
            # Calculate average diversification
            avg_diversification = np.mean([
                result.diversification_metrics.get('overall_diversity', 0)
                for result in self.selection_history
            ])
            
            # Calculate exploration rate over time
            exploration_rates = []
            for result in self.selection_history[-20:]:  # Last 20 selections
                exploration_bonus = result.exploration_bonus
                if exploration_bonus:
                    avg_exploration = np.mean(list(exploration_bonus.values()))
                    exploration_rates.append(avg_exploration)
            
            avg_exploration = np.mean(exploration_rates) if exploration_rates else 0
            
            return {
                'total_selections': total_selections,
                'avg_diversification': avg_diversification,
                'avg_exploration': avg_exploration,
                'current_exploration_rate': self.exploration_rate,
                'current_k': self.current_k,
                'opportunities_tracked': len(self.performance_tracking)
            }
            
        except Exception as e:
            logger.warning(f"Error getting selection summary: {e}")
            return {}
