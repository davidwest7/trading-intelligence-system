#!/usr/bin/env python3
"""
Opportunity Builder: Merge + Costs + Constraints

This component builds final trading opportunities by merging signals from
Meta-Weighter and Top-K Selector, applying costs and constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from coordination.meta_weighter import BlendedSignal
from coordination.top_k_selector import Opportunity as TopKOpportunity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TradingOpportunity:
    """Final trading opportunity with all constraints and costs applied"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float
    confidence: float
    expected_return: float
    risk_score: float
    position_size: float
    entry_price: float
    target_price: float
    stop_loss: float
    transaction_cost: float
    market_impact: float
    total_cost: float
    net_expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    agent_contributions: Dict[str, float]
    constraints_satisfied: bool
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class OpportunityBuildResult:
    """Result of opportunity building process"""
    opportunities: List[TradingOpportunity]
    build_metrics: Dict[str, Any]
    constraint_violations: List[str]
    cost_analysis: Dict[str, float]
    timestamp: datetime

class OpportunityBuilder:
    """
    Opportunity Builder: Merge + Costs + Constraints
    
    Features:
    - Merge signals from Meta-Weighter and Top-K Selector
    - Apply transaction costs and market impact
    - Enforce position sizing constraints
    - Calculate risk-adjusted returns
    - Validate against portfolio constraints
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cost parameters
        self.commission_rate = self.config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = self.config.get('slippage_rate', 0.0005)     # 0.05%
        self.market_impact_rate = self.config.get('market_impact_rate', 0.0001)  # 0.01% per $100K
        
        # Position sizing constraints
        self.max_position_size = self.config.get('max_position_size', 0.10)  # 10% of portfolio
        self.min_position_size = self.config.get('min_position_size', 0.01)  # 1% of portfolio
        self.max_sector_exposure = self.config.get('max_sector_exposure', 0.25)  # 25% per sector
        self.max_agent_exposure = self.config.get('max_agent_exposure', 0.20)   # 20% per agent
        
        # Risk constraints
        self.max_risk_per_position = self.config.get('max_risk_per_position', 0.02)  # 2% risk per position
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.15)        # 15% total portfolio risk
        self.min_sharpe_ratio = self.config.get('min_sharpe_ratio', 0.5)            # Minimum Sharpe ratio
        
        # Market data
        self.current_prices = {}
        self.volatility_data = {}
        self.liquidity_data = {}
        
        # Build history
        self.build_history = []
        self.constraint_violations = []
        
        logger.info("Opportunity Builder initialized with cost and constraint parameters")
    
    def build_opportunities(self, blended_signals: List[BlendedSignal],
                          top_k_opportunities: List[TopKOpportunity],
                          market_data: pd.DataFrame,
                          portfolio_state: Dict[str, Any]) -> OpportunityBuildResult:
        """Build final trading opportunities"""
        try:
            # Merge signals from both sources
            merged_opportunities = self._merge_signals(blended_signals, top_k_opportunities)
            
            # Apply market data
            self._update_market_data(market_data)
            
            # Calculate costs and constraints
            opportunities_with_costs = self._apply_costs_and_constraints(
                merged_opportunities, portfolio_state
            )
            
            # Validate against portfolio constraints
            validated_opportunities = self._validate_portfolio_constraints(
                opportunities_with_costs, portfolio_state
            )
            
            # Calculate final metrics
            final_opportunities = self._calculate_final_metrics(validated_opportunities)
            
            # Generate build metrics
            build_metrics = self._calculate_build_metrics(final_opportunities)
            
            # Analyze costs
            cost_analysis = self._analyze_costs(final_opportunities)
            
            result = OpportunityBuildResult(
                opportunities=final_opportunities,
                build_metrics=build_metrics,
                constraint_violations=self.constraint_violations[-10:],  # Last 10 violations
                cost_analysis=cost_analysis,
                timestamp=datetime.now()
            )
            
            self.build_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error building opportunities: {e}")
            return OpportunityBuildResult([], {}, [], {}, datetime.now())
    
    def _merge_signals(self, blended_signals: List[BlendedSignal],
                      top_k_opportunities: List[TopKOpportunity]) -> List[Dict[str, Any]]:
        """Merge signals from Meta-Weighter and Top-K Selector"""
        try:
            merged = []
            
            # Create lookup for blended signals
            blended_lookup = {signal.symbol: signal for signal in blended_signals}
            
            for top_k_opp in top_k_opportunities:
                symbol = top_k_opp.symbol
                blended_signal = blended_lookup.get(symbol)
                
                if blended_signal:
                    # Merge signals with weighted average
                    merged_signal = {
                        'symbol': symbol,
                        'signal_strength': (blended_signal.blended_strength * 0.6 + 
                                          top_k_opp.signal_strength * 0.4),
                        'confidence': (blended_signal.confidence * 0.7 + 
                                     top_k_opp.confidence * 0.3),
                        'expected_return': top_k_opp.expected_return,
                        'risk_score': top_k_opp.risk_score,
                        'agent_contributions': blended_signal.agent_contributions,
                        'consensus_score': blended_signal.consensus_score,
                        'disagreement_score': blended_signal.disagreement_score,
                        'metadata': {**blended_signal.metadata, **top_k_opp.metadata}
                    }
                else:
                    # Use Top-K signal only
                    merged_signal = {
                        'symbol': symbol,
                        'signal_strength': top_k_opp.signal_strength,
                        'confidence': top_k_opp.confidence,
                        'expected_return': top_k_opp.expected_return,
                        'risk_score': top_k_opp.risk_score,
                        'agent_contributions': {top_k_opp.agent_id: 1.0},
                        'consensus_score': top_k_opp.signal_strength,
                        'disagreement_score': 0.0,
                        'metadata': top_k_opp.metadata
                    }
                
                merged.append(merged_signal)
            
            return merged
            
        except Exception as e:
            logger.warning(f"Error merging signals: {e}")
            return []
    
    def _update_market_data(self, market_data: pd.DataFrame):
        """Update current market data"""
        try:
            if market_data is not None and len(market_data) > 0:
                # Update current prices
                latest_data = market_data.groupby('symbol').tail(1)
                self.current_prices = dict(zip(latest_data['symbol'], latest_data['close']))
                
                # Calculate volatility
                for symbol in market_data['symbol'].unique():
                    symbol_data = market_data[market_data['symbol'] == symbol]
                    if len(symbol_data) > 20:
                        returns = symbol_data['close'].pct_change().dropna()
                        self.volatility_data[symbol] = returns.std()
                
                # Estimate liquidity (using volume as proxy)
                for symbol in market_data['symbol'].unique():
                    symbol_data = market_data[market_data['symbol'] == symbol]
                    if len(symbol_data) > 0:
                        avg_volume = symbol_data['volume'].mean()
                        self.liquidity_data[symbol] = avg_volume
                        
        except Exception as e:
            logger.warning(f"Error updating market data: {e}")
    
    def _apply_costs_and_constraints(self, opportunities: List[Dict[str, Any]],
                                   portfolio_state: Dict[str, Any]) -> List[TradingOpportunity]:
        """Apply transaction costs and position sizing constraints"""
        try:
            trading_opportunities = []
            portfolio_value = portfolio_state.get('total_value', 1000000)
            
            for opp in opportunities:
                symbol = opp['symbol']
                current_price = self.current_prices.get(symbol, 100)
                
                # Determine action based on signal strength
                if opp['signal_strength'] > 0.3:
                    action = 'BUY'
                elif opp['signal_strength'] < -0.3:
                    action = 'SELL'
                else:
                    action = 'HOLD'
                
                # Calculate position size based on Kelly Criterion
                position_size = self._calculate_position_size(opp, portfolio_value)
                
                # Apply position size constraints
                position_size = max(self.min_position_size, 
                                  min(self.max_position_size, position_size))
                
                # Calculate transaction costs
                transaction_cost = self._calculate_transaction_cost(
                    symbol, position_size, current_price
                )
                
                # Calculate market impact
                market_impact = self._calculate_market_impact(
                    symbol, position_size, current_price
                )
                
                # Calculate total cost
                total_cost = transaction_cost + market_impact
                
                # Calculate target and stop loss
                target_price, stop_loss = self._calculate_target_and_stop(
                    current_price, opp['signal_strength'], opp['risk_score']
                )
                
                # Calculate net expected return
                gross_return = opp['expected_return']
                net_return = gross_return - (total_cost / (position_size * current_price))
                
                # Calculate risk metrics
                sharpe_ratio = self._calculate_sharpe_ratio(net_return, opp['risk_score'])
                max_drawdown = self._calculate_max_drawdown(opp['risk_score'])
                
                # Check if constraints are satisfied
                constraints_satisfied = self._check_constraints(
                    opp, position_size, portfolio_state
                )
                
                trading_opp = TradingOpportunity(
                    symbol=symbol,
                    action=action,
                    signal_strength=opp['signal_strength'],
                    confidence=opp['confidence'],
                    expected_return=opp['expected_return'],
                    risk_score=opp['risk_score'],
                    position_size=position_size,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    transaction_cost=transaction_cost,
                    market_impact=market_impact,
                    total_cost=total_cost,
                    net_expected_return=net_return,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    agent_contributions=opp['agent_contributions'],
                    constraints_satisfied=constraints_satisfied,
                    timestamp=datetime.now(),
                    metadata=opp['metadata']
                )
                
                trading_opportunities.append(trading_opp)
            
            return trading_opportunities
            
        except Exception as e:
            logger.warning(f"Error applying costs and constraints: {e}")
            return []
    
    def _calculate_position_size(self, opportunity: Dict[str, Any], 
                               portfolio_value: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            
            expected_return = opportunity['expected_return']
            risk_score = opportunity['risk_score']
            confidence = opportunity['confidence']
            
            # Estimate win probability based on confidence and signal strength
            win_prob = 0.5 + (opportunity['signal_strength'] * confidence * 0.3)
            win_prob = max(0.1, min(0.9, win_prob))
            
            # Estimate odds (potential gain vs potential loss)
            potential_gain = abs(expected_return)
            potential_loss = risk_score
            
            if potential_loss > 0:
                odds = potential_gain / potential_loss
                kelly_fraction = (odds * win_prob - (1 - win_prob)) / odds
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = 0.05  # Default 5%
            
            return kelly_fraction
            
        except Exception as e:
            logger.warning(f"Error calculating position size: {e}")
            return 0.05  # Default 5%
    
    def _calculate_transaction_cost(self, symbol: str, position_size: float, 
                                  price: float) -> float:
        """Calculate transaction costs"""
        try:
            position_value = position_size * price
            
            # Commission
            commission = position_value * self.commission_rate
            
            # Slippage
            slippage = position_value * self.slippage_rate
            
            return commission + slippage
            
        except Exception as e:
            logger.warning(f"Error calculating transaction cost: {e}")
            return 0
    
    def _calculate_market_impact(self, symbol: str, position_size: float, 
                               price: float) -> float:
        """Calculate market impact cost"""
        try:
            position_value = position_size * price
            
            # Market impact increases with position size
            impact_rate = self.market_impact_rate * (position_value / 100000)  # Per $100K
            
            return position_value * impact_rate
            
        except Exception as e:
            logger.warning(f"Error calculating market impact: {e}")
            return 0
    
    def _calculate_target_and_stop(self, current_price: float, signal_strength: float,
                                 risk_score: float) -> Tuple[float, float]:
        """Calculate target price and stop loss"""
        try:
            # Target based on signal strength and volatility
            volatility = self.volatility_data.get('symbol', 0.02)
            target_multiplier = 1.0 + (abs(signal_strength) * volatility * 2)
            
            if signal_strength > 0:  # Long position
                target_price = current_price * target_multiplier
                stop_loss = current_price * (1 - risk_score)
            else:  # Short position
                target_price = current_price / target_multiplier
                stop_loss = current_price * (1 + risk_score)
            
            return target_price, stop_loss
            
        except Exception as e:
            logger.warning(f"Error calculating target and stop: {e}")
            return current_price * 1.05, current_price * 0.95
    
    def _calculate_sharpe_ratio(self, expected_return: float, risk_score: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            if risk_score > 0:
                return expected_return / risk_score
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_max_drawdown(self, risk_score: float) -> float:
        """Estimate maximum drawdown"""
        try:
            # Simple estimation based on risk score
            return risk_score * 2  # 2x risk score as max drawdown
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0.1
    
    def _check_constraints(self, opportunity: Dict[str, Any], position_size: float,
                          portfolio_state: Dict[str, Any]) -> bool:
        """Check if opportunity satisfies all constraints"""
        try:
            # Check Sharpe ratio
            if opportunity['expected_return'] / opportunity['risk_score'] < self.min_sharpe_ratio:
                self.constraint_violations.append(f"Low Sharpe ratio for {opportunity['symbol']}")
                return False
            
            # Check position size
            if position_size > self.max_position_size:
                self.constraint_violations.append(f"Position size too large for {opportunity['symbol']}")
                return False
            
            # Check sector exposure (if available)
            sector = opportunity['metadata'].get('sector', '')
            if sector and portfolio_state.get('sector_weights', {}).get(sector, 0) > self.max_sector_exposure:
                self.constraint_violations.append(f"Sector exposure limit for {opportunity['symbol']}")
                return False
            
            # Check agent exposure
            agent_id = list(opportunity['agent_contributions'].keys())[0] if opportunity['agent_contributions'] else ''
            if agent_id and portfolio_state.get('agent_weights', {}).get(agent_id, 0) > self.max_agent_exposure:
                self.constraint_violations.append(f"Agent exposure limit for {opportunity['symbol']}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking constraints: {e}")
            return False
    
    def _validate_portfolio_constraints(self, opportunities: List[TradingOpportunity],
                                      portfolio_state: Dict[str, Any]) -> List[TradingOpportunity]:
        """Validate opportunities against portfolio-level constraints"""
        try:
            validated = []
            total_risk = 0
            
            for opp in opportunities:
                # Check portfolio risk limit
                position_risk = opp.position_size * opp.risk_score
                if total_risk + position_risk <= self.max_portfolio_risk:
                    validated.append(opp)
                    total_risk += position_risk
                else:
                    self.constraint_violations.append(f"Portfolio risk limit for {opp.symbol}")
            
            return validated
            
        except Exception as e:
            logger.warning(f"Error validating portfolio constraints: {e}")
            return opportunities
    
    def _calculate_final_metrics(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Calculate final metrics for opportunities"""
        try:
            for opp in opportunities:
                # Recalculate Sharpe ratio with net return
                if opp.risk_score > 0:
                    opp.sharpe_ratio = opp.net_expected_return / opp.risk_score
                
                # Update metadata with final calculations
                opp.metadata.update({
                    'cost_ratio': opp.total_cost / (opp.position_size * opp.entry_price),
                    'risk_adjusted_return': opp.net_expected_return / opp.risk_score if opp.risk_score > 0 else 0,
                    'position_value': opp.position_size * opp.entry_price
                })
            
            return opportunities
            
        except Exception as e:
            logger.warning(f"Error calculating final metrics: {e}")
            return opportunities
    
    def _calculate_build_metrics(self, opportunities: List[TradingOpportunity]) -> Dict[str, Any]:
        """Calculate build process metrics"""
        try:
            if not opportunities:
                return {}
            
            total_opportunities = len(opportunities)
            valid_opportunities = sum(1 for opp in opportunities if opp.constraints_satisfied)
            
            avg_signal_strength = np.mean([opp.signal_strength for opp in opportunities])
            avg_confidence = np.mean([opp.confidence for opp in opportunities])
            avg_expected_return = np.mean([opp.net_expected_return for opp in opportunities])
            avg_sharpe = np.mean([opp.sharpe_ratio for opp in opportunities])
            
            total_cost = sum(opp.total_cost for opp in opportunities)
            total_value = sum(opp.position_size * opp.entry_price for opp in opportunities)
            cost_ratio = total_cost / total_value if total_value > 0 else 0
            
            return {
                'total_opportunities': total_opportunities,
                'valid_opportunities': valid_opportunities,
                'validation_rate': valid_opportunities / total_opportunities if total_opportunities > 0 else 0,
                'avg_signal_strength': avg_signal_strength,
                'avg_confidence': avg_confidence,
                'avg_expected_return': avg_expected_return,
                'avg_sharpe_ratio': avg_sharpe,
                'total_cost': total_cost,
                'total_value': total_value,
                'cost_ratio': cost_ratio
            }
            
        except Exception as e:
            logger.warning(f"Error calculating build metrics: {e}")
            return {}
    
    def _analyze_costs(self, opportunities: List[TradingOpportunity]) -> Dict[str, float]:
        """Analyze cost breakdown"""
        try:
            total_transaction_cost = sum(opp.transaction_cost for opp in opportunities)
            total_market_impact = sum(opp.market_impact for opp in opportunities)
            total_cost = sum(opp.total_cost for opp in opportunities)
            total_value = sum(opp.position_size * opp.entry_price for opp in opportunities)
            
            return {
                'total_transaction_cost': total_transaction_cost,
                'total_market_impact': total_market_impact,
                'total_cost': total_cost,
                'total_value': total_value,
                'transaction_cost_ratio': total_transaction_cost / total_value if total_value > 0 else 0,
                'market_impact_ratio': total_market_impact / total_value if total_value > 0 else 0,
                'total_cost_ratio': total_cost / total_value if total_value > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing costs: {e}")
            return {}
    
    def get_build_summary(self) -> Dict[str, Any]:
        """Get summary of opportunity building performance"""
        try:
            total_builds = len(self.build_history)
            if total_builds == 0:
                return {}
            
            # Calculate average metrics
            avg_validation_rate = np.mean([
                build.build_metrics.get('validation_rate', 0) 
                for build in self.build_history
            ])
            
            avg_cost_ratio = np.mean([
                build.cost_analysis.get('total_cost_ratio', 0)
                for build in self.build_history
            ])
            
            avg_expected_return = np.mean([
                build.build_metrics.get('avg_expected_return', 0)
                for build in self.build_history
            ])
            
            return {
                'total_builds': total_builds,
                'avg_validation_rate': avg_validation_rate,
                'avg_cost_ratio': avg_cost_ratio,
                'avg_expected_return': avg_expected_return,
                'total_constraint_violations': len(self.constraint_violations)
            }
            
        except Exception as e:
            logger.warning(f"Error getting build summary: {e}")
            return {}
