"""
Enhanced Unified Opportunity Scorer with Realistic Market-Based Scoring
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from common.opportunity_store import Opportunity


@dataclass
class MarketConditions:
    """Market condition factors for scoring"""
    volatility: float  # Market volatility (0-1)
    trend_strength: float  # Overall market trend (0-1)
    volume_profile: float  # Volume conditions (0-1)
    sector_performance: float  # Sector relative performance (0-1)


class EnhancedUnifiedOpportunityScorer:
    """
    Enhanced opportunity scorer with realistic market-based algorithms
    """
    
    def __init__(self):
        # Enhanced agent weights based on historical performance
        self.agent_weights = {
            'technical': 0.25,      # Technical analysis
            'sentiment': 0.15,      # Sentiment analysis
            'flow': 0.20,           # Order flow analysis
            'macro': 0.10,          # Macroeconomic factors
            'money_flows': 0.15,    # Money flow analysis
            'value_analysis': 0.10, # Fundamental analysis
            'insider': 0.05         # Insider activity
        }
        
        # Enhanced opportunity type weights
        self.opportunity_type_weights = {
            'imbalance': 0.30,      # Market imbalances
            'trend': 0.25,          # Trend following
            'liquidity': 0.20,      # Liquidity sweeps
            'breakout': 0.15,       # Breakout patterns
            'reversal': 0.10        # Reversal patterns
        }
        
        # Enhanced time horizon weights
        self.time_horizon_weights = {
            'intraday': 0.15,       # < 1 day
            'swing': 0.35,          # 1-5 days
            'position': 0.30,       # 1-4 weeks
            'long_term': 0.20       # > 1 month
        }
        
        # Market condition weights
        self.market_condition_weights = {
            'volatility': 0.25,
            'trend_strength': 0.30,
            'volume_profile': 0.20,
            'sector_performance': 0.25
        }
    
    def calculate_priority_score(self, opportunity: Opportunity) -> float:
        """
        Calculate enhanced priority score with market conditions
        """
        try:
            # Base score components
            agent_score = self._calculate_agent_score(opportunity)
            opportunity_score = self._calculate_opportunity_score(opportunity)
            time_score = self._calculate_time_score(opportunity)
            confidence_score = opportunity.confidence
            upside_score = self._calculate_upside_score(opportunity)
            recency_score = self._calculate_recency_score(opportunity)
            volatility_score = self._calculate_volatility_score(opportunity)
            
            # Market conditions score
            market_score = self._calculate_market_conditions_score(opportunity)
            
            # Enhanced weighted combination
            base_score = (
                agent_score * 0.20 +
                opportunity_score * 0.20 +
                time_score * 0.15 +
                confidence_score * 0.20 +
                upside_score * 0.15 +
                recency_score * 0.05 +
                volatility_score * 0.05
            )
            
            # Apply market conditions multiplier
            final_score = base_score * (0.7 + 0.3 * market_score)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error calculating priority score: {e}")
            return 0.5
    
    def _calculate_agent_score(self, opportunity: Opportunity) -> float:
        """Calculate agent-specific score"""
        agent_weight = self.agent_weights.get(opportunity.agent_type, 0.1)
        
        # Normalize to 0-1 range
        max_weight = max(self.agent_weights.values())
        return agent_weight / max_weight
    
    def _calculate_opportunity_score(self, opportunity: Opportunity) -> float:
        """Calculate opportunity type score"""
        # Extract opportunity type from entry reason
        opportunity_type = self._extract_opportunity_type(opportunity.entry_reason)
        type_weight = self.opportunity_type_weights.get(opportunity_type, 0.1)
        
        # Normalize to 0-1 range
        max_weight = max(self.opportunity_type_weights.values())
        return type_weight / max_weight
    
    def _calculate_time_score(self, opportunity: Opportunity) -> float:
        """Calculate time horizon score"""
        time_horizon = self._extract_time_horizon(opportunity.time_horizon)
        horizon_weight = self.time_horizon_weights.get(time_horizon, 0.2)
        
        # Normalize to 0-1 range
        max_weight = max(self.time_horizon_weights.values())
        return horizon_weight / max_weight
    
    def _calculate_upside_score(self, opportunity: Opportunity) -> float:
        """Calculate upside potential score"""
        # Normalize upside potential to 0-1 range
        # Assuming upside_potential is percentage (0-100)
        upside = opportunity.upside_potential
        if upside <= 0:
            return 0.0
        elif upside >= 50:  # 50%+ upside is maximum score
            return 1.0
        else:
            return upside / 50.0
    
    def _calculate_recency_score(self, opportunity: Opportunity) -> float:
        """Calculate recency score based on discovery time"""
        try:
            # Calculate hours since discovery
            hours_since = (datetime.now() - opportunity.discovered_at).total_seconds() / 3600
            
            # Score decreases over time
            if hours_since <= 1:  # Within 1 hour
                return 1.0
            elif hours_since <= 24:  # Within 1 day
                return 0.8
            elif hours_since <= 72:  # Within 3 days
                return 0.6
            elif hours_since <= 168:  # Within 1 week
                return 0.4
            else:
                return 0.2
        except:
            return 0.5
    
    def _calculate_volatility_score(self, opportunity: Opportunity) -> float:
        """Calculate volatility-adjusted score"""
        # Higher volatility can be good for some strategies
        # but we want to penalize extreme volatility
        volatility = getattr(opportunity, 'volatility', 0.02)  # Default 2%
        
        if volatility <= 0.01:  # Low volatility
            return 0.7
        elif volatility <= 0.03:  # Normal volatility
            return 1.0
        elif volatility <= 0.05:  # High volatility
            return 0.8
        else:  # Extreme volatility
            return 0.5
    
    def _calculate_market_conditions_score(self, opportunity: Opportunity) -> float:
        """Calculate market conditions score"""
        try:
            # Get market conditions (in real implementation, this would come from market data)
            market_conditions = self._get_market_conditions(opportunity.ticker)
            
            # Calculate weighted market score
            market_score = (
                market_conditions.volatility * self.market_condition_weights['volatility'] +
                market_conditions.trend_strength * self.market_condition_weights['trend_strength'] +
                market_conditions.volume_profile * self.market_condition_weights['volume_profile'] +
                market_conditions.sector_performance * self.market_condition_weights['sector_performance']
            )
            
            return market_score
            
        except Exception as e:
            print(f"Error calculating market conditions score: {e}")
            return 0.5
    
    def _get_market_conditions(self, ticker: str) -> MarketConditions:
        """Get current market conditions for a ticker"""
        # In real implementation, this would fetch from market data APIs
        # For now, return realistic mock conditions
        
        # Simulate different market conditions based on ticker
        np.random.seed(hash(ticker) % 1000)
        
        return MarketConditions(
            volatility=np.random.uniform(0.1, 0.4),  # 10-40% volatility
            trend_strength=np.random.uniform(0.3, 0.8),  # 30-80% trend strength
            volume_profile=np.random.uniform(0.4, 0.9),  # 40-90% volume profile
            sector_performance=np.random.uniform(0.2, 0.9)  # 20-90% sector performance
        )
    
    def _extract_opportunity_type(self, entry_reason: str) -> str:
        """Extract opportunity type from entry reason"""
        entry_reason_lower = entry_reason.lower()
        
        if any(word in entry_reason_lower for word in ['imbalance', 'gap', 'fvg']):
            return 'imbalance'
        elif any(word in entry_reason_lower for word in ['trend', 'momentum', 'breakout']):
            return 'trend'
        elif any(word in entry_reason_lower for word in ['liquidity', 'sweep', 'wick']):
            return 'liquidity'
        elif any(word in entry_reason_lower for word in ['breakout', 'break']):
            return 'breakout'
        elif any(word in entry_reason_lower for word in ['reversal', 'divergence']):
            return 'reversal'
        else:
            return 'trend'  # Default
    
    def _extract_time_horizon(self, time_horizon: str) -> str:
        """Extract time horizon from time_horizon string"""
        time_horizon_lower = time_horizon.lower()
        
        if any(word in time_horizon_lower for word in ['intraday', 'day', '1d']):
            return 'intraday'
        elif any(word in time_horizon_lower for word in ['swing', 'short', '1-5d']):
            return 'swing'
        elif any(word in time_horizon_lower for word in ['position', 'medium', '1-4w']):
            return 'position'
        elif any(word in time_horizon_lower for word in ['long', 'long-term', '1m+']):
            return 'long_term'
        else:
            return 'swing'  # Default
    
    def rank_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Rank opportunities by priority score"""
        for opportunity in opportunities:
            opportunity.priority_score = self.calculate_priority_score(opportunity)
        
        return sorted(opportunities, key=lambda x: x.priority_score, reverse=True)
    
    def get_top_opportunities(self, opportunities: List[Opportunity], 
                            top_n: int = 10) -> List[Opportunity]:
        """Get top N opportunities"""
        ranked = self.rank_opportunities(opportunities)
        return ranked[:top_n]
    
    def calculate_portfolio_metrics(self, opportunities: List[Opportunity]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        if not opportunities:
            return {
                'total_opportunities': 0,
                'average_score': 0.0,
                'score_distribution': {},
                'agent_distribution': {},
                'risk_metrics': {}
            }
        
        # Calculate scores
        for opp in opportunities:
            opp.priority_score = self.calculate_priority_score(opp)
        
        scores = [opp.priority_score for opp in opportunities]
        
        # Score distribution
        score_ranges = {
            'High (0.8-1.0)': len([s for s in scores if s >= 0.8]),
            'Medium-High (0.6-0.8)': len([s for s in scores if 0.6 <= s < 0.8]),
            'Medium (0.4-0.6)': len([s for s in scores if 0.4 <= s < 0.6]),
            'Low (0.2-0.4)': len([s for s in scores if 0.2 <= s < 0.4]),
            'Very Low (0.0-0.2)': len([s for s in scores if s < 0.2])
        }
        
        # Agent distribution
        agent_counts = {}
        for opp in opportunities:
            agent_counts[opp.agent_type] = agent_counts.get(opp.agent_type, 0) + 1
        
        # Risk metrics
        confidences = [opp.confidence for opp in opportunities]
        upsides = [opp.upside_potential for opp in opportunities]
        
        return {
            'total_opportunities': len(opportunities),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'score_distribution': score_ranges,
            'agent_distribution': agent_counts,
            'risk_metrics': {
                'average_confidence': np.mean(confidences),
                'average_upside': np.mean(upsides),
                'high_confidence_ratio': len([c for c in confidences if c >= 0.7]) / len(confidences)
            }
        }
