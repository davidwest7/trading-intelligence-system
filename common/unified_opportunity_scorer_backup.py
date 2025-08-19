"""
Unified Opportunity Scorer - Ranks opportunities across all agents
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import math

class UnifiedOpportunityScorer:
    """Unified scoring system for opportunities across all agents"""
    
    def __init__(self):
        # Agent weights (can be adjusted based on performance)
        self.agent_weights = {
            'value_analysis': 0.20,      # Fundamental analysis
            'technical_analysis': 0.15,  # Technical patterns
            'money_flows': 0.15,         # Institutional flows
            'insider_analysis': 0.12,    # Insider activity
            'sentiment_analysis': 0.10,  # Market sentiment
            'macro_analysis': 0.08,      # Macro factors
            'flow_analysis': 0.08,       # Flow analysis
            'top_performers_analysis': 0.05,  # Top performers
            'undervalued_analysis': 0.03,     # Undervalued (subset of value)
            'causal_analysis': 0.02,          # Causal impact
            'hedging_analysis': 0.01,         # Hedging strategies
            'learning_analysis': 0.01         # Learning insights
        }
        
        # Opportunity type weights
        self.opportunity_type_weights = {
            'Value': 1.0,        # Strong fundamentals
            'Technical': 0.8,    # Technical patterns
            'Flow': 0.9,         # Institutional flows
            'Insider': 0.7,      # Insider activity
            'Sentiment': 0.6,    # Sentiment signals
            'Macro': 0.5,        # Macro factors
            'Top Performer': 0.8, # Top performers
            'Undervalued': 0.9,   # Undervalued stocks
            'Causal': 0.7,        # Causal impact
            'Hedging': 0.4,       # Hedging strategies
            'Learning': 0.6       # Learning insights
        }
        
        # Time horizon weights (shorter = higher weight)
        self.time_horizon_weights = {
            '1-2 weeks': 1.0,
            '2-4 weeks': 0.95,
            '1-3 months': 0.9,
            '1-4 weeks': 0.85,
            '1-6 months': 0.8,
            '3-6 months': 0.75,
            '6-12 months': 0.7,
            '12-18 months': 0.65,
            '18+ months': 0.6
        }
    
    def calculate_priority_score(self, opportunity: Any) -> float:
        """Calculate unified priority score for an opportunity"""
        try:
            # Base score from agent type
            agent_weight = self.agent_weights.get(opportunity.agent_type, 0.5)
            
            # Opportunity type weight
            type_weight = self.opportunity_type_weights.get(opportunity.opportunity_type, 0.5)
            
            # Time horizon weight
            time_weight = self.time_horizon_weights.get(opportunity.time_horizon, 0.5)
            
            # Upside potential (normalized to 0-1)
            upside_score = min(opportunity.upside_potential, 1.0)
            
            # Confidence score
            confidence_score = opportunity.confidence
            
            # Recency score (newer opportunities get higher scores)
            recency_score = self._calculate_recency_score(opportunity.discovered_at)
            
            # Volatility adjustment (if available in raw_data)
            volatility_score = self._calculate_volatility_score(opportunity.raw_data)
            
            # Calculate composite score
            base_score = (
                agent_weight * 0.25 +
                type_weight * 0.20 +
                time_weight * 0.15 +
                upside_score * 0.20 +
                confidence_score * 0.15 +
                recency_score * 0.05
            )
            
            # Apply volatility adjustment
            final_score = base_score * volatility_score
            
            # Normalize to 0-1 range
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error calculating priority score: {e}")
            return 0.0
    
    def _calculate_recency_score(self, discovered_at: datetime) -> float:
        """Calculate recency score (newer = higher score)"""
        try:
            age_hours = (datetime.now() - discovered_at).total_seconds() / 3600
            
            if age_hours < 1:  # Less than 1 hour
                return 1.0
            elif age_hours < 24:  # Less than 1 day
                return 0.9
            elif age_hours < 168:  # Less than 1 week
                return 0.8
            elif age_hours < 720:  # Less than 1 month
                return 0.7
            else:  # Older than 1 month
                return 0.5
        except:
            return 0.5
    
    def _calculate_volatility_score(self, raw_data: Dict[str, Any]) -> float:
        """Calculate volatility adjustment score"""
        try:
            # Extract volatility-related data if available
            volatility = raw_data.get('volatility', 0.0)
            volume_anomaly = raw_data.get('volume_anomaly_score', 0.0)
            unusual_activity = raw_data.get('unusual_activity_detected', False)
            
            # Higher volatility can be good for opportunities
            vol_score = min(volatility * 2, 1.0) if volatility > 0 else 0.5
            
            # Volume anomalies can indicate strong signals
            volume_score = volume_anomaly if volume_anomaly > 0 else 0.5
            
            # Unusual activity can be a strong signal
            activity_score = 1.2 if unusual_activity else 1.0
            
            # Combine scores
            return (vol_score * 0.4 + volume_score * 0.4 + activity_score * 0.2)
            
        except:
            return 1.0
    
    def rank_opportunities(self, opportunities: List[Any]) -> List[Any]:
        """Rank opportunities by priority score"""
        try:
            # Calculate scores for all opportunities
            scored_opportunities = []
            for opp in opportunities:
                opp.priority_score = self.calculate_priority_score(opp)
                scored_opportunities.append(opp)
            
            # Sort by priority score (highest first)
            scored_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            return scored_opportunities
            
        except Exception as e:
            print(f"Error ranking opportunities: {e}")
            return opportunities
    
    def get_top_opportunities(self, opportunities: List[Any], limit: int = 10) -> List[Any]:
        """Get top opportunities by priority score"""
        ranked = self.rank_opportunities(opportunities)
        return ranked[:limit]
    
    def calculate_portfolio_metrics(self, opportunities: List[Any]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics for opportunities"""
        try:
            if not opportunities:
                return {
                    'total_opportunities': 0,
                    'average_score': 0.0,
                    'score_distribution': {},
                    'agent_distribution': {},
                    'type_distribution': {},
                    'expected_return': 0.0,
                    'risk_score': 0.0
                }
            
            # Basic metrics
            total_opps = len(opportunities)
            avg_score = sum(opp.priority_score for opp in opportunities) / total_opps
            
            # Score distribution
            score_ranges = {'High (0.8+)': 0, 'Medium (0.5-0.8)': 0, 'Low (<0.5)': 0}
            for opp in opportunities:
                if opp.priority_score >= 0.8:
                    score_ranges['High (0.8+)'] += 1
                elif opp.priority_score >= 0.5:
                    score_ranges['Medium (0.5-0.8)'] += 1
                else:
                    score_ranges['Low (<0.5)'] += 1
            
            # Agent distribution
            agent_dist = {}
            for opp in opportunities:
                agent_dist[opp.agent_type] = agent_dist.get(opp.agent_type, 0) + 1
            
            # Type distribution
            type_dist = {}
            for opp in opportunities:
                type_dist[opp.opportunity_type] = type_dist.get(opp.opportunity_type, 0) + 1
            
            # Expected return (weighted average of upside potential)
            total_weight = sum(opp.priority_score for opp in opportunities)
            if total_weight > 0:
                expected_return = sum(opp.upside_potential * opp.priority_score for opp in opportunities) / total_weight
            else:
                expected_return = 0.0
            
            # Risk score (inverse of average confidence)
            avg_confidence = sum(opp.confidence for opp in opportunities) / total_opps
            risk_score = 1.0 - avg_confidence
            
            return {
                'total_opportunities': total_opps,
                'average_score': avg_score,
                'score_distribution': score_ranges,
                'agent_distribution': agent_dist,
                'type_distribution': type_dist,
                'expected_return': expected_return,
                'risk_score': risk_score
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {
                'total_opportunities': 0,
                'average_score': 0.0,
                'score_distribution': {},
                'agent_distribution': {},
                'type_distribution': {},
                'expected_return': 0.0,
                'risk_score': 0.0
            }

# Global scorer instance
unified_scorer = UnifiedOpportunityScorer()
