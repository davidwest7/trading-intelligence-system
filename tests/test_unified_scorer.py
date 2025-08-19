"""
Unit tests for Unified Opportunity Scorer
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from common.unified_opportunity_scorer import UnifiedOpportunityScorer
from common.opportunity_store import Opportunity


class TestUnifiedOpportunityScorer:
    """Test UnifiedOpportunityScorer class"""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance for testing"""
        return UnifiedOpportunityScorer()
    
    @pytest.fixture
    def sample_opportunity(self):
        """Create sample opportunity for testing"""
        return Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test opportunity",
            upside_potential=0.25,
            confidence=0.85,
            time_horizon="6-12 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={"test": "data"}
        )
    
    def test_scorer_initialization(self, scorer):
        """Test scorer initialization"""
        assert scorer.agent_weights is not None
        assert scorer.opportunity_type_weights is not None
        assert scorer.time_horizon_weights is not None
        
        # Check that all expected agent weights are present
        expected_agents = [
            'value_analysis', 'technical_analysis', 'money_flows',
            'insider_analysis', 'sentiment_analysis', 'macro_analysis',
            'flow_analysis', 'top_performers_analysis', 'undervalued_analysis',
            'causal_analysis', 'hedging_analysis', 'learning_analysis'
        ]
        
        for agent in expected_agents:
            assert agent in scorer.agent_weights
    
    def test_calculate_priority_score_basic(self, scorer, sample_opportunity):
        """Test basic priority score calculation"""
        score = scorer.calculate_priority_score(sample_opportunity)
        
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
        
        # Should be a reasonable value given the inputs
        assert score > 0.0
    
    def test_calculate_priority_score_high_value(self, scorer):
        """Test priority score for high-value opportunity"""
        high_value_opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",  # High weight (0.20)
            opportunity_type="Value",      # High weight (1.0)
            entry_reason="High value opportunity",
            upside_potential=0.50,         # High upside
            confidence=0.95,               # High confidence
            time_horizon="1-3 months",     # Short horizon (high weight)
            discovered_at=datetime.now(),  # Recent
            job_id="test_job",
            raw_data={}
        )
        
        score = scorer.calculate_priority_score(high_value_opp)
        assert score > 0.3  # Should be reasonably high score
    
    def test_calculate_priority_score_low_value(self, scorer):
        """Test priority score for low-value opportunity"""
        low_value_opp = Opportunity(
            id="test_001",
            ticker="TSLA",
            agent_type="learning_analysis",  # Low weight (0.01)
            opportunity_type="Learning",     # Low weight (0.6)
            entry_reason="Low value opportunity",
            upside_potential=0.05,           # Low upside
            confidence=0.30,                 # Low confidence
            time_horizon="12-18 months",     # Long horizon (low weight)
            discovered_at=datetime.now() - timedelta(days=30),  # Old
            job_id="test_job",
            raw_data={}
        )
        
        score = scorer.calculate_priority_score(low_value_opp)
        assert score < 0.3  # Should be low score
    
    def test_calculate_priority_score_edge_cases(self, scorer):
        """Test priority score with edge cases"""
        # Test with maximum values
        max_opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Maximum opportunity",
            upside_potential=1.0,           # Maximum upside
            confidence=1.0,                 # Maximum confidence
            time_horizon="1-2 weeks",       # Shortest horizon
            discovered_at=datetime.now(),   # Very recent
            job_id="test_job",
            raw_data={}
        )
        
        max_score = scorer.calculate_priority_score(max_opp)
        assert max_score <= 1.0  # Should not exceed 1.0
        
        # Test with minimum values
        min_opp = Opportunity(
            id="test_001",
            ticker="TSLA",
            agent_type="hedging_analysis",  # Lowest weight
            opportunity_type="Hedging",     # Low weight
            entry_reason="Minimum opportunity",
            upside_potential=0.0,           # No upside
            confidence=0.0,                 # No confidence
            time_horizon="18+ months",      # Longest horizon
            discovered_at=datetime.now() - timedelta(days=365),  # Very old
            job_id="test_job",
            raw_data={}
        )
        
        min_score = scorer.calculate_priority_score(min_opp)
        assert min_score >= 0.0  # Should not be negative
    
    def test_calculate_recency_score(self, scorer):
        """Test recency score calculation"""
        now = datetime.now()
        
        # Test very recent (less than 1 hour)
        recent_opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Recent",
            upside_potential=0.25,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=now - timedelta(minutes=30),
            job_id="test_job",
            raw_data={}
        )
        
        recent_score = scorer.calculate_priority_score(recent_opp)
        
        # Test old (more than 1 month)
        old_opp = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Old",
            upside_potential=0.25,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=now - timedelta(days=60),
            job_id="test_job",
            raw_data={}
        )
        
        old_score = scorer.calculate_priority_score(old_opp)
        
        # Recent should have higher score
        assert recent_score > old_score
    
    def test_calculate_volatility_score(self, scorer):
        """Test volatility score calculation"""
        # Test with volatility data
        volatile_opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Volatile",
            upside_potential=0.25,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={
                'volatility': 0.8,
                'volume_anomaly_score': 0.9,
                'unusual_activity_detected': True
            }
        )
        
        volatile_score = scorer.calculate_priority_score(volatile_opp)
        
        # Test without volatility data
        stable_opp = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Stable",
            upside_potential=0.25,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={}
        )
        
        stable_score = scorer.calculate_priority_score(stable_opp)
        
        # Both should be valid scores
        assert 0.0 <= volatile_score <= 1.0
        assert 0.0 <= stable_score <= 1.0
    
    def test_rank_opportunities(self, scorer):
        """Test opportunity ranking"""
        # Create multiple opportunities with different scores
        opp1 = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="High value",
            upside_potential=0.50,
            confidence=0.95,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={}
        )
        
        opp2 = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="learning_analysis",
            opportunity_type="Learning",
            entry_reason="Low value",
            upside_potential=0.05,
            confidence=0.30,
            time_horizon="12-18 months",
            discovered_at=datetime.now() - timedelta(days=30),
            job_id="test_job",
            raw_data={}
        )
        
        opp3 = Opportunity(
            id="test_003",
            ticker="MSFT",
            agent_type="technical_analysis",
            opportunity_type="Technical",
            entry_reason="Medium value",
            upside_potential=0.25,
            confidence=0.70,
            time_horizon="3-6 months",
            discovered_at=datetime.now() - timedelta(days=7),
            job_id="test_job",
            raw_data={}
        )
        
        opportunities = [opp1, opp2, opp3]
        ranked = scorer.rank_opportunities(opportunities)
        
        # Should have same number of opportunities
        assert len(ranked) == 3
        
        # Should be ranked by priority score (highest first)
        assert ranked[0].priority_score >= ranked[1].priority_score
        assert ranked[1].priority_score >= ranked[2].priority_score
        
        # AAPL should be ranked highest (high value)
        assert ranked[0].ticker == "AAPL"
        
        # TSLA should be ranked lowest (low value)
        assert ranked[2].ticker == "TSLA"
    
    def test_get_top_opportunities(self, scorer):
        """Test getting top opportunities"""
        # Create multiple opportunities
        opportunities = []
        for i in range(5):
            opp = Opportunity(
                id=f"test_{i:03d}",
                ticker=f"STOCK{i}",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason=f"Test {i}",
                upside_potential=0.1 + (i * 0.1),
                confidence=0.5 + (i * 0.1),
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            opportunities.append(opp)
        
        # Get top 3 opportunities
        top_3 = scorer.get_top_opportunities(opportunities, limit=3)
        assert len(top_3) == 3
        
        # Should be ranked by priority score
        assert top_3[0].priority_score >= top_3[1].priority_score
        assert top_3[1].priority_score >= top_3[2].priority_score
    
    def test_calculate_portfolio_metrics(self, scorer):
        """Test portfolio metrics calculation"""
        # Create opportunities from different agents
        opportunities = []
        
        # Value analysis opportunities
        for i in range(3):
            opp = Opportunity(
                id=f"value_{i:03d}",
                ticker=f"VALUE{i}",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason=f"Value {i}",
                upside_potential=0.2 + (i * 0.1),
                confidence=0.7 + (i * 0.1),
                time_horizon="6-12 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            opportunities.append(opp)
        
        # Technical analysis opportunities
        for i in range(2):
            opp = Opportunity(
                id=f"tech_{i:03d}",
                ticker=f"TECH{i}",
                agent_type="technical_analysis",
                opportunity_type="Technical",
                entry_reason=f"Technical {i}",
                upside_potential=0.15 + (i * 0.05),
                confidence=0.6 + (i * 0.1),
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            opportunities.append(opp)
        
        # Calculate priority scores for opportunities
        for opp in opportunities:
            opp.priority_score = scorer.calculate_priority_score(opp)
        
        # Calculate metrics
        metrics = scorer.calculate_portfolio_metrics(opportunities)
        
        # Check basic metrics
        assert metrics['total_opportunities'] == 5
        assert metrics['average_score'] > 0.0
        assert metrics['expected_return'] > 0.0
        assert metrics['risk_score'] >= 0.0
        
        # Check distributions
        assert 'value_analysis' in metrics['agent_distribution']
        assert 'technical_analysis' in metrics['agent_distribution']
        assert metrics['agent_distribution']['value_analysis'] == 3
        assert metrics['agent_distribution']['technical_analysis'] == 2
        
        # Check score distribution
        assert 'High (0.8+)' in metrics['score_distribution']
        assert 'Medium (0.5-0.8)' in metrics['score_distribution']
        assert 'Low (<0.5)' in metrics['score_distribution']
    
    def test_calculate_portfolio_metrics_empty(self, scorer):
        """Test portfolio metrics with empty opportunities"""
        metrics = scorer.calculate_portfolio_metrics([])
        
        assert metrics['total_opportunities'] == 0
        assert metrics['average_score'] == 0.0
        assert metrics['expected_return'] == 0.0
        assert metrics['risk_score'] == 0.0
        assert metrics['agent_distribution'] == {}
        assert metrics['type_distribution'] == {}
        assert metrics['score_distribution'] == {}
    
    def test_error_handling(self, scorer):
        """Test error handling in scorer"""
        # Test with invalid opportunity
        with patch.object(scorer, '_calculate_recency_score', side_effect=Exception("Recency error")):
            opp = Opportunity(
                id="test_001",
                ticker="AAPL",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason="Test",
                upside_potential=0.25,
                confidence=0.8,
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            
            score = scorer.calculate_priority_score(opp)
            assert score == 0.0  # Should return 0 on error
        
        # Test with invalid portfolio metrics
        with patch.object(scorer, '_calculate_recency_score', side_effect=Exception("Metrics error")):
            metrics = scorer.calculate_portfolio_metrics([])
            assert metrics['total_opportunities'] == 0  # Should return default values
    
    def test_agent_weight_coverage(self, scorer):
        """Test that all agent types have weights"""
        # Test all agent types
        agent_types = [
            'value_analysis', 'technical_analysis', 'money_flows',
            'insider_analysis', 'sentiment_analysis', 'macro_analysis',
            'flow_analysis', 'top_performers_analysis', 'undervalued_analysis',
            'causal_analysis', 'hedging_analysis', 'learning_analysis'
        ]
        
        for agent_type in agent_types:
            opp = Opportunity(
                id="test_001",
                ticker="AAPL",
                agent_type=agent_type,
                opportunity_type="Value",
                entry_reason="Test",
                upside_potential=0.25,
                confidence=0.8,
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            
            score = scorer.calculate_priority_score(opp)
            assert 0.0 <= score <= 1.0  # Should calculate valid score
    
    def test_opportunity_type_weight_coverage(self, scorer):
        """Test that all opportunity types have weights"""
        # Test all opportunity types
        opportunity_types = [
            'Value', 'Technical', 'Flow', 'Insider', 'Sentiment', 'Macro',
            'Top Performer', 'Undervalued', 'Causal', 'Hedging', 'Learning'
        ]
        
        for opp_type in opportunity_types:
            opp = Opportunity(
                id="test_001",
                ticker="AAPL",
                agent_type="value_analysis",
                opportunity_type=opp_type,
                entry_reason="Test",
                upside_potential=0.25,
                confidence=0.8,
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            
            score = scorer.calculate_priority_score(opp)
            assert 0.0 <= score <= 1.0  # Should calculate valid score
    
    def test_time_horizon_weight_coverage(self, scorer):
        """Test that all time horizons have weights"""
        # Test all time horizons
        time_horizons = [
            '1-2 weeks', '2-4 weeks', '1-3 months', '1-4 weeks',
            '1-6 months', '3-6 months', '6-12 months', '12-18 months', '18+ months'
        ]
        
        for time_horizon in time_horizons:
            opp = Opportunity(
                id="test_001",
                ticker="AAPL",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason="Test",
                upside_potential=0.25,
                confidence=0.8,
                time_horizon=time_horizon,
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            
            score = scorer.calculate_priority_score(opp)
            assert 0.0 <= score <= 1.0  # Should calculate valid score
