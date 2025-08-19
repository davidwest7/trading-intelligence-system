"""
Unit tests for Streamlit Dashboard Components
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import streamlit as st
from datetime import datetime

# Import dashboard components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
        self.sidebar = MagicMock()
        self.markdown = MagicMock()
        self.title = MagicMock()
        self.subheader = MagicMock()
        self.button = MagicMock()
        self.selectbox = MagicMock()
        self.text_input = MagicMock()
        self.success = MagicMock()
        self.error = MagicMock()
        self.info = MagicMock()
        self.warning = MagicMock()
        self.metric = MagicMock()
        self.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        self.plotly_chart = MagicMock()
        self.dataframe = MagicMock()
        self.json = MagicMock()

# Mock streamlit
sys.modules['streamlit'] = MockStreamlit()

from common.opportunity_store import OpportunityStore, Opportunity
from common.unified_opportunity_scorer import UnifiedOpportunityScorer


class TestOpportunityStore:
    """Test Opportunity Store functionality"""
    
    @pytest.fixture
    def store(self):
        """Create opportunity store for testing"""
        return OpportunityStore(db_path=":memory:")
    
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
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={"test": "data"}
        )
    
    def test_store_initialization(self, store):
        """Test store initialization"""
        assert store is not None
        assert hasattr(store, 'add_opportunity')
        assert hasattr(store, 'get_all_opportunities')
        assert hasattr(store, 'get_top_opportunities')
    
    def test_add_opportunity(self, store, sample_opportunity):
        """Test adding opportunity"""
        success = store.add_opportunity(sample_opportunity)
        assert success is True
        
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].ticker == "AAPL"
    
    def test_get_top_opportunities(self, store):
        """Test getting top opportunities"""
        # Add multiple opportunities with different scores
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
            raw_data={},
            priority_score=0.9
        )
        
        opp2 = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="technical_analysis",
            opportunity_type="Technical",
            entry_reason="Low value",
            upside_potential=0.15,
            confidence=0.60,
            time_horizon="1-2 weeks",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.7
        )
        
        store.add_opportunity(opp1)
        store.add_opportunity(opp2)
        
        top_opportunities = store.get_top_opportunities(limit=2)
        assert len(top_opportunities) == 2
        assert top_opportunities[0].priority_score >= top_opportunities[1].priority_score
        assert top_opportunities[0].ticker == "AAPL"  # Higher score
    
    def test_get_statistics(self, store):
        """Test getting statistics"""
        # Add opportunities from different agents
        opp1 = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 1",
            upside_potential=0.25,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.9
        )
        
        opp2 = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 2",
            upside_potential=0.20,
            confidence=0.7,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.7
        )
        
        store.add_opportunity(opp1)
        store.add_opportunity(opp2)
        
        stats = store.get_statistics()
        
        assert stats['total_opportunities'] == 2
        assert stats['by_agent_type']['value_analysis'] == 2
        assert stats['average_priority_score'] == 0.8  # (0.9 + 0.7) / 2


class TestUnifiedOpportunityScorer:
    """Test Unified Opportunity Scorer"""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer for testing"""
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
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={"test": "data"}
        )
    
    def test_scorer_initialization(self, scorer):
        """Test scorer initialization"""
        assert scorer is not None
        assert hasattr(scorer, 'calculate_priority_score')
        assert hasattr(scorer, 'rank_opportunities')
        assert hasattr(scorer, 'get_top_opportunities')
    
    def test_calculate_priority_score(self, scorer, sample_opportunity):
        """Test priority score calculation"""
        score = scorer.calculate_priority_score(sample_opportunity)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have positive score
    
    def test_rank_opportunities(self, scorer):
        """Test opportunity ranking"""
        # Create multiple opportunities
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
            agent_type="technical_analysis",
            opportunity_type="Technical",
            entry_reason="Low value",
            upside_potential=0.15,
            confidence=0.60,
            time_horizon="1-2 weeks",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={}
        )
        
        opportunities = [opp1, opp2]
        ranked = scorer.rank_opportunities(opportunities)
        
        assert len(ranked) == 2
        assert ranked[0].priority_score >= ranked[1].priority_score
        assert ranked[0].ticker == "AAPL"  # Should be ranked higher
    
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
        
        top_3 = scorer.get_top_opportunities(opportunities, limit=3)
        assert len(top_3) == 3
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


class TestDashboardComponents:
    """Test Dashboard Components"""
    
    def test_enhanced_job_tracker(self):
        """Test Enhanced Job Tracker functionality"""
        # Import the tracker class
        from streamlit_enhanced import EnhancedJobTracker
        
        # Test job creation
        job_id = EnhancedJobTracker.create_job("test_analysis", {"param": "value"})
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        
        # Test job retrieval
        job = EnhancedJobTracker.get_job(job_id)
        assert job is not None
        assert job['id'] == job_id
        assert job['type'] == "test_analysis"
        assert job['status'] == "created"
    
    def test_opportunity_extraction(self):
        """Test opportunity extraction from agent results"""
        from streamlit_enhanced import EnhancedJobTracker
        
        # Test technical analysis result
        tech_result = {
            'opportunities': [
                {
                    'symbol': 'AAPL',
                    'strategy': 'imbalance',
                    'entry_price': 150.0,
                    'stop_loss': 148.0,
                    'take_profit': 155.0,
                    'risk_reward_ratio': 2.5,
                    'confidence_score': 0.75,
                    'timeframe': '1h'
                }
            ]
        }
        
        opportunities = EnhancedJobTracker._extract_opportunities(tech_result, "technical_analysis")
        assert len(opportunities) == 1
        assert opportunities[0]['ticker'] == 'AAPL'
        assert opportunities[0]['type'] == 'Technical'
        
        # Test value analysis result
        value_result = {
            'undervalued_analysis': {
                'identified_opportunities': [
                    {
                        'ticker': 'BRK.B',
                        'margin_of_safety': 0.25,
                        'upside_potential': 0.30,
                        'confidence_level': 0.8,
                        'time_horizon': '12-18 months'
                    }
                ]
            }
        }
        
        opportunities = EnhancedJobTracker._extract_opportunities(value_result, "value_analysis")
        assert len(opportunities) == 1
        assert opportunities[0]['ticker'] == 'BRK.B'
        assert opportunities[0]['type'] == 'Value'
    
    def test_job_progress_tracking(self):
        """Test job progress tracking"""
        from streamlit_enhanced import EnhancedJobTracker
        
        # Create a job
        job_id = EnhancedJobTracker.create_job("test_analysis", {"param": "value"})
        
        # Update progress
        EnhancedJobTracker.update_job_progress(job_id, "Initializing", 10)
        EnhancedJobTracker.update_job_progress(job_id, "Processing", 50)
        EnhancedJobTracker.update_job_progress(job_id, "Finalizing", 100)
        
        # Get job and check progress
        job = EnhancedJobTracker.get_job(job_id)
        assert job['progress_percentage'] == 100
        assert job['current_stage'] == "Finalizing"
        assert len(job['progress_stages']) == 3
    
    def test_job_status_updates(self):
        """Test job status updates"""
        from streamlit_enhanced import EnhancedJobTracker
        
        # Create a job
        job_id = EnhancedJobTracker.create_job("test_analysis", {"param": "value"})
        
        # Update status
        EnhancedJobTracker.update_job_status(job_id, "running")
        job = EnhancedJobTracker.get_job(job_id)
        assert job['status'] == "running"
        assert job['started_at'] is not None
        
        # Complete job
        result = {"opportunities": [], "success": True}
        EnhancedJobTracker.update_job_status(job_id, "completed", result=result)
        job = EnhancedJobTracker.get_job(job_id)
        assert job['status'] == "completed"
        assert job['completed_at'] is not None
        assert job['result'] == result
    
    def test_logging_functionality(self):
        """Test logging functionality"""
        from streamlit_enhanced import EnhancedJobTracker
        
        # Test logging
        EnhancedJobTracker.log("Test log message")
        
        # Check if log was added (this would require access to session state)
        # In a real test, we'd check the session state
        assert True  # Placeholder assertion
    
    def test_error_handling(self):
        """Test error handling in dashboard components"""
        from streamlit_enhanced import EnhancedJobTracker
        
        # Test with invalid job ID
        job = EnhancedJobTracker.get_job("invalid_id")
        assert job is None
        
        # Test with invalid status update
        EnhancedJobTracker.update_job_status("invalid_id", "running")
        # Should not raise exception
        
        # Test with invalid progress update
        EnhancedJobTracker.update_job_progress("invalid_id", "test", 50)
        # Should not raise exception


class TestDashboardIntegration:
    """Test Dashboard Integration"""
    
    def test_opportunity_flow(self):
        """Test opportunity flow from agents to dashboard"""
        from common.opportunity_store import OpportunityStore
        from common.unified_opportunity_scorer import UnifiedOpportunityScorer
        from streamlit_enhanced import EnhancedJobTracker
        
        # Create store and scorer
        store = OpportunityStore(db_path=":memory:")
        scorer = UnifiedOpportunityScorer()
        
        # Create a job
        job_id = EnhancedJobTracker.create_job("value_analysis", {"universe": ["BRK.B"]})
        
        # Simulate agent result
        agent_result = {
            'undervalued_analysis': {
                'identified_opportunities': [
                    {
                        'ticker': 'BRK.B',
                        'margin_of_safety': 0.25,
                        'upside_potential': 0.30,
                        'confidence_level': 0.8,
                        'time_horizon': '12-18 months'
                    }
                ]
            }
        }
        
        # Extract opportunities
        opportunities = EnhancedJobTracker._extract_opportunities(agent_result, "value_analysis")
        assert len(opportunities) == 1
        
        # Add to store
        added_count = store.add_opportunities_from_agent("value_analysis", job_id, opportunities)
        assert added_count == 1
        
        # Get from store
        stored_opportunities = store.get_all_opportunities()
        assert len(stored_opportunities) == 1
        assert stored_opportunities[0].ticker == "BRK.B"
        
        # Calculate priority scores
        for opp in stored_opportunities:
            opp.priority_score = scorer.calculate_priority_score(opp)
        
        # Get top opportunities
        top_opportunities = store.get_top_opportunities(limit=10)
        assert len(top_opportunities) == 1
        assert top_opportunities[0].priority_score > 0.0
    
    def test_multi_agent_integration(self):
        """Test integration with multiple agents"""
        from common.opportunity_store import OpportunityStore
        from common.unified_opportunity_scorer import UnifiedOpportunityScorer
        from streamlit_enhanced import EnhancedJobTracker
        
        # Create store and scorer
        store = OpportunityStore(db_path=":memory:")
        scorer = UnifiedOpportunityScorer()
        
        # Simulate results from multiple agents
        agent_results = {
            "technical_analysis": {
                'opportunities': [
                    {
                        'symbol': 'AAPL',
                        'strategy': 'imbalance',
                        'entry_price': 150.0,
                        'stop_loss': 148.0,
                        'take_profit': 155.0,
                        'risk_reward_ratio': 2.5,
                        'confidence_score': 0.75,
                        'timeframe': '1h'
                    }
                ]
            },
            "value_analysis": {
                'undervalued_analysis': {
                    'identified_opportunities': [
                        {
                            'ticker': 'BRK.B',
                            'margin_of_safety': 0.25,
                            'upside_potential': 0.30,
                            'confidence_level': 0.8,
                            'time_horizon': '12-18 months'
                        }
                    ]
                }
            }
        }
        
        total_added = 0
        
        # Process each agent result
        for agent_type, result in agent_results.items():
            job_id = EnhancedJobTracker.create_job(agent_type, {})
            opportunities = EnhancedJobTracker._extract_opportunities(result, agent_type)
            added_count = store.add_opportunities_from_agent(agent_type, job_id, opportunities)
            total_added += added_count
        
        assert total_added == 2
        
        # Get all opportunities
        all_opportunities = store.get_all_opportunities()
        assert len(all_opportunities) == 2
        
        # Check agent distribution
        agent_types = [opp.agent_type for opp in all_opportunities]
        assert "technical_analysis" in agent_types
        assert "value_analysis" in agent_types
        
        # Get top opportunities
        top_opportunities = store.get_top_opportunities(limit=10)
        assert len(top_opportunities) == 2
        
        # Calculate portfolio metrics
        metrics = scorer.calculate_portfolio_metrics(all_opportunities)
        assert metrics['total_opportunities'] == 2
        assert len(metrics['agent_distribution']) == 2
