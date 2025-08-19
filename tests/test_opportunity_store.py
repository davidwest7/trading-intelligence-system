"""
Unit tests for Opportunity Store
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from common.opportunity_store import OpportunityStore, Opportunity


class TestOpportunity:
    """Test Opportunity dataclass"""
    
    def test_opportunity_creation(self):
        """Test creating an opportunity"""
        opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test opportunity",
            upside_potential=0.15,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={"test": "data"}
        )
        
        assert opp.id == "test_001"
        assert opp.ticker == "AAPL"
        assert opp.agent_type == "value_analysis"
        assert opp.opportunity_type == "Value"
        assert opp.upside_potential == 0.15
        assert opp.confidence == 0.8
        assert opp.time_horizon == "1-3 months"
        assert opp.job_id == "test_job"
        assert opp.raw_data == {"test": "data"}
        assert opp.priority_score == 0.0
        assert opp.status == "active"
    
    def test_opportunity_to_dict(self):
        """Test opportunity to_dict method"""
        now = datetime.now()
        opp = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test opportunity",
            upside_potential=0.15,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=now,
            job_id="test_job",
            raw_data={"test": "data"}
        )
        
        opp_dict = opp.to_dict()
        
        assert opp_dict['id'] == "test_001"
        assert opp_dict['ticker'] == "AAPL"
        assert opp_dict['discovered_at'] == now.isoformat()
        assert opp_dict['raw_data'] == {"test": "data"}


class TestOpportunityStore:
    """Test OpportunityStore class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def store(self, temp_db):
        """Create opportunity store with temp database"""
        return OpportunityStore(db_path=temp_db)
    
    @pytest.fixture
    def sample_opportunity(self):
        """Create sample opportunity for testing"""
        return Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test opportunity",
            upside_potential=0.15,
            confidence=0.8,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={"test": "data"}
        )
    
    def test_store_initialization(self, temp_db):
        """Test store initialization"""
        store = OpportunityStore(db_path=temp_db)
        assert store.db_path == temp_db
        assert store.lock is not None
    
    def test_add_opportunity(self, store, sample_opportunity):
        """Test adding an opportunity"""
        success = store.add_opportunity(sample_opportunity)
        assert success is True
        
        # Verify it was added
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].ticker == "AAPL"
    
    def test_add_opportunity_duplicate(self, store, sample_opportunity):
        """Test adding duplicate opportunity (should replace)"""
        # Add first time
        success1 = store.add_opportunity(sample_opportunity)
        assert success1 is True
        
        # Modify and add again
        sample_opportunity.confidence = 0.9
        success2 = store.add_opportunity(sample_opportunity)
        assert success2 is True
        
        # Should have only one opportunity with updated confidence
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].confidence == 0.9
    
    def test_add_opportunities_from_agent(self, store):
        """Test adding multiple opportunities from agent"""
        opportunities_data = [
            {
                'ticker': 'AAPL',
                'type': 'Value',
                'entry_reason': 'Test 1',
                'upside_potential': 0.15,
                'confidence': 0.8,
                'time_horizon': '1-3 months'
            },
            {
                'ticker': 'TSLA',
                'type': 'Technical',
                'entry_reason': 'Test 2',
                'upside_potential': 0.25,
                'confidence': 0.7,
                'time_horizon': '1-2 weeks'
            }
        ]
        
        added_count = store.add_opportunities_from_agent(
            'value_analysis', 'test_job', opportunities_data
        )
        
        assert added_count == 2
        
        # Verify opportunities were added
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 2
        tickers = [opp.ticker for opp in opportunities]
        assert 'AAPL' in tickers
        assert 'TSLA' in tickers
    
    def test_get_all_opportunities(self, store, sample_opportunity):
        """Test getting all opportunities"""
        # Add opportunity
        store.add_opportunity(sample_opportunity)
        
        # Get all opportunities
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].ticker == "AAPL"
    
    def test_get_all_opportunities_with_status_filter(self, store, sample_opportunity):
        """Test getting opportunities with status filter"""
        # Add active opportunity
        store.add_opportunity(sample_opportunity)
        
        # Add expired opportunity
        expired_opp = Opportunity(
            id="test_002",
            ticker="TSLA",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Expired opportunity",
            upside_potential=0.10,
            confidence=0.6,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            status="expired"
        )
        store.add_opportunity(expired_opp)
        
        # Get only active opportunities
        active_opportunities = store.get_all_opportunities(status="active")
        assert len(active_opportunities) == 1
        assert active_opportunities[0].ticker == "AAPL"
        
        # Get only expired opportunities
        expired_opportunities = store.get_all_opportunities(status="expired")
        assert len(expired_opportunities) == 1
        assert expired_opportunities[0].ticker == "TSLA"
    
    def test_get_top_opportunities(self, store):
        """Test getting top opportunities"""
        # Add multiple opportunities with different scores
        opp1 = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 1",
            upside_potential=0.15,
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
            upside_potential=0.25,
            confidence=0.7,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.7
        )
        
        opp3 = Opportunity(
            id="test_003",
            ticker="MSFT",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 3",
            upside_potential=0.20,
            confidence=0.6,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.8
        )
        
        store.add_opportunity(opp1)
        store.add_opportunity(opp2)
        store.add_opportunity(opp3)
        
        # Get top 2 opportunities
        top_opportunities = store.get_top_opportunities(limit=2)
        assert len(top_opportunities) == 2
        
        # Should be ordered by priority score (highest first)
        assert top_opportunities[0].priority_score == 0.9
        assert top_opportunities[0].ticker == "AAPL"
        assert top_opportunities[1].priority_score == 0.8
        assert top_opportunities[1].ticker == "MSFT"
    
    def test_get_opportunities_by_agent(self, store):
        """Test getting opportunities by agent type"""
        # Add opportunities from different agents
        opp1 = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 1",
            upside_potential=0.15,
            confidence=0.8,
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
            entry_reason="Test 2",
            upside_potential=0.25,
            confidence=0.7,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={}
        )
        
        store.add_opportunity(opp1)
        store.add_opportunity(opp2)
        
        # Get value analysis opportunities
        value_opportunities = store.get_opportunities_by_agent("value_analysis")
        assert len(value_opportunities) == 1
        assert value_opportunities[0].ticker == "AAPL"
        
        # Get technical analysis opportunities
        tech_opportunities = store.get_opportunities_by_agent("technical_analysis")
        assert len(tech_opportunities) == 1
        assert tech_opportunities[0].ticker == "TSLA"
    
    def test_update_priority_scores(self, store, sample_opportunity):
        """Test updating priority scores"""
        # Add opportunity
        store.add_opportunity(sample_opportunity)
        
        # Mock scoring function
        def mock_scoring_function(opp):
            return 0.85
        
        # Update priority scores
        store.update_priority_scores(mock_scoring_function)
        
        # Verify scores were updated
        opportunities = store.get_all_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].priority_score == 0.85
    
    def test_get_statistics(self, store):
        """Test getting statistics"""
        # Add opportunities from different agents
        opp1 = Opportunity(
            id="test_001",
            ticker="AAPL",
            agent_type="value_analysis",
            opportunity_type="Value",
            entry_reason="Test 1",
            upside_potential=0.15,
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
            upside_potential=0.25,
            confidence=0.7,
            time_horizon="1-3 months",
            discovered_at=datetime.now(),
            job_id="test_job",
            raw_data={},
            priority_score=0.7
        )
        
        store.add_opportunity(opp1)
        store.add_opportunity(opp2)
        
        # Get statistics
        stats = store.get_statistics()
        
        assert stats['total_opportunities'] == 2
        assert stats['by_agent_type']['value_analysis'] == 2
        assert stats['average_priority_score'] == 0.8  # (0.9 + 0.7) / 2
    
    def test_error_handling(self, store):
        """Test error handling in store operations"""
        # Test adding opportunity with database error
        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            opp = Opportunity(
                id="test_001",
                ticker="AAPL",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason="Test",
                upside_potential=0.15,
                confidence=0.8,
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={}
            )
            
            success = store.add_opportunity(opp)
            assert success is False
