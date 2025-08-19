"""
Simplified Agent Tests - Focus on public APIs and basic functionality
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json

# Import agents
from agents.undervalued.agent import UndervaluedAgent
from agents.technical.agent import TechnicalAgent
from agents.sentiment.agent import SentimentAgent
from agents.flow.agent import FlowAgent
from agents.macro.agent import MacroAgent
from agents.moneyflows.agent import MoneyFlowsAgent
from agents.insider.agent import InsiderAgent

# Import models
from common.opportunity_store import Opportunity, OpportunityStore
from common.unified_opportunity_scorer import UnifiedOpportunityScorer


class TestUndervaluedAgent:
    """Test Undervalued Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create undervalued agent instance"""
        return UndervaluedAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic processing"""
        result = await agent.process(universe=['AAPL', 'TSLA'])
        
        assert isinstance(result, dict)
        assert 'undervalued_analysis' in result
        
        analysis = result['undervalued_analysis']
        assert 'analysis_universe' in analysis
        assert isinstance(analysis['analysis_universe'], list)
    
    @pytest.mark.asyncio
    async def test_process_empty_universe(self, agent):
        """Test processing with empty universe"""
        result = await agent.process(universe=[])
        
        assert isinstance(result, dict)
        assert 'undervalued_analysis' in result
        
        analysis = result['undervalued_analysis']
        assert 'analysis_universe' in analysis
        # The agent may use a default universe when empty is provided
        assert isinstance(analysis['analysis_universe'], list)
    
    @pytest.mark.asyncio
    async def test_process_single_stock(self, agent):
        """Test processing with single stock"""
        result = await agent.process(universe=['AAPL'])
        
        assert isinstance(result, dict)
        assert 'undervalued_analysis' in result
        
        analysis = result['undervalued_analysis']
        assert 'analysis_universe' in analysis
        assert 'AAPL' in analysis['analysis_universe']


class TestTechnicalAgent:
    """Test Technical Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create technical agent instance"""
        return TechnicalAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'find_opportunities')
    
    @pytest.mark.asyncio
    async def test_find_opportunities_basic(self, agent):
        """Test basic opportunity finding"""
        payload = {
            'symbols': ['AAPL'],
            'timeframes': ['1h'],
            'strategies': ['imbalance']
        }
        
        result = await agent.find_opportunities(payload)
        
        assert isinstance(result, dict)
        assert 'opportunities' in result
        assert 'metadata' in result
        
        opportunities = result['opportunities']
        assert isinstance(opportunities, list)
        # May be empty depending on market conditions
    
    @pytest.mark.asyncio
    async def test_find_opportunities_empty_symbols(self, agent):
        """Test with empty symbols"""
        payload = {
            'symbols': [],
            'timeframes': ['1h'],
            'strategies': ['imbalance']
        }
        
        result = await agent.find_opportunities(payload)
        
        assert isinstance(result, dict)
        assert 'opportunities' in result
        assert 'metadata' in result
        # Should handle gracefully even with empty symbols


class TestSentimentAgent:
    """Test Sentiment Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create sentiment agent instance"""
        return SentimentAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic sentiment processing"""
        result = await agent.process(tickers=['AAPL', 'TSLA'], window='1d')
        
        assert isinstance(result, dict)
        assert 'sentiment_analysis' in result
        
        analysis = result['sentiment_analysis']
        assert 'sentiment_data' in analysis
        assert 'analysis_summary' in analysis
    
    @pytest.mark.asyncio
    async def test_process_empty_tickers(self, agent):
        """Test with empty tickers"""
        result = await agent.process(tickers=[], window='1d')
        
        assert isinstance(result, dict)
        assert 'sentiment_analysis' in result
        
        analysis = result['sentiment_analysis']
        assert 'sentiment_data' in analysis
        assert isinstance(analysis['sentiment_data'], list)


class TestFlowAgent:
    """Test Flow Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create flow agent instance"""
        return FlowAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic flow processing"""
        result = await agent.process(tickers=['AAPL', 'TSLA'])
        
        assert isinstance(result, dict)
        assert 'flow_analyses' in result
        
        analyses = result['flow_analyses']
        assert isinstance(analyses, list)
        # May be empty depending on data availability


class TestMacroAgent:
    """Test Macro Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create macro agent instance"""
        return MacroAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic macro processing"""
        result = await agent.process()
        
        assert isinstance(result, dict)
        assert 'macro_analysis' in result
        
        analysis = result['macro_analysis']
        # Check for any of the expected keys that might be present
        expected_keys = ['economic_calendar', 'active_events', 'recent_indicators', 'upcoming_indicators']
        assert any(key in analysis for key in expected_keys)


class TestMoneyFlowsAgent:
    """Test Money Flows Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create money flows agent instance"""
        return MoneyFlowsAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic money flows processing"""
        result = await agent.process(tickers=['AAPL', 'TSLA'])
        
        assert isinstance(result, dict)
        assert 'money_flow_analyses' in result
        
        analyses = result['money_flow_analyses']
        assert isinstance(analyses, list)
        # May be empty depending on data availability


class TestInsiderAgent:
    """Test Insider Agent - Simplified"""
    
    @pytest.fixture
    def agent(self):
        """Create insider agent instance"""
        return InsiderAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent):
        """Test basic insider processing"""
        result = await agent.process(tickers=['AAPL', 'TSLA'])
        
        assert isinstance(result, dict)
        assert 'insider_analyses' in result
        
        analyses = result['insider_analyses']
        assert isinstance(analyses, list)
        # May be empty depending on data availability


class TestAgentIntegration:
    """Test Agent Integration - Simplified"""
    
    @pytest.mark.asyncio
    async def test_agent_result_consistency(self):
        """Test that all agents return consistent result structures"""
        agents = [
            (UndervaluedAgent(), 'universe', ['BRK.B']),
            (TechnicalAgent(), 'symbols', ['AAPL']),
            (SentimentAgent(), 'tickers', ['AAPL']),
            (FlowAgent(), 'symbols', ['AAPL']),
            (MacroAgent(), 'assets', ['SPY']),
            (MoneyFlowsAgent(), 'tickers', ['AAPL']),
            (InsiderAgent(), 'tickers', ['AAPL'])
        ]
        
        for agent, param_name, param_value in agents:
            # Test that each agent can be initialized
            assert agent is not None
            
            # Test that each agent has a process method
            assert hasattr(agent, 'process') or hasattr(agent, 'find_opportunities')
            
            # Test basic functionality (skip if method doesn't exist)
            if hasattr(agent, 'process'):
                if param_name == 'universe':
                    result = await agent.process(universe=param_value)
                elif param_name == 'tickers':
                    if isinstance(agent, SentimentAgent):
                        result = await agent.process(tickers=param_value, window='1d')
                    else:
                        result = await agent.process(tickers=param_value)
                elif param_name == 'symbols':
                    result = await agent.process(tickers=param_value)
                elif param_name == 'assets':
                    result = await agent.process()
                
                assert isinstance(result, dict)
                assert len(result) > 0
            elif hasattr(agent, 'find_opportunities'):
                payload = {
                    'symbols': param_value,
                    'timeframes': ['1h'],
                    'strategies': ['imbalance']
                }
                result = await agent.find_opportunities(payload)
                assert isinstance(result, dict)
                assert 'opportunities' in result
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test that agents handle errors gracefully"""
        agents = [
            UndervaluedAgent(),
            TechnicalAgent(),
            SentimentAgent(),
            FlowAgent(),
            MacroAgent(),
            MoneyFlowsAgent(),
            InsiderAgent()
        ]
        
        for agent in agents:
            # Test with invalid inputs
            try:
                if hasattr(agent, 'process'):
                    if isinstance(agent, SentimentAgent):
                        result = await agent.process(tickers=None, window='1d')
                    elif isinstance(agent, UndervaluedAgent):
                        result = await agent.process(universe=None)
                    elif isinstance(agent, MacroAgent):
                        result = await agent.process()
                    else:
                        result = await agent.process(tickers=None)
                elif hasattr(agent, 'find_opportunities'):
                    result = await agent.find_opportunities(None)
                
                # Should not raise exception, should return some result
                assert isinstance(result, dict)
            except Exception as e:
                # If exception is raised, it should be handled gracefully
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in ["error", "invalid", "argument", "type", "none"])


class TestAgentPerformance:
    """Test Agent Performance - Simplified"""
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self):
        """Test that agents respond within reasonable time"""
        agents = [
            (UndervaluedAgent(), 'universe', ['AAPL']),
            (TechnicalAgent(), 'symbols', ['AAPL']),
            (SentimentAgent(), 'tickers', ['AAPL']),
            (FlowAgent(), 'symbols', ['AAPL']),
            (MacroAgent(), 'assets', ['SPY']),
            (MoneyFlowsAgent(), 'tickers', ['AAPL']),
            (InsiderAgent(), 'tickers', ['AAPL'])
        ]
        
        for agent, param_name, param_value in agents:
            start_time = datetime.now()
            
            try:
                if hasattr(agent, 'process'):
                    if param_name == 'universe':
                        result = await agent.process(universe=param_value)
                    elif param_name == 'tickers':
                        if isinstance(agent, SentimentAgent):
                            result = await agent.process(tickers=param_value, window='1d')
                        else:
                            result = await agent.process(tickers=param_value)
                    elif param_name == 'symbols':
                        result = await agent.process(tickers=param_value)
                    elif param_name == 'assets':
                        result = await agent.process()
                elif hasattr(agent, 'find_opportunities'):
                    payload = {
                        'symbols': param_value,
                        'timeframes': ['1h'],
                        'strategies': ['imbalance']
                    }
                    result = await agent.find_opportunities(payload)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Should respond within 10 seconds
                assert response_time < 10.0
                assert isinstance(result, dict)
                
            except Exception as e:
                # If there's an error, it should be handled quickly
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                assert response_time < 5.0
