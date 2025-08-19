"""
Unit tests for Value Analysis Agent
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np

from agents.undervalued.agent import UndervaluedAgent


class TestUndervaluedAgent:
    """Test UndervaluedAgent class"""
    
    @pytest.fixture
    def agent(self):
        """Create undervalued agent instance for testing"""
        return UndervaluedAgent()
    
    @pytest.fixture
    def sample_universe(self):
        """Create sample universe for testing"""
        return ['BRK.B', 'JPM', 'XOM', 'AAPL', 'MSFT']
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'process')
    
    @pytest.mark.asyncio
    async def test_process_basic(self, agent, sample_universe):
        """Test basic processing"""
        result = await agent.process(universe=sample_universe)
        
        assert isinstance(result, dict)
        assert 'undervalued_analysis' in result
        assert 'identified_opportunities' in result['undervalued_analysis']
        assert 'analysis_summary' in result['undervalued_analysis']
        assert 'success' in result['undervalued_analysis']
        
        # Should have opportunities
        opportunities = result['undervalued_analysis']['identified_opportunities']
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check opportunity structure
        for opp in opportunities:
            assert 'ticker' in opp
            assert 'current_price' in opp
            assert 'fair_value' in opp
            assert 'margin_of_safety' in opp
            assert 'upside_potential' in opp
            assert 'confidence_level' in opp
            assert 'valuation_method' in opp
            assert 'time_horizon' in opp
    
    @pytest.mark.asyncio
    async def test_process_with_mock_data(self, agent):
        """Test processing with mock financial data"""
        universe = ['BRK.B', 'JPM']
        
        # Mock financial data
        mock_data = {
            'BRK.B': {
                'current_price': 350.0,
                'fair_value': 400.0,
                'pe_ratio': 15.0,
                'pb_ratio': 1.2,
                'debt_to_equity': 0.3,
                'revenue_growth': 0.08,
                'profit_margin': 0.12
            },
            'JPM': {
                'current_price': 150.0,
                'fair_value': 180.0,
                'pe_ratio': 12.0,
                'pb_ratio': 1.5,
                'debt_to_equity': 0.8,
                'revenue_growth': 0.05,
                'profit_margin': 0.25
            }
        }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            result = await agent.process(universe=universe)
            
            assert result['undervalued_analysis']['success'] is True
            opportunities = result['undervalued_analysis']['identified_opportunities']
            
            # Should find opportunities for both stocks
            assert len(opportunities) == 2
            
            # Check BRK.B opportunity
            brk_opp = next(opp for opp in opportunities if opp['ticker'] == 'BRK.B')
            assert brk_opp['current_price'] == 350.0
            assert brk_opp['fair_value'] == 400.0
            assert brk_opp['margin_of_safety'] > 0.0
            assert brk_opp['upside_potential'] > 0.0
    
    @pytest.mark.asyncio
    async def test_process_empty_universe(self, agent):
        """Test processing with empty universe"""
        result = await agent.process(universe=[])
        
        assert result['undervalued_analysis']['success'] is True
        opportunities = result['undervalued_analysis']['identified_opportunities']
        assert len(opportunities) == 0
        assert 'No stocks provided' in result['undervalued_analysis']['analysis_summary']
    
    @pytest.mark.asyncio
    async def test_process_error_handling(self, agent, sample_universe):
        """Test error handling in processing"""
        # Mock _get_financial_data to raise exception
        with patch.object(agent, '_get_financial_data', side_effect=Exception("Data error")):
            result = await agent.process(universe=sample_universe)
            
            assert result['undervalued_analysis']['success'] is False
            assert 'error' in result['undervalued_analysis']
            assert 'Data error' in result['undervalued_analysis']['error']
    
    @pytest.mark.asyncio
    async def test_process_valuation_methods(self, agent):
        """Test different valuation methods"""
        universe = ['AAPL']
        
        # Mock data with different valuation scenarios
        mock_data = {
            'AAPL': {
                'current_price': 150.0,
                'fair_value': 180.0,
                'pe_ratio': 20.0,
                'pb_ratio': 8.0,
                'debt_to_equity': 0.1,
                'revenue_growth': 0.15,
                'profit_margin': 0.25,
                'free_cash_flow': 1000000000,
                'book_value': 20.0
            }
        }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            result = await agent.process(universe=universe)
            
            opportunities = result['undervalued_analysis']['identified_opportunities']
            assert len(opportunities) == 1
            
            opp = opportunities[0]
            assert 'valuation_method' in opp
            assert opp['margin_of_safety'] > 0.0
            assert opp['upside_potential'] > 0.0
    
    @pytest.mark.asyncio
    async def test_process_confidence_levels(self, agent):
        """Test confidence level calculation"""
        universe = ['BRK.B']
        
        # Mock data with high-quality metrics
        mock_data = {
            'BRK.B': {
                'current_price': 350.0,
                'fair_value': 400.0,
                'pe_ratio': 15.0,
                'pb_ratio': 1.2,
                'debt_to_equity': 0.3,
                'revenue_growth': 0.08,
                'profit_margin': 0.12,
                'free_cash_flow': 50000000000,
                'book_value': 300.0
            }
        }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            result = await agent.process(universe=universe)
            
            opportunities = result['undervalued_analysis']['identified_opportunities']
            assert len(opportunities) == 1
            
            opp = opportunities[0]
            assert 0.0 <= opp['confidence_level'] <= 1.0
            assert opp['confidence_level'] > 0.5  # Should have reasonable confidence
    
    @pytest.mark.asyncio
    async def test_process_time_horizon(self, agent):
        """Test time horizon assignment"""
        universe = ['JPM']
        
        mock_data = {
            'JPM': {
                'current_price': 150.0,
                'fair_value': 180.0,
                'pe_ratio': 12.0,
                'pb_ratio': 1.5,
                'debt_to_equity': 0.8,
                'revenue_growth': 0.05,
                'profit_margin': 0.25
            }
        }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            result = await agent.process(universe=universe)
            
            opportunities = result['undervalued_analysis']['identified_opportunities']
            assert len(opportunities) == 1
            
            opp = opportunities[0]
            assert 'time_horizon' in opp
            assert isinstance(opp['time_horizon'], str)
            assert len(opp['time_horizon']) > 0
    
    @pytest.mark.asyncio
    async def test_process_analysis_summary(self, agent):
        """Test analysis summary generation"""
        universe = ['BRK.B', 'JPM', 'XOM']
        
        mock_data = {
            'BRK.B': {
                'current_price': 350.0,
                'fair_value': 400.0,
                'pe_ratio': 15.0,
                'pb_ratio': 1.2,
                'debt_to_equity': 0.3,
                'revenue_growth': 0.08,
                'profit_margin': 0.12
            },
            'JPM': {
                'current_price': 150.0,
                'fair_value': 180.0,
                'pe_ratio': 12.0,
                'pb_ratio': 1.5,
                'debt_to_equity': 0.8,
                'revenue_growth': 0.05,
                'profit_margin': 0.25
            },
            'XOM': {
                'current_price': 100.0,
                'fair_value': 90.0,  # Overvalued
                'pe_ratio': 25.0,
                'pb_ratio': 2.0,
                'debt_to_equity': 0.5,
                'revenue_growth': 0.02,
                'profit_margin': 0.08
            }
        }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            result = await agent.process(universe=universe)
            
            summary = result['undervalued_analysis']['analysis_summary']
            
            # Check summary contains expected information
            assert '3 stocks analyzed' in summary
            assert '2 undervalued opportunities' in summary
            assert 'BRK.B' in summary
            assert 'JPM' in summary
            assert 'XOM' not in summary  # Should not be in opportunities
    
    @pytest.mark.asyncio
    async def test_process_performance(self, agent):
        """Test performance with large universe"""
        # Create large universe
        universe = [f'STOCK{i}' for i in range(20)]  # 20 stocks
        
        # Mock data for all stocks
        mock_data = {}
        for ticker in universe:
            mock_data[ticker] = {
                'current_price': 100.0,
                'fair_value': 120.0,
                'pe_ratio': 15.0,
                'pb_ratio': 1.5,
                'debt_to_equity': 0.5,
                'revenue_growth': 0.05,
                'profit_margin': 0.15
            }
        
        with patch.object(agent, '_get_financial_data', return_value=mock_data):
            import time
            start_time = time.time()
            result = await agent.process(universe=universe)
            end_time = time.time()
            
            # Should complete within reasonable time (less than 10 seconds)
            assert end_time - start_time < 10.0
            assert result['undervalued_analysis']['success'] is True
            assert len(result['undervalued_analysis']['identified_opportunities']) == 20
    
    def test_get_financial_data_mock(self, agent):
        """Test financial data retrieval (mocked)"""
        universe = ['BRK.B', 'JPM']
        
        # This should work even without real data
        data = agent._get_financial_data(universe)
        
        assert isinstance(data, dict)
        assert len(data) == 2
        assert 'BRK.B' in data
        assert 'JPM' in data
        
        # Check data structure
        for ticker, ticker_data in data.items():
            assert isinstance(ticker_data, dict)
            assert 'current_price' in ticker_data
            assert 'fair_value' in ticker_data
            assert 'pe_ratio' in ticker_data
            assert 'pb_ratio' in ticker_data
            assert 'debt_to_equity' in ticker_data
            assert 'revenue_growth' in ticker_data
            assert 'profit_margin' in ticker_data
    
    def test_calculate_fair_value(self, agent):
        """Test fair value calculation"""
        # Test with sample financial data
        financial_data = {
            'current_price': 100.0,
            'pe_ratio': 15.0,
            'pb_ratio': 1.5,
            'debt_to_equity': 0.5,
            'revenue_growth': 0.05,
            'profit_margin': 0.15,
            'free_cash_flow': 1000000,
            'book_value': 50.0
        }
        
        fair_value = agent._calculate_fair_value(financial_data)
        
        assert isinstance(fair_value, float)
        assert fair_value > 0.0
        assert fair_value != financial_data['current_price']  # Should be different
    
    def test_calculate_margin_of_safety(self, agent):
        """Test margin of safety calculation"""
        current_price = 100.0
        fair_value = 120.0
        
        margin = agent._calculate_margin_of_safety(current_price, fair_value)
        
        assert isinstance(margin, float)
        assert margin > 0.0
        assert margin <= 1.0  # Should be percentage between 0 and 1
        
        # Test with overvalued stock
        overvalued_margin = agent._calculate_margin_of_safety(120.0, 100.0)
        assert overvalued_margin == 0.0  # No margin of safety for overvalued stock
    
    def test_calculate_upside_potential(self, agent):
        """Test upside potential calculation"""
        current_price = 100.0
        fair_value = 120.0
        
        upside = agent._calculate_upside_potential(current_price, fair_value)
        
        assert isinstance(upside, float)
        assert upside > 0.0
        assert upside == 0.2  # 20% upside
        
        # Test with overvalued stock
        overvalued_upside = agent._calculate_upside_potential(120.0, 100.0)
        assert overvalued_upside == 0.0  # No upside for overvalued stock
    
    def test_calculate_confidence_level(self, agent):
        """Test confidence level calculation"""
        financial_data = {
            'pe_ratio': 15.0,
            'pb_ratio': 1.5,
            'debt_to_equity': 0.5,
            'revenue_growth': 0.05,
            'profit_margin': 0.15
        }
        
        confidence = agent._calculate_confidence_level(financial_data)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test with high-quality metrics
        high_quality_data = {
            'pe_ratio': 10.0,
            'pb_ratio': 1.0,
            'debt_to_equity': 0.2,
            'revenue_growth': 0.10,
            'profit_margin': 0.25
        }
        
        high_confidence = agent._calculate_confidence_level(high_quality_data)
        assert high_confidence > confidence  # Should have higher confidence
    
    def test_determine_valuation_method(self, agent):
        """Test valuation method determination"""
        financial_data = {
            'pe_ratio': 15.0,
            'pb_ratio': 1.5,
            'debt_to_equity': 0.5,
            'revenue_growth': 0.05,
            'profit_margin': 0.15,
            'free_cash_flow': 1000000,
            'book_value': 50.0
        }
        
        method = agent._determine_valuation_method(financial_data)
        
        assert isinstance(method, str)
        assert len(method) > 0
        assert method in ['DCF', 'PE Ratio', 'PB Ratio', 'Asset-Based', 'Hybrid']
    
    def test_determine_time_horizon(self, agent):
        """Test time horizon determination"""
        financial_data = {
            'pe_ratio': 15.0,
            'pb_ratio': 1.5,
            'debt_to_equity': 0.5,
            'revenue_growth': 0.05,
            'profit_margin': 0.15
        }
        
        horizon = agent._determine_time_horizon(financial_data)
        
        assert isinstance(horizon, str)
        assert len(horizon) > 0
        assert horizon in ['6-12 months', '12-18 months', '18+ months']
