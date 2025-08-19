"""
Sample test for Technical Agent
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from agents.technical.agent import TechnicalAgent
from agents.technical.models import AnalysisPayload


class TestTechnicalAgent:
    """Test suite for Technical Strategy Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance"""
        return TechnicalAgent()
    
    @pytest.mark.asyncio
    async def test_find_opportunities_basic(self, agent):
        """Test basic opportunity finding functionality"""
        # Arrange
        payload = {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframes": ["15m", "1h"],
            "strategies": ["imbalance"],
            "min_score": 0.5,
            "max_risk": 0.02,
            "lookback_periods": 100
        }
        
        # Act
        result = await agent.find_opportunities(payload)
        
        # Assert
        assert "opportunities" in result
        assert "metadata" in result
        assert isinstance(result["opportunities"], list)
        assert isinstance(result["metadata"], dict)
        
        # Check metadata
        metadata = result["metadata"]
        assert "analysis_time_ms" in metadata
        assert "symbols_analyzed" in metadata
        assert "opportunities_found" in metadata
        assert metadata["symbols_analyzed"] == 2
    
    @pytest.mark.asyncio
    async def test_find_opportunities_with_filtering(self, agent):
        """Test opportunity filtering by score and risk"""
        # Arrange
        payload = {
            "symbols": ["AAPL"],
            "timeframes": ["1h"],
            "strategies": ["imbalance", "trend"],
            "min_score": 0.8,  # High threshold
            "max_risk": 0.01,  # Low risk tolerance
            "lookback_periods": 50
        }
        
        # Act
        result = await agent.find_opportunities(payload)
        
        # Assert
        opportunities = result["opportunities"]
        for opp in opportunities:
            assert opp["confidence_score"] >= 0.8
            # Check risk calculation
            risk_pct = abs(opp["stop_loss"] - opp["entry_price"]) / opp["entry_price"]
            assert risk_pct <= 0.01
    
    @pytest.mark.asyncio
    async def test_find_opportunities_empty_symbols(self, agent):
        """Test handling of empty symbol list"""
        # Arrange
        payload = {
            "symbols": [],
            "timeframes": ["1h"],
            "strategies": ["imbalance"]
        }
        
        # Act
        result = await agent.find_opportunities(payload)
        
        # Assert
        assert result["opportunities"] == []
        assert result["metadata"]["symbols_analyzed"] == 0
    
    @pytest.mark.asyncio
    async def test_find_opportunities_invalid_strategy(self, agent):
        """Test handling of invalid strategy names"""
        # Arrange
        payload = {
            "symbols": ["EURUSD"],
            "timeframes": ["1h"],
            "strategies": ["invalid_strategy", "imbalance"]
        }
        
        # Act
        result = await agent.find_opportunities(payload)
        
        # Assert - should fallback to default strategies
        assert "opportunities" in result
        assert "metadata" in result
    
    def test_analysis_payload_validation(self):
        """Test AnalysisPayload model validation"""
        # Valid payload
        payload = AnalysisPayload(
            symbols=["EURUSD"],
            timeframes=["1h"],
            strategies=["imbalance"]
        )
        assert payload.symbols == ["EURUSD"]
        assert payload.min_score == 0.6  # Default value
        
        # Test defaults
        payload_minimal = AnalysisPayload(symbols=["AAPL"])
        assert payload_minimal.timeframes == ["15m", "1h", "4h"]
        assert payload_minimal.strategies == ["imbalance", "fvg", "liquidity_sweep", "trend"]


if __name__ == "__main__":
    # Run a simple test
    async def run_test():
        agent = TechnicalAgent()
        payload = {
            "symbols": ["EURUSD"],
            "timeframes": ["1h"],
            "strategies": ["imbalance"],
            "min_score": 0.5
        }
        
        result = await agent.find_opportunities(payload)
        print(f"Found {len(result['opportunities'])} opportunities")
        print(f"Analysis took {result['metadata']['analysis_time_ms']} ms")
        
        if result['opportunities']:
            opp = result['opportunities'][0]
            print(f"Top opportunity: {opp['symbol']} - {opp['strategy']} - Score: {opp['confidence_score']:.3f}")
    
    asyncio.run(run_test())
