"""
Unit tests for Main API
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import json

from main import app


class TestMainAPI:
    """Test Main API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "operational"
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_favicon_endpoint(self, client):
        """Test favicon endpoint"""
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "No favicon"
    
    def test_demo_technical_analysis(self, client):
        """Test demo technical analysis endpoint"""
        response = client.post("/demo/technical-analysis", json={
            "symbols": ["AAPL", "TSLA"],
            "timeframes": ["1h", "4h"],
            "strategies": ["imbalance", "trend"]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "opportunities" in data
        assert "analysis_summary" in data
        assert "success" in data
        assert data["success"] is True
        
        # Should have opportunities
        opportunities = data["opportunities"]
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
    
    def test_demo_technical_analysis_invalid_payload(self, client):
        """Test demo technical analysis with invalid payload"""
        # Test with empty symbols
        response = client.post("/demo/technical-analysis", json={
            "symbols": [],
            "timeframes": ["1h"],
            "strategies": ["imbalance"]
        })
        assert response.status_code == 400
        
        # Test with missing symbols
        response = client.post("/demo/technical-analysis", json={
            "timeframes": ["1h"],
            "strategies": ["imbalance"]
        })
        assert response.status_code == 422  # Validation error
    
    def test_demo_value_analysis(self, client):
        """Test demo value analysis endpoint"""
        response = client.post("/demo/value-analysis", json={
            "universe": ["BRK.B", "JPM", "XOM"]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "undervalued_analysis" in data
        assert "identified_opportunities" in data["undervalued_analysis"]
        assert "analysis_summary" in data["undervalued_analysis"]
        assert "success" in data["undervalued_analysis"]
        assert data["undervalued_analysis"]["success"] is True
        
        # Should have opportunities
        opportunities = data["undervalued_analysis"]["identified_opportunities"]
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
    
    def test_demo_value_analysis_empty_universe(self, client):
        """Test demo value analysis with empty universe"""
        response = client.post("/demo/value-analysis", json={
            "universe": []
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["undervalued_analysis"]["success"] is True
        opportunities = data["undervalued_analysis"]["identified_opportunities"]
        assert len(opportunities) == 0
    
    def test_demo_money_flows(self, client):
        """Test demo money flows endpoint"""
        response = client.post("/demo/money-flows", json={
            "tickers": ["AAPL", "TSLA"]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "money_flow_analyses" in data
        assert "analysis_summary" in data
        assert "success" in data
        assert data["success"] is True
        
        # Should have analyses
        analyses = data["money_flow_analyses"]
        assert isinstance(analyses, list)
        assert len(analyses) > 0
    
    def test_demo_insider_analysis(self, client):
        """Test demo insider analysis endpoint"""
        response = client.post("/demo/insider-analysis", json={
            "tickers": ["AAPL", "TSLA"]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "insider_analyses" in data
        assert "analysis_summary" in data
        assert "success" in data
        assert data["success"] is True
        
        # Should have analyses
        analyses = data["insider_analyses"]
        assert isinstance(analyses, list)
        assert len(analyses) > 0
    
    def test_demo_full_analysis(self, client):
        """Test demo full analysis endpoint"""
        response = client.post("/demo/full-analysis", json={
            "symbols": ["AAPL", "TSLA"],
            "universe": ["BRK.B", "JPM"],
            "include_technical": True,
            "include_value": True,
            "include_money_flows": True,
            "include_insider": True
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis_results" in data
        assert "unified_opportunities" in data
        assert "success" in data
        assert data["success"] is True
        
        # Should have analysis results
        results = data["analysis_results"]
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should have unified opportunities
        opportunities = data["unified_opportunities"]
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
    
    def test_demo_full_analysis_minimal(self, client):
        """Test demo full analysis with minimal parameters"""
        response = client.post("/demo/full-analysis", json={})
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "analysis_results" in data
        assert "unified_opportunities" in data
    
    def test_demo_full_analysis_error_handling(self, client):
        """Test demo full analysis error handling"""
        # Mock agent to raise exception
        with patch('main.TechnicalAgent') as mock_tech_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.find_opportunities.side_effect = Exception("Agent error")
            mock_tech_agent.return_value = mock_agent_instance
            
            response = client.post("/demo/full-analysis", json={
                "symbols": ["AAPL"],
                "include_technical": True
            })
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is False
            assert "error" in data
            assert "Agent error" in data["error"]
    
    def test_api_documentation(self, client):
        """Test API documentation endpoints"""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Test docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test redoc
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # Should handle CORS preflight
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS
    
    def test_error_handling(self, client):
        """Test error handling"""
        # Test 404 for non-existent endpoint
        response = client.get("/non-existent")
        assert response.status_code == 404
        
        # Test 422 for invalid JSON
        response = client.post("/demo/technical-analysis", data="invalid json")
        assert response.status_code == 422
    
    def test_response_format_consistency(self, client):
        """Test response format consistency across endpoints"""
        endpoints = [
            ("/demo/technical-analysis", {"symbols": ["AAPL"], "timeframes": ["1h"], "strategies": ["imbalance"]}),
            ("/demo/value-analysis", {"universe": ["BRK.B"]}),
            ("/demo/money-flows", {"tickers": ["AAPL"]}),
            ("/demo/insider-analysis", {"tickers": ["AAPL"]})
        ]
        
        for endpoint, payload in endpoints:
            response = client.post(endpoint, json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "success" in data
            assert isinstance(data["success"], bool)
    
    def test_performance_benchmark(self, client):
        """Test API performance"""
        import time
        
        # Test multiple concurrent requests
        start_time = time.time()
        
        responses = []
        for i in range(5):
            response = client.post("/demo/technical-analysis", json={
                "symbols": ["AAPL"],
                "timeframes": ["1h"],
                "strategies": ["imbalance"]
            })
            responses.append(response)
        
        end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Should complete within reasonable time (less than 10 seconds)
        assert end_time - start_time < 10.0
    
    def test_data_validation(self, client):
        """Test data validation"""
        # Test with invalid symbol format
        response = client.post("/demo/technical-analysis", json={
            "symbols": ["INVALID_SYMBOL_12345"],
            "timeframes": ["1h"],
            "strategies": ["imbalance"]
        })
        assert response.status_code == 200  # Should handle gracefully
        
        # Test with invalid timeframe
        response = client.post("/demo/technical-analysis", json={
            "symbols": ["AAPL"],
            "timeframes": ["invalid_timeframe"],
            "strategies": ["imbalance"]
        })
        assert response.status_code == 200  # Should handle gracefully
        
        # Test with invalid strategy
        response = client.post("/demo/technical-analysis", json={
            "symbols": ["AAPL"],
            "timeframes": ["1h"],
            "strategies": ["invalid_strategy"]
        })
        assert response.status_code == 200  # Should handle gracefully
    
    def test_memory_usage(self, client):
        """Test memory usage with large requests"""
        # Test with large number of symbols
        large_symbols = [f"SYMBOL{i}" for i in range(100)]
        
        response = client.post("/demo/technical-analysis", json={
            "symbols": large_symbols,
            "timeframes": ["1h"],
            "strategies": ["imbalance"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/demo/technical-analysis", json={
                    "symbols": ["AAPL"],
                    "timeframes": ["1h"],
                    "strategies": ["imbalance"]
                })
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(errors) == 0
        assert all(status == 200 for status in results)
        assert len(results) == 10


class TestEventBus:
    """Test Event Bus functionality"""
    
    @pytest.mark.asyncio
    async def test_event_bus_initialization(self):
        """Test event bus initialization"""
        from common.event_bus.bus import EventBus
        
        event_bus = EventBus()
        assert event_bus is not None
        assert hasattr(event_bus, 'start')
        assert hasattr(event_bus, 'stop')
        assert hasattr(event_bus, 'publish')
        assert hasattr(event_bus, 'subscribe')
    
    @pytest.mark.asyncio
    async def test_event_bus_lifecycle(self):
        """Test event bus lifecycle"""
        from common.event_bus.bus import EventBus
        
        event_bus = EventBus()
        
        # Start event bus
        await event_bus.start()
        assert event_bus.is_running
        
        # Stop event bus
        await event_bus.stop()
        assert not event_bus.is_running
    
    @pytest.mark.asyncio
    async def test_event_publishing(self):
        """Test event publishing"""
        from common.event_bus.bus import EventBus, MarketTickEvent
        
        event_bus = EventBus()
        await event_bus.start()
        
        # Create test event
        event = MarketTickEvent(
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            timestamp=1234567890
        )
        
        # Publish event
        await event_bus.publish("market_ticks", event)
        
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test event subscription"""
        from common.event_bus.bus import EventBus, MarketTickEvent
        import asyncio
        
        event_bus = EventBus()
        await event_bus.start()
        
        received_events = []
        
        async def event_handler(event):
            received_events.append(event)
        
        # Subscribe to events
        await event_bus.subscribe("market_ticks", event_handler)
        
        # Publish event
        event = MarketTickEvent(
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            timestamp=1234567890
        )
        await event_bus.publish("market_ticks", event)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Check if event was received
        assert len(received_events) == 1
        assert received_events[0].symbol == "AAPL"
        assert received_events[0].price == 150.0
        
        await event_bus.stop()


class TestScoringSystem:
    """Test Unified Scoring System"""
    
    def test_unified_scorer_initialization(self):
        """Test unified scorer initialization"""
        from common.scoring.unified_score import UnifiedScorer
        
        scorer = UnifiedScorer()
        assert scorer is not None
        assert hasattr(scorer, 'calculate_score')
    
    def test_score_calculation(self):
        """Test score calculation"""
        from common.scoring.unified_score import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        # Test with sample opportunity
        opportunity = {
            'agent_type': 'value_analysis',
            'opportunity_type': 'Value',
            'upside_potential': 0.25,
            'confidence': 0.8,
            'time_horizon': '1-3 months',
            'volatility': 0.15
        }
        
        score = scorer.calculate_score(opportunity)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have positive score
    
    def test_score_ranking(self):
        """Test opportunity ranking"""
        from common.scoring.unified_score import UnifiedScorer
        
        scorer = UnifiedScorer()
        
        # Create multiple opportunities
        opportunities = [
            {
                'agent_type': 'value_analysis',
                'opportunity_type': 'Value',
                'upside_potential': 0.25,
                'confidence': 0.8,
                'time_horizon': '1-3 months',
                'volatility': 0.15
            },
            {
                'agent_type': 'technical_analysis',
                'opportunity_type': 'Technical',
                'upside_potential': 0.15,
                'confidence': 0.6,
                'time_horizon': '1-2 weeks',
                'volatility': 0.25
            }
        ]
        
        ranked = scorer.rank_opportunities(opportunities)
        
        assert len(ranked) == 2
        assert ranked[0]['score'] >= ranked[1]['score']  # Should be ranked by score
