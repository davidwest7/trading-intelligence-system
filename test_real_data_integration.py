#!/usr/bin/env python3
"""
Comprehensive Test for Real Data Integration
Verifies that all agents are using real APIs instead of mock data
"""

import asyncio
import os
import sys
from typing import Dict, Any
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Import all agents
from agents.technical.agent_complete import TechnicalAgent
from agents.top_performers.agent_complete import TopPerformersAgent
from agents.sentiment.agent_complete import SentimentAgent
from agents.flow.agent_complete import FlowAgent
from agents.macro.agent_complete import MacroAgent
from agents.undervalued.agent_complete import UndervaluedAgent

# Import Polygon adapter for verification
from common.data_adapters.polygon_adapter import PolygonDataAdapter

class RealDataIntegrationTester:
    """Test class to verify real data integration across all agents"""
    
    def __init__(self):
        self.test_config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'news_api_key': os.getenv('NEWS_API_KEY'),
            'fred_api_key': os.getenv('FRED_API_KEY'),
            'symbols': ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        }
        
        self.test_results = {}
    
    async def test_polygon_api_connection(self) -> bool:
        """Test Polygon.io API connection"""
        print("🔍 Testing Polygon.io API connection...")
        
        try:
            polygon_adapter = PolygonDataAdapter(self.test_config)
            is_connected = await polygon_adapter.connect()
            
            if is_connected:
                print("✅ Polygon.io API connection successful")
                
                # Test data retrieval
                try:
                    quote = await polygon_adapter.get_quote('AAPL')
                    if quote and 'price' in quote:
                        print(f"✅ Real AAPL quote retrieved: ${quote['price']}")
                        return True
                    else:
                        print("❌ No price data in quote response")
                        return False
                except Exception as e:
                    print(f"❌ Error retrieving real quote: {e}")
                    return False
            else:
                print("❌ Polygon.io API connection failed")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Polygon.io API: {e}")
            return False
    
    async def test_technical_agent_real_data(self) -> bool:
        """Test Technical Agent real data integration"""
        print("\n🔍 Testing Technical Agent real data integration...")
        
        try:
            agent = TechnicalAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Technical Agent initialization failed")
                return False
            
            print("✅ Technical Agent initialized with real data")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real technical signals")
                
                # Verify signal metadata contains real data indicators
                for signal in signals[:2]:  # Check first 2 signals
                    metadata = signal.metadata
                    if 'entry_price' in metadata and 'stop_loss' in metadata:
                        print(f"✅ Signal contains real price data: ${metadata['entry_price']}")
                    else:
                        print("❌ Signal missing real price data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Technical Agent: {e}")
            return False
    
    async def test_top_performers_agent_real_data(self) -> bool:
        """Test Top Performers Agent real data integration"""
        print("\n🔍 Testing Top Performers Agent real data integration...")
        
        try:
            agent = TopPerformersAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Top Performers Agent initialization failed")
                return False
            
            print("✅ Top Performers Agent initialized with real data")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real top performers signals")
                
                # Verify signal metadata contains real performance data
                for signal in signals[:2]:
                    metadata = signal.metadata
                    if 'return_1m' in metadata and 'sharpe_ratio' in metadata:
                        print(f"✅ Signal contains real performance data: {metadata['return_1m']:.2%} return")
                    else:
                        print("❌ Signal missing real performance data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Top Performers Agent: {e}")
            return False
    
    async def test_sentiment_agent_real_data(self) -> bool:
        """Test Sentiment Agent real data integration"""
        print("\n🔍 Testing Sentiment Agent real data integration...")
        
        try:
            agent = SentimentAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Sentiment Agent initialization failed")
                return False
            
            print("✅ Sentiment Agent initialized with real APIs")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real sentiment signals")
                
                # Verify signal metadata contains real sentiment data
                for signal in signals[:2]:
                    metadata = signal.metadata
                    if 'total_posts' in metadata and 'sentiment_distribution' in metadata:
                        print(f"✅ Signal contains real sentiment data: {metadata['total_posts']} posts analyzed")
                    else:
                        print("❌ Signal missing real sentiment data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Sentiment Agent: {e}")
            return False
    
    async def test_flow_agent_real_data(self) -> bool:
        """Test Flow Agent real data integration"""
        print("\n🔍 Testing Flow Agent real data integration...")
        
        try:
            agent = FlowAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Flow Agent initialization failed")
                return False
            
            print("✅ Flow Agent initialized with real data")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real flow signals")
                
                # Verify signal metadata contains real flow data
                for signal in signals[:2]:
                    metadata = signal.metadata
                    if 'order_imbalance' in metadata and 'breadth_indicators' in metadata:
                        print(f"✅ Signal contains real flow data: {metadata['order_imbalance']:.3f} imbalance")
                    else:
                        print("❌ Signal missing real flow data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Flow Agent: {e}")
            return False
    
    async def test_macro_agent_real_data(self) -> bool:
        """Test Macro Agent real data integration"""
        print("\n🔍 Testing Macro Agent real data integration...")
        
        try:
            agent = MacroAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Macro Agent initialization failed")
                return False
            
            print("✅ Macro Agent initialized with real APIs")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real macro signals")
                
                # Verify signal metadata contains real macro data
                for signal in signals[:2]:
                    metadata = signal.metadata
                    if 'economic_conditions' in metadata and 'upcoming_events' in metadata:
                        print(f"✅ Signal contains real macro data: {len(metadata['upcoming_events'])} events")
                    else:
                        print("❌ Signal missing real macro data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Macro Agent: {e}")
            return False
    
    async def test_undervalued_agent_real_data(self) -> bool:
        """Test Undervalued Agent real data integration"""
        print("\n🔍 Testing Undervalued Agent real data integration...")
        
        try:
            agent = UndervaluedAgent(self.test_config)
            initialized = await agent.initialize()
            
            if not initialized:
                print("❌ Undervalued Agent initialization failed")
                return False
            
            print("✅ Undervalued Agent initialized with real data")
            
            # Test signal generation
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} real undervalued signals")
                
                # Verify signal metadata contains real fundamental data
                for signal in signals[:2]:
                    metadata = signal.metadata
                    if 'financial_metrics' in metadata and 'peer_comparison' in metadata:
                        print(f"✅ Signal contains real fundamental data: P/E {metadata['financial_metrics']['pe_ratio']:.2f}")
                    else:
                        print("❌ Signal missing real fundamental data")
                        return False
                
                await agent.cleanup()
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Undervalued Agent: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive real data integration test"""
        print("🚀 Starting Comprehensive Real Data Integration Test")
        print("=" * 60)
        
        test_results = {}
        
        # Test Polygon API connection first
        polygon_connected = await self.test_polygon_api_connection()
        test_results['polygon_api'] = polygon_connected
        
        if not polygon_connected:
            print("❌ Polygon API connection failed - cannot test market data agents")
            return test_results
        
        # Test all agents
        agents_to_test = [
            ('technical_agent', self.test_technical_agent_real_data),
            ('top_performers_agent', self.test_top_performers_agent_real_data),
            ('sentiment_agent', self.test_sentiment_agent_real_data),
            ('flow_agent', self.test_flow_agent_real_data),
            ('macro_agent', self.test_macro_agent_real_data),
            ('undervalued_agent', self.test_undervalued_agent_real_data)
        ]
        
        for agent_name, test_func in agents_to_test:
            try:
                result = await test_func()
                test_results[agent_name] = result
            except Exception as e:
                print(f"❌ Error testing {agent_name}: {e}")
                test_results[agent_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 REAL DATA INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:25} {status}")
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL AGENTS SUCCESSFULLY USING REAL DATA!")
        else:
            print("⚠️  Some agents still using mock data or failed to connect")
        
        return test_results

async def main():
    """Main test function"""
    tester = RealDataIntegrationTester()
    results = await tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All tests passed - Real data integration successful!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - Check API keys and connections")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
