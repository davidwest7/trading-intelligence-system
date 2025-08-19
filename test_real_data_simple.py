#!/usr/bin/env python3
"""
Simple Real Data Test
Tests Polygon.io adapter and individual agents
"""
import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

load_dotenv('env_real_keys.env')

async def test_polygon_adapter():
    """Test Polygon.io adapter directly"""
    print("🔍 **TESTING POLYGON.IO ADAPTER**")
    print("-" * 40)
    
    try:
        from common.data_adapters.polygon_adapter import PolygonAdapter
        
        config = {'polygon_api_key': os.getenv('POLYGON_API_KEY')}
        adapter = PolygonAdapter(config)
        
        # Test connection
        is_connected = await adapter.connect()
        print(f"✅ Connection: {is_connected}")
        
        if is_connected:
            # Test real-time quote
            quote = await adapter.get_real_time_quote('AAPL')
            print(f"✅ AAPL Quote: ${quote['price']:.2f} ({quote['change_percent']:.2f}%)")
            
            # Test performance rankings
            rankings = await adapter.get_performance_rankings(limit=5)
            print(f"✅ Top Gainers: {len(rankings)} stocks")
            if rankings:
                print(f"   #1: {rankings[0]['symbol']} +{rankings[0]['change_percent']:.1f}%")
            
            # Test sector performance
            sectors = await adapter.get_sector_performance()
            print(f"✅ Sectors: {len(sectors)} sectors")
            if sectors:
                print(f"   Top: {sectors[0]['sector']} +{sectors[0]['change_percent']:.1f}%")
            
            return True
        else:
            print("❌ Failed to connect to Polygon.io")
            return False
            
    except Exception as e:
        print(f"❌ Polygon adapter test failed: {e}")
        return False

async def test_technical_agent():
    """Test Technical Agent"""
    print("\n🔧 **TESTING TECHNICAL AGENT**")
    print("-" * 40)
    
    try:
        # Import with correct path
        sys.path.append('agents')
        from technical.agent_real_data import RealDataTechnicalAgent
        
        config = {'polygon_api_key': os.getenv('POLYGON_API_KEY')}
        agent = RealDataTechnicalAgent(config)
        
        result = await agent.analyze_technical_indicators(['AAPL', 'TSLA'])
        
        print(f"✅ Technical Agent: Analyzed {result['tickers_analyzed']} tickers")
        print(f"   Overall Sentiment: {result['overall_signals']['overall_sentiment']}")
        
        # Show sample data
        for ticker in ['AAPL', 'TSLA']:
            if ticker in result['technical_analysis']:
                analysis = result['technical_analysis'][ticker]
                print(f"   {ticker}: ${analysis['current_price']:.2f} (RSI: {analysis.get('rsi', 0):.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_top_performers_agent():
    """Test Top Performers Agent"""
    print("\n🏆 **TESTING TOP PERFORMERS AGENT**")
    print("-" * 40)
    
    try:
        sys.path.append('agents')
        from top_performers.agent_real_data import RealDataTopPerformersAgent
        
        config = {'polygon_api_key': os.getenv('POLYGON_API_KEY')}
        agent = RealDataTopPerformersAgent(config)
        
        result = await agent.analyze_top_performers()
        
        print(f"✅ Top Performers Agent: Successfully analyzed")
        print(f"   Top Gainers: {len(result['top_gainers'])} stocks")
        print(f"   Momentum Regime: {result['momentum_analysis']['momentum_regime']}")
        
        if result['top_gainers']:
            top_gainer = result['top_gainers'][0]
            print(f"   #1 Gainer: {top_gainer['symbol']} +{top_gainer['change_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Top Performers Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 **SIMPLE REAL DATA TEST**")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    results = {}
    
    # Test Polygon adapter
    results['polygon'] = await test_polygon_adapter()
    
    # Test Technical Agent
    results['technical'] = await test_technical_agent()
    
    # Test Top Performers Agent
    results['top_performers'] = await test_top_performers_agent()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 **TEST SUMMARY**")
    print("=" * 50)
    
    successful = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"✅ Successful Tests: {successful}/{total}")
    print(f"📈 Success Rate: {successful/total*100:.0f}%")
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test}")
    
    if successful >= 2:
        print(f"\n🎉 **STATUS: GOOD**")
        print("   Real data integration working!")
    else:
        print(f"\n🔴 **STATUS: NEEDS ATTENTION**")
        print("   Some tests failed.")

if __name__ == "__main__":
    asyncio.run(main())
