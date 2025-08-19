#!/usr/bin/env python3
"""
Polygon.io Integration Test
Tests the Polygon.io API key and demonstrates real data for 6 agents
"""
import os
import asyncio
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

load_dotenv('env_real_keys.env')

async def test_polygon_integration():
    """Test Polygon.io API integration"""
    print("🚀 **POLYGON.IO INTEGRATION TEST**")
    print("=" * 50)
    
    # Test API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ Polygon.io API key not found in env_real_keys.env")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Import the adapter
        from common.data_adapters.polygon_adapter import PolygonAdapter
        
        # Initialize adapter
        config = {'polygon_api_key': api_key}
        adapter = PolygonAdapter(config)
        
        # Test connection
        print("\n🔗 Testing connection...")
        is_connected = await adapter.connect()
        if not is_connected:
            print("❌ Failed to connect to Polygon.io")
            return False
        print("✅ Successfully connected to Polygon.io")
        
        # Test symbols
        test_symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ']
        
        print(f"\n📊 **TESTING 6 AGENTS WITH REAL DATA**")
        print("=" * 50)
        
        # ==================== TECHNICAL AGENT TESTS ====================
        print("\n🔧 **1. TECHNICAL AGENT - Real-time Data**")
        print("-" * 30)
        
        for symbol in test_symbols[:2]:
            print(f"\n📈 Testing {symbol}:")
            
            # Real-time quote
            quote = await adapter.get_real_time_quote(symbol)
            print(f"   Price: ${quote['price']:.2f}")
            print(f"   Change: {quote['change_percent']:.2f}%")
            print(f"   Volume: {quote['volume']:,}")
            
            # Intraday data
            intraday = await adapter.get_intraday_data(symbol, interval="5", limit=10)
            if not intraday.empty:
                print(f"   Intraday bars: {len(intraday)}")
                latest = intraday.iloc[-1]
                print(f"   Latest: ${latest['close']:.2f} (Vol: {latest['volume']:,})")
            
            # Options data
            options = await adapter.get_options_data(symbol)
            print(f"   Options contracts: {options['options_count']}")
        
        # ==================== FLOW AGENT TESTS ====================
        print("\n🌊 **2. FLOW AGENT - Level 2 Data**")
        print("-" * 30)
        
        for symbol in test_symbols[:2]:
            print(f"\n📊 Testing {symbol}:")
            
            # Level 2 data
            level2 = await adapter.get_level2_data(symbol)
            print(f"   Bid: ${level2['bid']:.2f} (Size: {level2['bid_size']:,})")
            print(f"   Ask: ${level2['ask']:.2f} (Size: {level2['ask_size']:,})")
            
            # Unusual options activity
            unusual = await adapter.get_unusual_options_activity(symbol)
            if unusual:
                print(f"   Unusual options: {len(unusual)} contracts")
        
        # ==================== MONEY FLOWS AGENT TESTS ====================
        print("\n💰 **3. MONEY FLOWS AGENT - Institutional Flow**")
        print("-" * 30)
        
        for symbol in test_symbols[:2]:
            print(f"\n🏦 Testing {symbol}:")
            
            # Institutional flow
            flow = await adapter.get_institutional_flow(symbol)
            print(f"   Total Volume: {flow['total_volume']:,}")
            print(f"   Large Trades: {flow['large_trades_count']}")
            print(f"   Avg Trade Size: {flow['avg_trade_size']:.0f}")
            print(f"   Institutional Score: {flow['institutional_flow_score']:.2%}")
        
        # ==================== TOP PERFORMERS AGENT TESTS ====================
        print("\n🏆 **4. TOP PERFORMERS AGENT - Performance Rankings**")
        print("-" * 30)
        
        # Performance rankings
        rankings = await adapter.get_performance_rankings(limit=10)
        print(f"\n📈 Top 10 Gainers:")
        for rank in rankings[:5]:
            print(f"   {rank['rank']}. {rank['symbol']}: {rank['change_percent']:.2f}% (${rank['price']:.2f})")
        
        # Sector performance
        sectors = await adapter.get_sector_performance()
        print(f"\n🏭 Top Sectors:")
        for sector in sectors[:5]:
            print(f"   {sector['sector']}: {sector['change_percent']:.2f}%")
        
        # ==================== UNDERVALUED AGENT TESTS ====================
        print("\n💎 **5. UNDERVALUED AGENT - Financial Data**")
        print("-" * 30)
        
        for symbol in test_symbols[:2]:
            print(f"\n📋 Testing {symbol}:")
            
            # Financial statements
            financials = await adapter.get_financial_statements(symbol)
            print(f"   Revenue: ${financials['revenue']:,.0f}")
            print(f"   Net Income: ${financials['net_income']:,.0f}")
            print(f"   Total Assets: ${financials['total_assets']:,.0f}")
            
            # Valuation metrics
            valuation = await adapter.get_valuation_metrics(symbol)
            print(f"   P/E Ratio: {valuation['pe_ratio']:.2f}")
            print(f"   P/B Ratio: {valuation['pb_ratio']:.2f}")
            print(f"   Market Cap: ${valuation['market_cap']:,.0f}")
        
        # ==================== MACRO AGENT TESTS ====================
        print("\n🌍 **6. MACRO AGENT - Economic Indicators**")
        print("-" * 30)
        
        # Economic indicators
        indicators = await adapter.get_economic_indicators()
        print(f"\n📊 Major Indices:")
        for index, data in indicators.items():
            if isinstance(data, dict) and data and 'change_percent' in data:
                print(f"   {index.upper()}: {data['change_percent']:.2f}% (${data['price']:.2f})")
            else:
                print(f"   {index.upper()}: No data available")
        
        # Currency data
        currencies = await adapter.get_currency_data()
        print(f"\n💱 Currency Performance:")
        for currency, data in currencies['currencies'].items():
            if isinstance(data, dict) and data and 'change_percent' in data:
                print(f"   {currency}: {data['change_percent']:.2f}%")
            else:
                print(f"   {currency}: No data available")
        
        # ==================== HEALTH CHECK ====================
        print("\n🏥 **SYSTEM HEALTH CHECK**")
        print("-" * 30)
        
        health = await adapter.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Connected: {health['connected']}")
        print(f"   Cache Size: {health['cache_size']}")
        
        print("\n" + "=" * 50)
        print("🎉 **POLYGON.IO INTEGRATION SUCCESSFUL!**")
        print("✅ 6 agents now have real market data")
        print("✅ Real-time quotes, intraday data, and institutional flow")
        print("✅ Performance rankings and sector analysis")
        print("✅ Financial statements and valuation metrics")
        print("✅ Economic indicators and currency data")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Polygon.io Integration Test...")
    
    success = await test_polygon_integration()
    
    if success:
        print("\n🎯 **NEXT STEPS:**")
        print("1. Integrate Polygon.io adapter into your agents")
        print("2. Replace mock data with real data calls")
        print("3. Test the enhanced system with real market data")
        print("4. Deploy to production")
    else:
        print("\n❌ **ISSUES FOUND:**")
        print("1. Check API key configuration")
        print("2. Verify network connectivity")
        print("3. Review error messages above")

if __name__ == "__main__":
    asyncio.run(main())
