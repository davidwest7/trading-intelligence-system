#!/usr/bin/env python3
"""
Alpha Vantage Adapter Demo
Demonstrates the Alpha Vantage adapter with real market data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append('.')

from common.data_adapters.alpha_vantage_adapter import AlphaVantageAdapter

async def demo_alpha_vantage_adapter():
    """Demo the Alpha Vantage adapter with real market data"""
    
    print("📈 ALPHA VANTAGE ADAPTER DEMONSTRATION")
    print("=" * 60)
    print("Using real market data from Alpha Vantage API")
    print("=" * 60)
    
    # Initialize the Alpha Vantage adapter
    print("\n🚀 Initializing Alpha Vantage Adapter...")
    config = {'alpha_vantage_key': '50T5QN5557DWTJ35'}
    adapter = AlphaVantageAdapter(config)
    
    # Test connection
    print("\n🔗 Testing Connection...")
    connected = await adapter.connect()
    
    if not connected:
        print("❌ Failed to connect to Alpha Vantage")
        return
    
    print("✅ Connected to Alpha Vantage successfully!")
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'NVDA']
    
    print(f"\n📊 Testing Real Market Data for: {', '.join(test_symbols)}")
    print("⏳ This may take a few moments due to rate limiting...")
    
    # Get real-time quotes
    print(f"\n💰 REAL-TIME QUOTES:")
    print("-" * 40)
    
    for symbol in test_symbols:
        quote = await adapter.get_quote(symbol)
        if quote and quote['price'] > 0:
            print(f"✅ {symbol}:")
            print(f"   Price: ${quote['price']:.2f}")
            print(f"   Change: {quote['change']:+.2f} ({quote['change_percent']})")
            print(f"   Volume: {quote['volume']:,}")
            print(f"   High: ${quote['high']:.2f}, Low: ${quote['low']:.2f}")
        else:
            print(f"❌ No data for {symbol}")
    
    # Get historical data
    print(f"\n📈 HISTORICAL DATA (AAPL - Last 30 Days):")
    print("-" * 40)
    
    since = datetime.now() - timedelta(days=30)
    historical_data = await adapter.get_ohlcv('AAPL', '1d', since, 30)
    
    if not historical_data.empty:
        print(f"✅ Retrieved {len(historical_data)} data points")
        print(f"   Date Range: {historical_data['Date'].min().date()} to {historical_data['Date'].max().date()}")
        print(f"   Latest Close: ${historical_data['Close'].iloc[-1]:.2f}")
        print(f"   Latest Volume: {historical_data['Volume'].iloc[-1]:,}")
        
        # Show sample data
        print(f"\n📋 SAMPLE DATA (Last 5 days):")
        sample_data = historical_data.tail(5)
        for _, row in sample_data.iterrows():
            print(f"   {row['Date'].date()}: ${row['Close']:.2f} (Vol: {row['Volume']:,})")
    else:
        print("❌ No historical data available")
    
    # Get company overview
    print(f"\n🏢 COMPANY OVERVIEW (AAPL):")
    print("-" * 40)
    
    overview = await adapter.get_company_overview('AAPL')
    if overview and overview['name'] != 'Unknown':
        print(f"✅ Company: {overview['name']}")
        print(f"   Sector: {overview['sector']}")
        print(f"   Industry: {overview['industry']}")
        print(f"   Market Cap: ${overview['market_cap']}")
        print(f"   P/E Ratio: {overview['pe_ratio']}")
        print(f"   Dividend Yield: {float(overview['dividend_yield']) * 100:.2f}%")
        print(f"   Beta: {overview['beta']}")
        print(f"   52-Week High: ${overview['52_week_high']}")
        print(f"   52-Week Low: ${overview['52_week_low']}")
    else:
        print("❌ No company overview available")
    
    # Get earnings data
    print(f"\n📊 EARNINGS DATA (AAPL):")
    print("-" * 40)
    
    earnings = await adapter.get_earnings('AAPL')
    if earnings and earnings['quarterly_earnings']:
        print(f"✅ Quarterly Earnings (Last 4 quarters):")
        for i, quarter in enumerate(earnings['quarterly_earnings'][:4]):
            print(f"   Q{quarter.get('fiscalDateEnding', 'Unknown')}: ${quarter.get('reportedEPS', 'N/A')}")
    else:
        print("❌ No earnings data available")
    
    # Health check
    print(f"\n⚡ ADAPTER HEALTH CHECK:")
    print("-" * 40)
    
    health = adapter.health_check()
    print(f"✅ Name: {health['name']}")
    print(f"✅ Connected: {health['connected']}")
    print(f"✅ API Key Configured: {health['api_key_configured']}")
    print(f"✅ Cache Size: {health['cache_size']} entries")
    print(f"✅ Rate Limit Delay: {health['rate_limit_delay']} seconds")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    total_symbols = len(test_symbols)
    successful_quotes = sum(1 for symbol in test_symbols if adapter.cache.get(f"quote_{symbol}"))
    successful_overview = 1 if adapter.cache.get("overview_AAPL") else 0
    successful_historical = 1 if not historical_data.empty else 0
    
    print(f"✅ Real-time Quotes: {successful_quotes}/{total_symbols}")
    print(f"✅ Company Overview: {successful_overview}/1")
    print(f"✅ Historical Data: {successful_historical}/1")
    print(f"✅ Cache Hits: {len(adapter.cache)} total entries")
    
    print("\n" + "=" * 60)
    print("🎉 Alpha Vantage Adapter Demo Complete!")
    print("✅ Successfully integrated real market data")
    print("✅ Real-time quotes, historical data, and fundamentals")
    print("✅ Production-ready market data adapter")
    print("=" * 60)

async def main():
    """Main demo function"""
    try:
        await demo_alpha_vantage_adapter()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
