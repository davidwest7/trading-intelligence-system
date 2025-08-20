#!/usr/bin/env python3
"""
Polygon.io Pro Comprehensive Data Analysis
Check ALL available data points and capabilities
"""

import sys
import os
import asyncio
import requests
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

async def analyze_polygon_pro_capabilities():
    """Analyze ALL available data points from Polygon.io Pro"""
    
    print('🔍 POLYGON.IO PRO COMPREHENSIVE DATA ANALYSIS')
    print('='*60)
    print('🎯 CHECKING ALL AVAILABLE DATA POINTS')
    print()

    # Polygon.io Pro API key
    api_key = '_pHZNzCpoXpz3mopfluN_oyXwyZhibWy'
    base_url = 'https://api.polygon.io'
    
    # Test different endpoints
    endpoints_to_test = [
        # Market Data
        ('/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31', 'Daily Aggregates'),
        ('/v2/aggs/ticker/AAPL/range/1/minute/2024-01-01/2024-01-02', 'Intraday Aggregates'),
        ('/v2/snapshot/locale/us/markets/stocks/tickers', 'Market Snapshot'),
        ('/v2/snapshot/locale/us/markets/stocks/tickers/AAPL', 'Stock Snapshot'),
        
        # Options Data
        ('/v3/snapshot/options/AAPL', 'Options Snapshot'),
        ('/v3/reference/options/contracts', 'Options Contracts'),
        ('/v2/aggs/ticker/O:AAPL230616C00050000/range/1/day/2024-01-01/2024-01-31', 'Options Aggregates'),
        
        # Forex Data
        ('/v2/snapshot/locale/global/markets/forex/tickers', 'Forex Snapshot'),
        ('/v2/aggs/ticker/C:EURUSD/range/1/day/2024-01-01/2024-01-31', 'Forex Aggregates'),
        
        # Crypto Data
        ('/v2/snapshot/locale/global/markets/crypto/tickers', 'Crypto Snapshot'),
        ('/v2/aggs/ticker/X:BTCUSD/range/1/day/2024-01-01/2024-01-31', 'Crypto Aggregates'),
        
        # Futures Data
        ('/v2/snapshot/locale/global/markets/crypto/tickers', 'Futures Snapshot'),
        ('/v2/aggs/ticker/ES:ES1!/range/1/day/2024-01-01/2024-01-31', 'Futures Aggregates'),
        
        # Level 2 Data
        ('/v2/snapshot/locale/us/markets/stocks/tickers/AAPL/book', 'Level 2 Book'),
        
        # Trades Data
        ('/v3/trades/AAPL/2024-01-02', 'Trades Data'),
        
        # Quotes Data
        ('/v3/quotes/AAPL/2024-01-02', 'Quotes Data'),
        
        # Technical Indicators
        ('/v1/indicators/sma/AAPL?timespan=day&window=50&series_type=close', 'SMA Indicator'),
        ('/v1/indicators/ema/AAPL?timespan=day&window=50&series_type=close', 'EMA Indicator'),
        ('/v1/indicators/rsi/AAPL?timespan=day&window=14&series_type=close', 'RSI Indicator'),
        ('/v1/indicators/macd/AAPL?timespan=day&series_type=close', 'MACD Indicator'),
        
        # Reference Data
        ('/v3/reference/tickers', 'Tickers Reference'),
        ('/v3/reference/tickers/AAPL', 'Ticker Details'),
        ('/v3/reference/tickers/AAPL/financials', 'Financials'),
        ('/v3/reference/tickers/AAPL/earnings', 'Earnings'),
        ('/v3/reference/tickers/AAPL/dividends', 'Dividends'),
        ('/v3/reference/tickers/AAPL/splits', 'Splits'),
        
        # News Data
        ('/v2/reference/news', 'News Articles'),
        ('/v2/reference/news/ticker/AAPL', 'Ticker News'),
        
        # Economic Data
        ('/v1/indicators/values/UNRATE', 'Economic Indicators'),
        
        # Market Status
        ('/v1/marketstatus/now', 'Market Status'),
        ('/v1/marketstatus/upcoming', 'Upcoming Market Events'),
        
        # Exchanges
        ('/v3/reference/exchanges', 'Exchanges'),
        ('/v3/reference/exchanges/AAPL', 'Exchange Details'),
        
        # Market Holidays
        ('/v1/marketstatus/upcoming', 'Market Holidays'),
        
        # Conditions
        ('/v3/reference/conditions', 'Trade Conditions'),
        
        # Sectors
        ('/v3/reference/sectors', 'Sectors'),
        ('/v3/reference/sectors/AAPL', 'Ticker Sectors'),
        
        # Previous Close
        ('/v2/aggs/ticker/AAPL/prev', 'Previous Close'),
        
        # Grouped Daily
        ('/v2/aggs/grouped/locale/us/market/stocks/2024-01-02', 'Grouped Daily'),
        
        # Open/Close
        ('/v1/open-close/AAPL/2024-01-02', 'Open/Close'),
        
        # Last Trade
        ('/v2/last/trade/AAPL', 'Last Trade'),
        
        # Last Quote
        ('/v2/last/quote/AAPL', 'Last Quote'),
    ]
    
    successful_endpoints = []
    failed_endpoints = []
    
    print('🔍 TESTING ALL POLYGON.IO PRO ENDPOINTS...')
    print()
    
    for endpoint, description in endpoints_to_test:
        try:
            url = f"{base_url}{endpoint}"
            params = {'apiKey': api_key}
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                successful_endpoints.append((endpoint, description))
                print(f'✅ {description}: {endpoint}')
            else:
                failed_endpoints.append((endpoint, description, response.status_code))
                print(f'❌ {description}: {endpoint} (HTTP {response.status_code})')
                
        except Exception as e:
            failed_endpoints.append((endpoint, description, str(e)))
            print(f'❌ {description}: {endpoint} (Error: {e})')
    
    print()
    print('📊 POLYGON.IO PRO CAPABILITIES SUMMARY')
    print('='*50)
    print(f'✅ Successful Endpoints: {len(successful_endpoints)}')
    print(f'❌ Failed Endpoints: {len(failed_endpoints)}')
    print()
    
    print('🎯 AVAILABLE DATA POINTS:')
    print('='*30)
    
    # Categorize successful endpoints
    categories = {
        'Market Data': [],
        'Options': [],
        'Forex': [],
        'Crypto': [],
        'Futures': [],
        'Level 2': [],
        'Trades/Quotes': [],
        'Technical Indicators': [],
        'Reference Data': [],
        'News': [],
        'Economic': [],
        'Market Status': [],
        'Other': []
    }
    
    for endpoint, description in successful_endpoints:
        if 'options' in endpoint.lower():
            categories['Options'].append(description)
        elif 'forex' in endpoint.lower() or 'eurusd' in endpoint.lower():
            categories['Forex'].append(description)
        elif 'crypto' in endpoint.lower() or 'btcusd' in endpoint.lower():
            categories['Crypto'].append(description)
        elif 'futures' in endpoint.lower() or 'es1' in endpoint.lower():
            categories['Futures'].append(description)
        elif 'book' in endpoint.lower():
            categories['Level 2'].append(description)
        elif 'trades' in endpoint.lower() or 'quotes' in endpoint.lower():
            categories['Trades/Quotes'].append(description)
        elif 'indicators' in endpoint.lower():
            categories['Technical Indicators'].append(description)
        elif 'reference' in endpoint.lower() or 'financials' in endpoint.lower() or 'earnings' in endpoint.lower():
            categories['Reference Data'].append(description)
        elif 'news' in endpoint.lower():
            categories['News'].append(description)
        elif 'unrate' in endpoint.lower():
            categories['Economic'].append(description)
        elif 'marketstatus' in endpoint.lower():
            categories['Market Status'].append(description)
        elif 'aggs' in endpoint.lower() or 'snapshot' in endpoint.lower():
            categories['Market Data'].append(description)
        else:
            categories['Other'].append(description)
    
    for category, items in categories.items():
        if items:
            print(f'\n📊 {category.upper()}:')
            for item in items:
                print(f'   ✅ {item}')
    
    print()
    print('🎯 AGENT COVERAGE ANALYSIS:')
    print('='*30)
    
    # Map to agents
    agent_coverage = {
        'Technical Agent': ['Market Data', 'Technical Indicators', 'Trades/Quotes'],
        'Flow Agent': ['Level 2', 'Trades/Quotes', 'Market Data'],
        'Money Flows Agent': ['Level 2', 'Trades/Quotes'],
        'Top Performers Agent': ['Market Data', 'Reference Data'],
        'Undervalued Agent': ['Reference Data', 'Market Data'],
        'Macro Agent': ['Economic', 'Market Status'],
        'Causal Agent': ['News', 'Reference Data'],
        'Insider Agent': ['Reference Data', 'Trades/Quotes'],
        'Sentiment Agent': ['News'],
        'Learning Agent': ['Market Data', 'Technical Indicators']
    }
    
    for agent, required_categories in agent_coverage.items():
        available_categories = [cat for cat in required_categories if categories[cat]]
        coverage_percentage = len(available_categories) / len(required_categories) * 100
        
        if coverage_percentage == 100:
            print(f'✅ {agent}: 100% COVERED')
        elif coverage_percentage > 50:
            print(f'🟡 {agent}: {coverage_percentage:.0f}% COVERED')
        else:
            print(f'❌ {agent}: {coverage_percentage:.0f}% COVERED')
    
    print()
    print('💰 COST ANALYSIS:')
    print('='*20)
    print(f'💵 Polygon.io Pro: $199/month (ALREADY PAID)')
    print(f'📊 Coverage: {len([cat for cat in categories.values() if cat])} data categories')
    print(f'🎯 Agent Coverage: {len([agent for agent, cats in agent_coverage.items() if all(categories[cat] for cat in cats)])}/10 agents')
    
    print()
    print('🚀 RECOMMENDATIONS:')
    print('='*20)
    print('1. ✅ Polygon.io Pro covers MOST data needs')
    print('2. 🔍 Check failed endpoints for additional capabilities')
    print('3. 📊 Maximize usage of available endpoints')
    print('4. 💰 Significant cost savings vs individual APIs')

if __name__ == "__main__":
    asyncio.run(analyze_polygon_pro_capabilities())
