#!/usr/bin/env python3
"""
Alpha Vantage Comprehensive Data Analysis
Check ALL available data points and capabilities
"""

import sys
import os
import asyncio
import requests
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

async def analyze_alpha_vantage_capabilities():
    """Analyze ALL available data points from Alpha Vantage"""
    
    print('🔍 ALPHA VANTAGE COMPREHENSIVE DATA ANALYSIS')
    print('='*60)
    print('🎯 CHECKING ALL AVAILABLE DATA POINTS')
    print()

    # Alpha Vantage API key
    api_key = '50T5QN5557DWTJ35'
    base_url = 'https://www.alphavantage.co/query'
    
    # Test different endpoints
    endpoints_to_test = [
        # Time Series Data
        ('function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey=' + api_key, 'Intraday Time Series'),
        ('function=TIME_SERIES_DAILY&symbol=AAPL&apikey=' + api_key, 'Daily Time Series'),
        ('function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&apikey=' + api_key, 'Daily Adjusted Time Series'),
        ('function=TIME_SERIES_WEEKLY&symbol=AAPL&apikey=' + api_key, 'Weekly Time Series'),
        ('function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=AAPL&apikey=' + api_key, 'Weekly Adjusted Time Series'),
        ('function=TIME_SERIES_MONTHLY&symbol=AAPL&apikey=' + api_key, 'Monthly Time Series'),
        ('function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=AAPL&apikey=' + api_key, 'Monthly Adjusted Time Series'),
        
        # Fundamental Data
        ('function=INCOME_STATEMENT&symbol=AAPL&apikey=' + api_key, 'Income Statement'),
        ('function=BALANCE_SHEET&symbol=AAPL&apikey=' + api_key, 'Balance Sheet'),
        ('function=CASH_FLOW&symbol=AAPL&apikey=' + api_key, 'Cash Flow'),
        ('function=EARNINGS&symbol=AAPL&apikey=' + api_key, 'Earnings'),
        ('function=LISTING_STATUS&apikey=' + api_key, 'Listing Status'),
        
        # Technical Indicators
        ('function=SMA&symbol=AAPL&interval=daily&time_period=10&series_type=close&apikey=' + api_key, 'Simple Moving Average'),
        ('function=EMA&symbol=AAPL&interval=daily&time_period=10&series_type=close&apikey=' + api_key, 'Exponential Moving Average'),
        ('function=RSI&symbol=AAPL&interval=daily&time_period=14&series_type=close&apikey=' + api_key, 'Relative Strength Index'),
        ('function=MACD&symbol=AAPL&interval=daily&series_type=close&apikey=' + api_key, 'MACD'),
        ('function=BBANDS&symbol=AAPL&interval=daily&time_period=20&series_type=close&apikey=' + api_key, 'Bollinger Bands'),
        ('function=STOCH&symbol=AAPL&interval=daily&apikey=' + api_key, 'Stochastic Oscillator'),
        ('function=ADX&symbol=AAPL&interval=daily&time_period=14&apikey=' + api_key, 'Average Directional Index'),
        ('function=CCI&symbol=AAPL&interval=daily&time_period=20&apikey=' + api_key, 'Commodity Channel Index'),
        ('function=AROON&symbol=AAPL&interval=daily&time_period=14&apikey=' + api_key, 'Aroon'),
        ('function=OBV&symbol=AAPL&interval=daily&apikey=' + api_key, 'On Balance Volume'),
        
        # Forex Data
        ('function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=JPY&apikey=' + api_key, 'Currency Exchange Rate'),
        ('function=FX_INTRADAY&from_symbol=USD&to_symbol=JPY&interval=1min&apikey=' + api_key, 'FX Intraday'),
        ('function=FX_DAILY&from_symbol=USD&to_symbol=JPY&apikey=' + api_key, 'FX Daily'),
        ('function=FX_WEEKLY&from_symbol=USD&to_symbol=JPY&apikey=' + api_key, 'FX Weekly'),
        ('function=FX_MONTHLY&from_symbol=USD&to_symbol=JPY&apikey=' + api_key, 'FX Monthly'),
        
        # Crypto Data
        ('function=CURRENCY_EXCHANGE_RATE&from_currency=BTC&to_currency=USD&apikey=' + api_key, 'Crypto Exchange Rate'),
        ('function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=' + api_key, 'Crypto Daily'),
        ('function=DIGITAL_CURRENCY_WEEKLY&symbol=BTC&market=USD&apikey=' + api_key, 'Crypto Weekly'),
        ('function=DIGITAL_CURRENCY_MONTHLY&symbol=BTC&market=USD&apikey=' + api_key, 'Crypto Monthly'),
        
        # Economic Indicators
        ('function=REAL_GDP&interval=quarterly&apikey=' + api_key, 'Real GDP'),
        ('function=REAL_GDP_PER_CAPITA&apikey=' + api_key, 'Real GDP Per Capita'),
        ('function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey=' + api_key, 'Treasury Yield'),
        ('function=FEDERAL_FUNDS_RATE&interval=monthly&apikey=' + api_key, 'Federal Funds Rate'),
        ('function=CPI&interval=monthly&apikey=' + api_key, 'Consumer Price Index'),
        ('function=INFLATION&apikey=' + api_key, 'Inflation'),
        ('function=INFLATION_EXPECTATION&apikey=' + api_key, 'Inflation Expectation'),
        ('function=CONSUMER_SENTIMENT&apikey=' + api_key, 'Consumer Sentiment'),
        ('function=RETAIL_SALES&apikey=' + api_key, 'Retail Sales'),
        ('function=DURABLES&apikey=' + api_key, 'Durables'),
        ('function=UNEMPLOYMENT&apikey=' + api_key, 'Unemployment'),
        ('function=NONFARM_PAYROLL&apikey=' + api_key, 'Nonfarm Payroll'),
        
        # Commodities
        ('function=WTI&interval=daily&apikey=' + api_key, 'WTI Crude Oil'),
        ('function=BRENT&interval=daily&apikey=' + api_key, 'Brent Crude Oil'),
        ('function=NATURAL_GAS&interval=daily&apikey=' + api_key, 'Natural Gas'),
        ('function=COPPER&interval=daily&apikey=' + api_key, 'Copper'),
        ('function=ALUMINUM&interval=daily&apikey=' + api_key, 'Aluminum'),
        ('function=WHEAT&interval=daily&apikey=' + api_key, 'Wheat'),
        ('function=CORN&interval=daily&apikey=' + api_key, 'Corn'),
        ('function=COTTON&interval=daily&apikey=' + api_key, 'Cotton'),
        ('function=SUGAR&interval=daily&apikey=' + api_key, 'Sugar'),
        ('function=COFFEE&interval=daily&apikey=' + api_key, 'Coffee'),
        
        # Sector Performance
        ('function=SECTOR&apikey=' + api_key, 'Sector Performance'),
        
        # Company Information
        ('function=OVERVIEW&symbol=AAPL&apikey=' + api_key, 'Company Overview'),
        ('function=SYMBOL_SEARCH&keywords=AAPL&apikey=' + api_key, 'Symbol Search'),
        
        # News & Sentiment
        ('function=NEWS_SENTIMENT&tickers=AAPL&apikey=' + api_key, 'News Sentiment'),
        ('function=TOP_GAINERS_LOSERS&apikey=' + api_key, 'Top Gainers/Losers'),
        
        # Global Market Data
        ('function=GLOBAL_QUOTE&symbol=AAPL&apikey=' + api_key, 'Global Quote'),
        ('function=QUOTE_ENDPOINT&symbol=AAPL&apikey=' + api_key, 'Quote Endpoint'),
    ]
    
    successful_endpoints = []
    failed_endpoints = []
    
    print('🔍 TESTING ALL ALPHA VANTAGE ENDPOINTS...')
    print()
    
    for endpoint, description in endpoints_to_test:
        try:
            url = f"{base_url}?{endpoint}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we got valid data (not error message)
                if 'Error Message' not in data and 'Note' not in data:
                    successful_endpoints.append((endpoint, description))
                    print(f'✅ {description}')
                else:
                    error_msg = data.get('Error Message', data.get('Note', 'Unknown error'))
                    failed_endpoints.append((endpoint, description, error_msg))
                    print(f'❌ {description}: {error_msg[:50]}...')
            else:
                failed_endpoints.append((endpoint, description, f'HTTP {response.status_code}'))
                print(f'❌ {description}: HTTP {response.status_code}')
                
        except Exception as e:
            failed_endpoints.append((endpoint, description, str(e)))
            print(f'❌ {description}: Error: {e}')
    
    print()
    print('📊 ALPHA VANTAGE CAPABILITIES SUMMARY')
    print('='*50)
    print(f'✅ Successful Endpoints: {len(successful_endpoints)}')
    print(f'❌ Failed Endpoints: {len(failed_endpoints)}')
    print()
    
    print('🎯 AVAILABLE DATA POINTS:')
    print('='*30)
    
    # Categorize successful endpoints
    categories = {
        'Time Series': [],
        'Fundamental Data': [],
        'Technical Indicators': [],
        'Forex': [],
        'Crypto': [],
        'Economic Indicators': [],
        'Commodities': [],
        'Sector Performance': [],
        'Company Info': [],
        'News & Sentiment': [],
        'Global Market': []
    }
    
    for endpoint, description in successful_endpoints:
        if 'TIME_SERIES' in endpoint:
            categories['Time Series'].append(description)
        elif 'INCOME_STATEMENT' in endpoint or 'BALANCE_SHEET' in endpoint or 'CASH_FLOW' in endpoint or 'EARNINGS' in endpoint:
            categories['Fundamental Data'].append(description)
        elif any(indicator in endpoint for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'ADX', 'CCI', 'AROON', 'OBV']):
            categories['Technical Indicators'].append(description)
        elif 'FX_' in endpoint or 'CURRENCY_EXCHANGE_RATE' in endpoint:
            categories['Forex'].append(description)
        elif 'DIGITAL_CURRENCY' in endpoint or ('BTC' in endpoint and 'CURRENCY' in endpoint):
            categories['Crypto'].append(description)
        elif any(econ in endpoint for econ in ['GDP', 'TREASURY', 'FEDERAL', 'CPI', 'INFLATION', 'CONSUMER', 'RETAIL', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM']):
            categories['Economic Indicators'].append(description)
        elif any(commodity in endpoint for commodity in ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE']):
            categories['Commodities'].append(description)
        elif 'SECTOR' in endpoint:
            categories['Sector Performance'].append(description)
        elif 'OVERVIEW' in endpoint or 'SYMBOL_SEARCH' in endpoint:
            categories['Company Info'].append(description)
        elif 'NEWS' in endpoint or 'GAINERS' in endpoint:
            categories['News & Sentiment'].append(description)
        elif 'GLOBAL' in endpoint or 'QUOTE' in endpoint:
            categories['Global Market'].append(description)
    
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
        'Technical Agent': ['Time Series', 'Technical Indicators'],
        'Flow Agent': ['Time Series', 'Global Market'],
        'Money Flows Agent': ['Time Series', 'Global Market'],
        'Top Performers Agent': ['Time Series', 'Sector Performance'],
        'Undervalued Agent': ['Fundamental Data', 'Company Info'],
        'Macro Agent': ['Economic Indicators', 'Commodities'],
        'Causal Agent': ['News & Sentiment', 'Fundamental Data'],
        'Insider Agent': ['Fundamental Data', 'Company Info'],
        'Sentiment Agent': ['News & Sentiment'],
        'Learning Agent': ['Time Series', 'Technical Indicators']
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
    print(f'💵 Alpha Vantage: $49.99/month (ALREADY PAID)')
    print(f'📊 Coverage: {len([cat for cat in categories.values() if cat])} data categories')
    print(f'🎯 Agent Coverage: {len([agent for agent, cats in agent_coverage.items() if all(categories[cat] for cat in cats)])}/10 agents')
    
    print()
    print('🚀 RECOMMENDATIONS:')
    print('='*20)
    print('1. ✅ Alpha Vantage provides excellent fundamental data')
    print('2. 📊 Strong coverage for economic indicators')
    print('3. 💰 Good value for fundamental analysis')
    print('4. 🔄 Complements Polygon.io Pro well')

if __name__ == "__main__":
    asyncio.run(analyze_alpha_vantage_capabilities())
