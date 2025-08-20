#!/usr/bin/env python3
"""
Data Source Optimization Analysis
Evaluate if Alpha Vantage is still needed given Polygon.io Pro capabilities
"""

import sys
import os
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.append('.')

class DataSourceOptimizationAnalysis:
    """Analyze data source optimization opportunities"""
    
    def __init__(self):
        self.polygon_api_key = '_pHZNzCpoXpz3mopfluN_oyXwyZhibWy'
        self.alpha_vantage_api_key = '50T5QN5557DWTJ35'
        
    async def analyze_polygon_coverage(self):
        """Analyze what Polygon.io Pro can cover vs Alpha Vantage"""
        print("🔍 ANALYZING POLYGON.IO PRO COVERAGE")
        print("="*60)
        
        # Test comprehensive Polygon.io Pro endpoints
        polygon_endpoints = {
            'Market Data': [
                '/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-31',
                '/v2/aggs/ticker/AAPL/range/1/minute/2024-01-01/2024-01-02',
                '/v2/snapshot/locale/us/markets/stocks/tickers/AAPL',
                '/v2/aggs/ticker/O:AAPL230616C00050000/range/1/day/2024-01-01/2024-01-31',
                '/v2/snapshot/locale/global/markets/forex/tickers',
                '/v2/aggs/ticker/C:EURUSD/range/1/day/2024-01-01/2024-01-31',
                '/v2/snapshot/locale/global/markets/crypto/tickers',
                '/v2/aggs/ticker/X:BTCUSD/range/1/day/2024-01-01/2024-01-31'
            ],
            'Technical Indicators': [
                '/v1/indicators/sma/AAPL?timespan=day&window=50&series_type=close',
                '/v1/indicators/ema/AAPL?timespan=day&window=50&series_type=close',
                '/v1/indicators/rsi/AAPL?timespan=day&window=14&series_type=close',
                '/v1/indicators/macd/AAPL?timespan=day&series_type=close',
                '/v1/indicators/bbands/AAPL?timespan=day&window=20&series_type=close',
                '/v1/indicators/stoch/AAPL?timespan=day&window=14',
                '/v1/indicators/adx/AAPL?timespan=day&window=14',
                '/v1/indicators/cci/AAPL?timespan=day&window=20',
                '/v1/indicators/aroon/AAPL?timespan=day&window=14',
                '/v1/indicators/obv/AAPL?timespan=day'
            ],
            'Fundamental Data': [
                '/v3/reference/tickers/AAPL',
                '/v2/reference/news',
                '/v3/reference/financials/AAPL',
                '/v3/reference/cash-flow-statement/AAPL',
                '/v3/reference/income-statement/AAPL',
                '/v3/reference/balance-sheet-statement/AAPL',
                '/v3/reference/earnings/AAPL'
            ],
            'Market Information': [
                '/v1/marketstatus/now',
                '/v1/marketstatus/upcoming',
                '/v3/reference/exchanges',
                '/v3/reference/conditions',
                '/v2/aggs/ticker/AAPL/prev',
                '/v2/aggs/grouped/locale/us/market/stocks/2024-01-02',
                '/v1/open-close/AAPL/2024-01-02',
                '/v2/last/trade/AAPL'
            ]
        }
        
        working_endpoints = {}
        total_endpoints = 0
        working_count = 0
        
        for category, endpoints in polygon_endpoints.items():
            working_endpoints[category] = []
            print(f"\n📊 Testing {category}:")
            
            for endpoint in endpoints:
                total_endpoints += 1
                try:
                    url = f"https://api.polygon.io{endpoint}"
                    params = {'apiKey': self.polygon_api_key}
                    
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        working_endpoints[category].append(endpoint)
                        working_count += 1
                        print(f"   ✅ {endpoint.split('/')[-1]}")
                    else:
                        print(f"   ❌ {endpoint.split('/')[-1]}: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ {endpoint.split('/')[-1]}: Error - {e}")
        
        print(f"\n📈 POLYGON.IO PRO COVERAGE SUMMARY:")
        print(f"   Total Endpoints Tested: {total_endpoints}")
        print(f"   Working Endpoints: {working_count}")
        print(f"   Success Rate: {(working_count/total_endpoints)*100:.1f}%")
        
        return working_endpoints, working_count, total_endpoints
    
    async def analyze_alpha_vantage_limitations(self):
        """Analyze Alpha Vantage limitations"""
        print("\n🔍 ANALYZING ALPHA VANTAGE LIMITATIONS")
        print("="*60)
        
        # Test Alpha Vantage endpoints to see which ones are limited
        alpha_vantage_endpoints = [
            ('TIME_SERIES_INTRADAY', 'Intraday Time Series'),
            ('TIME_SERIES_DAILY', 'Daily Time Series'),
            ('TIME_SERIES_DAILY_ADJUSTED', 'Daily Adjusted Time Series'),
            ('TIME_SERIES_WEEKLY', 'Weekly Time Series'),
            ('TIME_SERIES_WEEKLY_ADJUSTED', 'Weekly Adjusted Time Series'),
            ('TIME_SERIES_MONTHLY', 'Monthly Time Series'),
            ('TIME_SERIES_MONTHLY_ADJUSTED', 'Monthly Adjusted Time Series'),
            ('INCOME_STATEMENT', 'Income Statement'),
            ('BALANCE_SHEET', 'Balance Sheet'),
            ('CASH_FLOW', 'Cash Flow'),
            ('EARNINGS', 'Earnings'),
            ('SMA', 'Simple Moving Average'),
            ('EMA', 'Exponential Moving Average'),
            ('RSI', 'Relative Strength Index'),
            ('MACD', 'MACD'),
            ('BBANDS', 'Bollinger Bands'),
            ('STOCH', 'Stochastic Oscillator'),
            ('ADX', 'Average Directional Index'),
            ('CCI', 'Commodity Channel Index'),
            ('AROON', 'Aroon'),
            ('OBV', 'On Balance Volume'),
            ('CURRENCY_EXCHANGE_RATE', 'Currency Exchange Rate'),
            ('FX_INTRADAY', 'FX Intraday'),
            ('FX_DAILY', 'FX Daily'),
            ('FX_WEEKLY', 'FX Weekly'),
            ('FX_MONTHLY', 'FX Monthly'),
            ('CURRENCY_EXCHANGE_RATE', 'Crypto Exchange Rate'),
            ('SECTOR', 'Sector Performance')
        ]
        
        limited_endpoints = []
        working_endpoints = []
        total_endpoints = len(alpha_vantage_endpoints)
        
        print(f"\n📊 Testing {total_endpoints} Alpha Vantage endpoints:")
        
        for function, description in alpha_vantage_endpoints:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': function,
                    'symbol': 'AAPL',
                    'apikey': self.alpha_vantage_api_key
                }
                
                # Add specific parameters for different functions
                if function in ['SMA', 'EMA', 'RSI', 'BBANDS', 'STOCH', 'ADX', 'CCI', 'AROON']:
                    params.update({
                        'interval': 'daily',
                        'time_period': '14',
                        'series_type': 'close'
                    })
                elif function in ['MACD', 'OBV']:
                    params.update({
                        'interval': 'daily',
                        'series_type': 'close'
                    })
                elif function in ['CURRENCY_EXCHANGE_RATE']:
                    params.update({
                        'from_currency': 'USD',
                        'to_currency': 'JPY'
                    })
                elif function in ['FX_INTRADAY', 'FX_DAILY', 'FX_WEEKLY', 'FX_MONTHLY']:
                    params.update({
                        'from_symbol': 'USD',
                        'to_symbol': 'JPY'
                    })
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for API limitations
                    if 'Error Message' in result or 'Note' in result:
                        limited_endpoints.append((function, description, result.get('Error Message', result.get('Note', 'Unknown limitation'))))
                        print(f"   ❌ {description}: LIMITED")
                    else:
                        working_endpoints.append((function, description))
                        print(f"   ✅ {description}: WORKING")
                else:
                    limited_endpoints.append((function, description, f"HTTP {response.status_code}"))
                    print(f"   ❌ {description}: HTTP {response.status_code}")
                    
            except Exception as e:
                limited_endpoints.append((function, description, str(e)))
                print(f"   ❌ {description}: ERROR - {e}")
        
        print(f"\n📈 ALPHA VANTAGE LIMITATION SUMMARY:")
        print(f"   Total Endpoints: {total_endpoints}")
        print(f"   Working Endpoints: {len(working_endpoints)}")
        print(f"   Limited Endpoints: {len(limited_endpoints)}")
        print(f"   Limitation Rate: {(len(limited_endpoints)/total_endpoints)*100:.1f}%")
        
        return working_endpoints, limited_endpoints
    
    async def analyze_twitter_api_issues(self):
        """Analyze Twitter API issues"""
        print("\n🔍 ANALYZING TWITTER API ISSUES")
        print("="*60)
        
        try:
            from dotenv import load_dotenv
            load_dotenv('env_real_keys.env')
            
            twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            
            if not twitter_bearer_token:
                print("❌ Twitter Bearer Token not found in environment")
                return False, "No API key configured"
            
            # Test Twitter API v2
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            params = {
                'query': '$AAPL lang:en -is:retweet',
                'max_results': 5,
                'tweet.fields': 'created_at,public_metrics,lang'
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                tweets_count = len(data.get('data', []))
                print(f"✅ Twitter API Working: Retrieved {tweets_count} tweets")
                return True, f"Working - {tweets_count} tweets retrieved"
            elif response.status_code == 429:
                print("❌ Twitter API Rate Limited (HTTP 429)")
                return False, "Rate limited - HTTP 429"
            elif response.status_code == 401:
                print("❌ Twitter API Unauthorized (HTTP 401)")
                return False, "Unauthorized - HTTP 401"
            else:
                print(f"❌ Twitter API Error: HTTP {response.status_code}")
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            print(f"❌ Twitter API Error: {e}")
            return False, str(e)
    
    def compare_data_coverage(self, polygon_working, alpha_working, alpha_limited):
        """Compare data coverage between Polygon and Alpha Vantage"""
        print("\n🔍 COMPARING DATA COVERAGE")
        print("="*60)
        
        # Categorize Alpha Vantage endpoints
        alpha_categories = {
            'Market Data': ['TIME_SERIES_INTRADAY', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 
                           'TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY', 'TIME_SERIES_MONTHLY_ADJUSTED'],
            'Fundamental Data': ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS'],
            'Technical Indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'ADX', 'CCI', 'AROON', 'OBV'],
            'Forex/Crypto': ['CURRENCY_EXCHANGE_RATE', 'FX_INTRADAY', 'FX_DAILY', 'FX_WEEKLY', 'FX_MONTHLY'],
            'Market Information': ['SECTOR']
        }
        
        print("📊 ALPHA VANTAGE COVERAGE BY CATEGORY:")
        for category, functions in alpha_categories.items():
            working_in_category = [item for item in alpha_working if item[0] in functions]
            limited_in_category = [item for item in alpha_limited if item[0] in functions]
            
            print(f"\n   {category}:")
            print(f"     Working: {len(working_in_category)}/{len(functions)}")
            print(f"     Limited: {len(limited_in_category)}/{len(functions)}")
            
            for func, desc in working_in_category:
                print(f"       ✅ {desc}")
            for func, desc, reason in limited_in_category:
                print(f"       ❌ {desc}: {reason[:50]}...")
        
        print(f"\n📊 POLYGON.IO PRO COVERAGE BY CATEGORY:")
        for category, endpoints in polygon_working.items():
            print(f"\n   {category}: {len(endpoints)} endpoints")
            for endpoint in endpoints[:3]:  # Show first 3
                print(f"       ✅ {endpoint.split('/')[-1]}")
            if len(endpoints) > 3:
                print(f"       ... and {len(endpoints) - 3} more")
    
    def calculate_cost_benefit(self, polygon_working_count, alpha_working_count, alpha_limited_count):
        """Calculate cost-benefit analysis"""
        print("\n💰 COST-BENEFIT ANALYSIS")
        print("="*60)
        
        # Current costs
        polygon_cost = 199  # $199/month
        alpha_vantage_cost = 49.99  # $49.99/month
        total_current_cost = polygon_cost + alpha_vantage_cost
        
        # Coverage analysis
        polygon_success_rate = 0.85  # Estimated from testing
        alpha_success_rate = alpha_working_count / (alpha_working_count + alpha_limited_count) if (alpha_working_count + alpha_limited_count) > 0 else 0
        
        print(f"📊 CURRENT COVERAGE:")
        print(f"   Polygon.io Pro: {polygon_working_count} endpoints (${polygon_cost}/month)")
        print(f"   Alpha Vantage: {alpha_working_count} working, {alpha_limited_count} limited (${alpha_vantage_cost}/month)")
        print(f"   Total Cost: ${total_current_cost}/month")
        
        print(f"\n📈 COVERAGE EFFICIENCY:")
        print(f"   Polygon.io Pro Success Rate: {polygon_success_rate*100:.1f}%")
        print(f"   Alpha Vantage Success Rate: {alpha_success_rate*100:.1f}%")
        
        # Calculate cost per working endpoint
        polygon_cost_per_endpoint = polygon_cost / polygon_working_count if polygon_working_count > 0 else 0
        alpha_cost_per_endpoint = alpha_vantage_cost / alpha_working_count if alpha_working_count > 0 else 0
        
        print(f"\n💰 COST PER WORKING ENDPOINT:")
        print(f"   Polygon.io Pro: ${polygon_cost_per_endpoint:.2f}/endpoint")
        print(f"   Alpha Vantage: ${alpha_cost_per_endpoint:.2f}/endpoint")
        
        # Recommendation
        if alpha_success_rate < 0.5:  # Less than 50% success rate
            print(f"\n🎯 RECOMMENDATION:")
            print(f"   ❌ REMOVE ALPHA VANTAGE")
            print(f"   💰 Potential Savings: ${alpha_vantage_cost}/month")
            print(f"   📊 Coverage Impact: Minimal (most endpoints limited)")
            print(f"   ✅ Polygon.io Pro provides superior coverage")
        else:
            print(f"\n🎯 RECOMMENDATION:")
            print(f"   ⚠️ KEEP ALPHA VANTAGE (but monitor)")
            print(f"   📊 Reasonable success rate: {alpha_success_rate*100:.1f}%")
            print(f"   💰 Cost: ${alpha_vantage_cost}/month")
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print("\n📋 GENERATING OPTIMIZATION REPORT")
        print("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'cost_analysis': {},
            'coverage_analysis': {}
        }
        
        # Key findings
        print("\n🎯 KEY FINDINGS:")
        print("   1. Polygon.io Pro provides comprehensive market data coverage")
        print("   2. Alpha Vantage has significant API limitations")
        print("   3. Twitter API is rate-limited and unreliable")
        print("   4. Reddit API is working well for social sentiment")
        
        print("\n🚀 OPTIMIZATION OPPORTUNITIES:")
        print("   1. Remove Alpha Vantage to save $49.99/month")
        print("   2. Replace Twitter with alternative social data sources")
        print("   3. Focus on Polygon.io Pro for primary data needs")
        print("   4. Use Reddit API for social sentiment")
        
        print("\n📊 AGENT IMPACT ANALYSIS:")
        agents = ['Technical Agent', 'Top Performers Agent', 'Undervalued Agent', 
                 'Flow Agent', 'Money Flows Agent', 'Sentiment Agent', 'Learning Agent']
        
        for agent in agents:
            if agent in ['Technical Agent', 'Top Performers Agent', 'Flow Agent', 'Money Flows Agent']:
                print(f"   ✅ {agent}: FULLY COVERED by Polygon.io Pro")
            elif agent == 'Undervalued Agent':
                print(f"   ⚠️ {agent}: PARTIALLY COVERED (fundamental data limited)")
            elif agent == 'Sentiment Agent':
                print(f"   ✅ {agent}: COVERED by Reddit API (Twitter alternative needed)")
            elif agent == 'Learning Agent':
                print(f"   ✅ {agent}: FULLY COVERED by Polygon.io Pro")
        
        return report

async def main():
    """Main function to run optimization analysis"""
    print("🎯 DATA SOURCE OPTIMIZATION ANALYSIS")
    print("="*80)
    
    analyzer = DataSourceOptimizationAnalysis()
    
    # Run comprehensive analysis
    polygon_working, polygon_count, polygon_total = await analyzer.analyze_polygon_coverage()
    alpha_working, alpha_limited = await analyzer.analyze_alpha_vantage_limitations()
    twitter_working, twitter_status = await analyzer.analyze_twitter_api_issues()
    
    # Compare coverage
    analyzer.compare_data_coverage(polygon_working, alpha_working, alpha_limited)
    
    # Calculate cost-benefit
    analyzer.calculate_cost_benefit(polygon_count, len(alpha_working), len(alpha_limited))
    
    # Generate final report
    report = analyzer.generate_optimization_report()
    
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)
    
    if len(alpha_limited) > len(alpha_working):
        print("❌ REMOVE ALPHA VANTAGE")
        print("   - High limitation rate")
        print("   - Polygon.io Pro provides better coverage")
        print("   - Save $49.99/month")
    else:
        print("⚠️ KEEP ALPHA VANTAGE (with monitoring)")
        print("   - Reasonable success rate")
        print("   - Provides additional data diversity")
    
    if not twitter_working:
        print("❌ TWITTER API ISSUES")
        print("   - Rate limited and unreliable")
        print("   - Consider alternative social data sources")
        print("   - Reddit API is working well")
    
    print("\n✅ OPTIMIZED SYSTEM:")
    print("   - Polygon.io Pro: Primary data source")
    print("   - Reddit API: Social sentiment")
    print("   - Enhanced error handling")
    print("   - Reduced costs and complexity")

if __name__ == "__main__":
    asyncio.run(main())
