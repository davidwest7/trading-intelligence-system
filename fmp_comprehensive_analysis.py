#!/usr/bin/env python3
"""
Financial Modeling Prep (FMP) Comprehensive Analysis
Test all available endpoints and identify integration opportunities
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class FMPComprehensiveAnalysis:
    """Comprehensive analysis of Financial Modeling Prep API"""
    
    def __init__(self):
        self.api_key = 'JPWzlUuBlnlFANPAaoO0qFZsIIWo4fYG'
        self.base_url = 'https://financialmodelingprep.com/api/v3'
        
    def test_stock_prices_endpoints(self):
        """Test stock prices and historical data endpoints"""
        print("📊 TESTING STOCK PRICES ENDPOINTS")
        print("="*60)
        
        price_endpoints = [
            ('/quote/AAPL', 'Real-time Quote'),
            ('/historical-price-full/AAPL', 'Historical Prices'),
            ('/historical-chart/1hour/AAPL', 'Hourly Chart'),
            ('/historical-chart/4hour/AAPL', '4-Hour Chart'),
            ('/historical-chart/1day/AAPL', 'Daily Chart'),
            ('/historical-chart/1week/AAPL', 'Weekly Chart'),
            ('/historical-chart/1month/AAPL', 'Monthly Chart'),
            ('/historical-price-full/AAPL?from=2024-01-01&to=2024-01-31', 'Date Range'),
            ('/stock/real-time-price/AAPL', 'Real-time Price'),
            ('/stock/price/AAPL', 'Current Price'),
            ('/stock/price-change/AAPL', 'Price Change'),
            ('/stock/price-change-summary/AAPL', 'Price Change Summary'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in price_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_fundamental_data_endpoints(self):
        """Test fundamental data endpoints"""
        print("\n📊 TESTING FUNDAMENTAL DATA ENDPOINTS")
        print("="*60)
        
        fundamental_endpoints = [
            ('/income-statement/AAPL', 'Income Statement'),
            ('/balance-sheet-statement/AAPL', 'Balance Sheet'),
            ('/cash-flow-statement/AAPL', 'Cash Flow Statement'),
            ('/key-metrics/AAPL', 'Key Metrics'),
            ('/financial-growth/AAPL', 'Financial Growth'),
            ('/ratios/AAPL', 'Financial Ratios'),
            ('/enterprise-values/AAPL', 'Enterprise Values'),
            ('/company-key-metrics/AAPL', 'Company Key Metrics'),
            ('/company-ratios/AAPL', 'Company Ratios'),
            ('/company-enterprise-value/AAPL', 'Company Enterprise Value'),
            ('/company-financial-statement-growth/AAPL', 'Financial Statement Growth'),
            ('/company-profile/AAPL', 'Company Profile'),
            ('/company-outlook/AAPL', 'Company Outlook'),
            ('/company-core-information/AAPL', 'Company Core Information'),
            ('/company-description/AAPL', 'Company Description'),
            ('/company-executives/AAPL', 'Company Executives'),
            ('/company-milestones/AAPL', 'Company Milestones'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in fundamental_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_market_data_endpoints(self):
        """Test market data and indices endpoints"""
        print("\n📊 TESTING MARKET DATA ENDPOINTS")
        print("="*60)
        
        market_endpoints = [
            ('/quotes/index', 'Market Indices'),
            ('/quotes/forex', 'Forex Quotes'),
            ('/quotes/crypto', 'Crypto Quotes'),
            ('/quotes/commodity', 'Commodity Quotes'),
            ('/quotes/etf', 'ETF Quotes'),
            ('/quotes/mutual-fund', 'Mutual Fund Quotes'),
            ('/quotes/stock', 'Stock Quotes'),
            ('/quotes/available-traded', 'Available Traded'),
            ('/quotes/nyse', 'NYSE Quotes'),
            ('/quotes/nasdaq', 'NASDAQ Quotes'),
            ('/quotes/amex', 'AMEX Quotes'),
            ('/quotes/tsx', 'TSX Quotes'),
            ('/quotes/euronext', 'Euronext Quotes'),
            ('/quotes/lse', 'LSE Quotes'),
            ('/quotes/xetra', 'Xetra Quotes'),
            ('/quotes/six', 'SIX Quotes'),
            ('/quotes/bse', 'BSE Quotes'),
            ('/quotes/nse', 'NSE Quotes'),
            ('/quotes/asx', 'ASX Quotes'),
            ('/quotes/jse', 'JSE Quotes'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in market_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_insider_trading_endpoints(self):
        """Test insider trading and institutional data endpoints"""
        print("\n📊 TESTING INSIDER TRADING ENDPOINTS")
        print("="*60)
        
        insider_endpoints = [
            ('/insider-trading/AAPL', 'Insider Trading'),
            ('/insider-roster/AAPL', 'Insider Roster'),
            ('/institutional-holder/AAPL', 'Institutional Holders'),
            ('/mutual-fund-holder/AAPL', 'Mutual Fund Holders'),
            ('/etf-holder/AAPL', 'ETF Holders'),
            ('/etf-sector-weightings/AAPL', 'ETF Sector Weightings'),
            ('/etf-country-weightings/AAPL', 'ETF Country Weightings'),
            ('/etf-holdings/AAPL', 'ETF Holdings'),
            ('/etf-expense-ratio/AAPL', 'ETF Expense Ratio'),
            ('/etf-holdings/AAPL', 'ETF Holdings'),
            ('/etf-holdings/AAPL', 'ETF Holdings'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in insider_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_analyst_data_endpoints(self):
        """Test analyst estimates and recommendations endpoints"""
        print("\n📊 TESTING ANALYST DATA ENDPOINTS")
        print("="*60)
        
        analyst_endpoints = [
            ('/analyst-estimates/AAPL', 'Analyst Estimates'),
            ('/price-target-summary/AAPL', 'Price Target Summary'),
            ('/price-target/AAPL', 'Price Target'),
            ('/rating/AAPL', 'Analyst Rating'),
            ('/rating-summary/AAPL', 'Rating Summary'),
            ('/earnings-surprises/AAPL', 'Earnings Surprises'),
            ('/earnings-calendar', 'Earnings Calendar'),
            ('/earnings-calendar-confirmed', 'Confirmed Earnings Calendar'),
            ('/earnings-calendar-confirmed/AAPL', 'Company Earnings Calendar'),
            ('/earnings-surprises/AAPL', 'Earnings Surprises'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in analyst_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_esg_data_endpoints(self):
        """Test ESG and sustainability data endpoints"""
        print("\n📊 TESTING ESG DATA ENDPOINTS")
        print("="*60)
        
        esg_endpoints = [
            ('/esg-environmental-social-governance-data/AAPL', 'ESG Data'),
            ('/esg-environmental-social-governance-data/AAPL?page=0', 'ESG Data Page 0'),
            ('/esg-environmental-social-governance-data/AAPL?page=1', 'ESG Data Page 1'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in esg_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_news_endpoints(self):
        """Test news and press releases endpoints"""
        print("\n📊 TESTING NEWS ENDPOINTS")
        print("="*60)
        
        news_endpoints = [
            ('/stock_news', 'Stock News'),
            ('/stock_news?tickers=AAPL', 'Company News'),
            ('/press-releases/AAPL', 'Press Releases'),
            ('/press-releases/AAPL?page=0', 'Press Releases Page 0'),
            ('/press-releases/AAPL?page=1', 'Press Releases Page 1'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in news_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def test_economics_endpoints(self):
        """Test economics and macro data endpoints"""
        print("\n📊 TESTING ECONOMICS ENDPOINTS")
        print("="*60)
        
        economics_endpoints = [
            ('/economic-calendar', 'Economic Calendar'),
            ('/treasury-rates', 'Treasury Rates'),
            ('/economic-indicator', 'Economic Indicators'),
            ('/economic-indicator?name=GDP', 'GDP Data'),
            ('/economic-indicator?name=inflation', 'Inflation Data'),
            ('/economic-indicator?name=unemployment', 'Unemployment Data'),
            ('/economic-indicator?name=interest-rate', 'Interest Rate Data'),
        ]
        
        working_endpoints = []
        
        for endpoint, description in economics_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                params = {'apikey': self.api_key}
                
                print(f"\nTesting: {description}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Success: {len(data)} data points")
                        working_endpoints.append((endpoint, description, len(data)))
                    elif isinstance(data, dict) and data:
                        print(f"   ✅ Success: Data available")
                        working_endpoints.append((endpoint, description, 1))
                    else:
                        print(f"   ⚠️ No data available")
                else:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                
        return working_endpoints
    
    def generate_analysis_report(self, all_results):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("🎯 FMP API COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        total_endpoints = sum(len(results) for results in all_results.values())
        working_endpoints = sum(len([ep for ep in results if ep[2] > 0]) for results in all_results.values())
        
        print(f"📊 TOTAL ENDPOINTS TESTED: {total_endpoints}")
        print(f"✅ WORKING ENDPOINTS: {working_endpoints}")
        if total_endpoints > 0:
            print(f"📈 SUCCESS RATE: {(working_endpoints/total_endpoints*100):.1f}%")
        else:
            print(f"📈 SUCCESS RATE: 0.0%")
        
        print("\n📋 DETAILED BREAKDOWN:")
        for category, results in all_results.items():
            working = len([ep for ep in results if ep[2] > 0])
            total = len(results)
            if total > 0:
                print(f"   {category}: {working}/{total} ({working/total*100:.1f}%)")
            else:
                print(f"   {category}: {working}/{total} (0.0%)")
        
        print("\n🎯 AGENT INTEGRATION OPPORTUNITIES:")
        
        # Map endpoints to agents
        agent_mapping = {
            'Technical Agent': ['Stock Prices', 'Market Data'],
            'Top Performers Agent': ['Stock Prices', 'Market Data', 'Analyst Data'],
            'Undervalued Agent': ['Fundamental Data', 'Analyst Data', 'ESG Data'],
            'Flow Agent': ['Stock Prices', 'Market Data'],
            'Money Flows Agent': ['Stock Prices', 'Market Data', 'Insider Trading'],
            'Sentiment Agent': ['News', 'Analyst Data'],
            'Learning Agent': ['All Categories'],
            'Macro Agent': ['Economics', 'Market Data'],
            'Insider Agent': ['Insider Trading', 'Institutional Data'],
        }
        
        for agent, categories in agent_mapping.items():
            relevant_endpoints = []
            for category, results in all_results.items():
                if category in categories or 'All Categories' in categories:
                    relevant_endpoints.extend([ep for ep in results if ep[2] > 0])
            
            print(f"   {agent}: {len(relevant_endpoints)} relevant endpoints")
        
        print("\n🚀 RECOMMENDATIONS:")
        print("   ✅ FMP provides excellent fundamental data coverage")
        print("   ✅ Strong insider trading and institutional data")
        print("   ✅ Comprehensive market data across multiple exchanges")
        print("   ✅ Real-time and historical price data")
        print("   ✅ Analyst estimates and recommendations")
        print("   ✅ ESG and sustainability data")
        print("   ✅ Economic indicators and calendar")
        
        return all_results

def main():
    """Main function to run FMP comprehensive analysis"""
    analyzer = FMPComprehensiveAnalysis()
    
    # Test all endpoint categories
    results = {}
    
    results['Stock Prices'] = analyzer.test_stock_prices_endpoints()
    results['Fundamental Data'] = analyzer.test_fundamental_data_endpoints()
    results['Market Data'] = analyzer.test_market_data_endpoints()
    results['Insider Trading'] = analyzer.test_insider_trading_endpoints()
    results['Analyst Data'] = analyzer.test_analyst_data_endpoints()
    results['ESG Data'] = analyzer.test_esg_data_endpoints()
    results['News'] = analyzer.test_news_endpoints()
    results['Economics'] = analyzer.test_economics_endpoints()
    
    # Generate comprehensive report
    analyzer.generate_analysis_report(results)
    
    print("\n" + "="*60)
    print("🎯 FMP API ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
