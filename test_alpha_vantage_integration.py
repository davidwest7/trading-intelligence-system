#!/usr/bin/env python3
"""
Alpha Vantage API Integration Test
Tests the Alpha Vantage API key and retrieves real market data
"""

import os
import asyncio
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv('env_real_keys.env')

class AlphaVantageTester:
    """Test Alpha Vantage API integration"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.results = {}
        
    async def test_api_key(self):
        """Test if the API key is valid"""
        print("🔑 Testing Alpha Vantage API Key...")
        
        try:
            # Test with a simple quote request
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Error Message' in data:
                    print(f"❌ API Key Error: {data['Error Message']}")
                    self.results['api_key'] = {'status': 'error', 'error': data['Error Message']}
                elif 'Note' in data:
                    print(f"⚠️  Rate Limit: {data['Note']}")
                    self.results['api_key'] = {'status': 'rate_limited', 'note': data['Note']}
                else:
                    print("✅ Alpha Vantage API Key is valid!")
                    self.results['api_key'] = {'status': 'success', 'data': data}
            else:
                print(f"❌ API Error: {response.status_code}")
                self.results['api_key'] = {'status': 'error', 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ API Key test failed: {e}")
            self.results['api_key'] = {'status': 'error', 'error': str(e)}
    
    async def get_real_time_quotes(self, symbols: list):
        """Get real-time quotes for multiple symbols"""
        print(f"\n📈 Getting Real-Time Quotes for: {', '.join(symbols)}")
        
        quotes = []
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        quotes.append({
                            'symbol': symbol,
                            'price': float(quote.get('05. price', 0)),
                            'change': float(quote.get('09. change', 0)),
                            'change_percent': quote.get('10. change percent', '0%'),
                            'volume': int(quote.get('06. volume', 0)),
                            'market_cap': quote.get('07. market cap', 'N/A'),
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"✅ {symbol}: ${quote.get('05. price', 'N/A')} ({quote.get('10. change percent', 'N/A')})")
                    else:
                        print(f"❌ No data for {symbol}")
                        
                # Rate limiting - Alpha Vantage free tier: 5 calls per minute
                await asyncio.sleep(12)  # Wait 12 seconds between calls
                
            except Exception as e:
                print(f"❌ Error getting quote for {symbol}: {e}")
        
        self.results['quotes'] = quotes
        return quotes
    
    async def get_historical_data(self, symbol: str, interval: str = 'daily'):
        """Get historical OHLCV data"""
        print(f"\n📊 Getting Historical Data for {symbol} ({interval})")
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY' if interval == 'daily' else 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            if interval != 'daily':
                params['interval'] = '60min'
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data or 'Time Series (60min)' in data:
                    time_series_key = 'Time Series (Daily)' if interval == 'daily' else 'Time Series (60min)'
                    time_series = data[time_series_key]
                    
                    # Convert to DataFrame
                    df_data = []
                    for date, values in list(time_series.items())[:30]:  # Last 30 days
                        df_data.append({
                            'date': date,
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': int(values['5. volume'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    print(f"✅ Retrieved {len(df)} data points for {symbol}")
                    print(f"   Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
                    print(f"   Latest Close: ${df['close'].iloc[-1]:.2f}")
                    print(f"   Volume: {df['volume'].iloc[-1]:,}")
                    
                    self.results['historical_data'] = {
                        'symbol': symbol,
                        'data_points': len(df),
                        'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
                        'latest_close': df['close'].iloc[-1],
                        'latest_volume': df['volume'].iloc[-1]
                    }
                    
                    return df
                else:
                    print(f"❌ No historical data for {symbol}")
                    return None
                    
            else:
                print(f"❌ Error getting historical data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_company_overview(self, symbol: str):
        """Get company fundamental data"""
        print(f"\n🏢 Getting Company Overview for {symbol}")
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Symbol' in data:
                    overview = {
                        'symbol': data.get('Symbol'),
                        'name': data.get('Name'),
                        'description': data.get('Description', '')[:200] + '...',
                        'sector': data.get('Sector'),
                        'industry': data.get('Industry'),
                        'market_cap': data.get('MarketCapitalization'),
                        'pe_ratio': data.get('PERatio'),
                        'dividend_yield': data.get('DividendYield'),
                        'beta': data.get('Beta'),
                        '52_week_high': data.get('52WeekHigh'),
                        '52_week_low': data.get('52WeekLow')
                    }
                    
                    print(f"✅ Company: {data.get('Name', 'N/A')}")
                    print(f"   Sector: {data.get('Sector', 'N/A')}")
                    print(f"   Market Cap: ${data.get('MarketCapitalization', 'N/A')}")
                    print(f"   P/E Ratio: {data.get('PERatio', 'N/A')}")
                    print(f"   Dividend Yield: {data.get('DividendYield', 'N/A')}")
                    
                    self.results['company_overview'] = overview
                    return overview
                else:
                    print(f"❌ No company overview for {symbol}")
                    return None
                    
            else:
                print(f"❌ Error getting company overview: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting company overview for {symbol}: {e}")
            return None
    
    async def run_all_tests(self):
        """Run all Alpha Vantage tests"""
        print("🚀 Starting Alpha Vantage API Integration Tests...")
        print("=" * 60)
        
        # Test API key
        await self.test_api_key()
        
        # Get real-time quotes
        symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        quotes = await self.get_real_time_quotes(symbols)
        
        # Get historical data for one symbol
        if quotes:
            historical_data = await self.get_historical_data('AAPL', 'daily')
        
        # Get company overview
        company_overview = await self.get_company_overview('AAPL')
        
        print("\n" + "=" * 60)
        print("📊 Alpha Vantage Test Results Summary:")
        print("=" * 60)
        
        # Display results
        if 'api_key' in self.results:
            status = "✅ SUCCESS" if self.results['api_key']['status'] == 'success' else "❌ FAILED"
            print(f"API Key: {status}")
        
        if 'quotes' in self.results:
            print(f"Real-time Quotes: ✅ {len(self.results['quotes'])} symbols")
        
        if 'historical_data' in self.results:
            print(f"Historical Data: ✅ {self.results['historical_data']['data_points']} data points")
        
        if 'company_overview' in self.results:
            print(f"Company Overview: ✅ {self.results['company_overview']['name']}")
        
        # Save results
        with open('alpha_vantage_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: alpha_vantage_test_results.json")
        
        return self.results

async def main():
    """Main test function"""
    tester = AlphaVantageTester()
    results = await tester.run_all_tests()
    
    # Summary
    successful_tests = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'success')
    total_tests = len(results)
    
    print(f"\n🎯 SUMMARY: {successful_tests}/{total_tests} tests successful")
    
    if successful_tests == total_tests:
        print("🎉 Alpha Vantage API integration successful!")
        print("✅ Ready for real market data integration")
    else:
        print("⚠️  Some tests need attention. Check the results above.")

if __name__ == "__main__":
    asyncio.run(main())
