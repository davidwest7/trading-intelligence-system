#!/usr/bin/env python3
"""
Finnhub API Integration Test
Quick test to verify API key and basic functionality
"""
import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

class FinnhubAPITest:
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.base_url = "https://finnhub.io/api/v1"
        
    async def test_api_key(self) -> Dict[str, Any]:
        """Test if API key is valid"""
        if not self.api_key:
            return {
                'status': 'ERROR',
                'message': 'No FINNHUB_API_KEY found in environment variables'
            }
        
        # Test with a simple quote request
        url = f"{self.base_url}/quote"
        params = {
            'symbol': 'AAPL',
            'token': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'SUCCESS',
                            'message': 'API key is valid',
                            'data': data
                        }
                    elif response.status == 401:
                        return {
                            'status': 'ERROR',
                            'message': 'Invalid API key'
                        }
                    else:
                        return {
                            'status': 'ERROR',
                            'message': f'API request failed: {response.status}'
                        }
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Connection error: {str(e)}'
            }
    
    async def test_company_news(self, symbol: str = 'AAPL') -> Dict[str, Any]:
        """Test company news endpoint"""
        if not self.api_key:
            return {'status': 'ERROR', 'message': 'No API key'}
        
        url = f"{self.base_url}/company-news"
        params = {
            'symbol': symbol,
            'from': '2025-01-01',
            'to': datetime.now().strftime('%Y-%m-%d'),
            'token': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'SUCCESS',
                            'message': f'Found {len(data)} news articles for {symbol}',
                            'data': data[:3]  # Show first 3 articles
                        }
                    else:
                        return {
                            'status': 'ERROR',
                            'message': f'News request failed: {response.status}'
                        }
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'News request error: {str(e)}'
            }
    
    async def test_financial_statements(self, symbol: str = 'AAPL') -> Dict[str, Any]:
        """Test financial statements endpoint"""
        if not self.api_key:
            return {'status': 'ERROR', 'message': 'No API key'}
        
        url = f"{self.base_url}/stock/financials-reported"
        params = {
            'symbol': symbol,
            'token': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'SUCCESS',
                            'message': f'Financial data available for {symbol}',
                            'data': data
                        }
                    else:
                        return {
                            'status': 'ERROR',
                            'message': f'Financials request failed: {response.status}'
                        }
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Financials request error: {str(e)}'
            }

async def main():
    """Run Finnhub API tests"""
    print("🚀 Finnhub API Integration Test")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('FINNHUB_API_KEY', '')
    if not api_key:
        print("❌ No FINNHUB_API_KEY found in environment variables")
        print("\n📝 To add your API key:")
        print("1. Get your API key from https://finnhub.io/")
        print("2. Add to env_real_keys.env:")
        print("   FINNHUB_API_KEY=your_api_key_here")
        return
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Run tests
    test = FinnhubAPITest()
    
    print("\n🔑 Testing API Key...")
    key_test = await test.test_api_key()
    print(f"   Status: {key_test['status']}")
    print(f"   Message: {key_test['message']}")
    
    if key_test['status'] == 'SUCCESS':
        print("\n📰 Testing Company News...")
        news_test = await test.test_company_news()
        print(f"   Status: {news_test['status']}")
        print(f"   Message: {news_test['message']}")
        
        print("\n📊 Testing Financial Statements...")
        financial_test = await test.test_financial_statements()
        print(f"   Status: {financial_test['status']}")
        print(f"   Message: {financial_test['message']}")
        
        print("\n🎉 Finnhub API is ready for integration!")
    else:
        print("\n❌ Please check your API key and try again")

if __name__ == "__main__":
    asyncio.run(main())
