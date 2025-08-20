#!/usr/bin/env python3
"""
Test NewsAPI Integration
Verify the NewsAPI key and functionality
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class NewsAPITester:
    """Test NewsAPI integration"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY', '')
        self.base_url = "https://newsapi.org/v2"
        
    async def test_api_key(self):
        """Test if the API key is valid"""
        print("🔑 Testing NewsAPI Key...")
        
        # Test with a simple request
        url = f"{self.base_url}/top-headlines"
        params = {
            'country': 'us',
            'apiKey': self.api_key,
            'pageSize': 1
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ NewsAPI Key is valid!")
                        print(f"📊 Status: {data.get('status', 'unknown')}")
                        print(f"📰 Total Results: {data.get('totalResults', 0)}")
                        return True
                    elif response.status == 401:
                        print("❌ NewsAPI Key is invalid")
                        return False
                    else:
                        print(f"⚠️ Unexpected status: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Error testing API key: {e}")
                return False
    
    async def test_symbol_search(self, symbol: str = 'AAPL'):
        """Test searching for news about a specific symbol"""
        print(f"\n📰 Testing symbol search for {symbol}...")
        
        url = f"{self.base_url}/everything"
        params = {
            'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        print(f"✅ Found {len(articles)} articles for {symbol}")
                        
                        # Display first few articles
                        for i, article in enumerate(articles[:3]):
                            print(f"\n📄 Article {i+1}:")
                            print(f"   Title: {article.get('title', 'N/A')}")
                            print(f"   Source: {article.get('source', {}).get('name', 'N/A')}")
                            print(f"   Published: {article.get('publishedAt', 'N/A')}")
                            print(f"   URL: {article.get('url', 'N/A')}")
                        
                        return articles
                    else:
                        print(f"❌ Error searching for {symbol}: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Error in symbol search: {e}")
                return []
    
    async def test_rate_limits(self):
        """Test rate limiting information"""
        print(f"\n⏱️ Testing rate limits...")
        
        # Make multiple requests to check rate limits
        url = f"{self.base_url}/top-headlines"
        params = {
            'country': 'us',
            'apiKey': self.api_key,
            'pageSize': 1
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(3):
                try:
                    async with session.get(url, params=params) as response:
                        print(f"   Request {i+1}: Status {response.status}")
                        
                        # Check rate limit headers
                        remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
                        reset = response.headers.get('X-RateLimit-Reset', 'Unknown')
                        
                        if remaining != 'Unknown':
                            print(f"   Remaining requests: {remaining}")
                        if reset != 'Unknown':
                            print(f"   Reset time: {reset}")
                            
                except Exception as e:
                    print(f"   Request {i+1} failed: {e}")
                
                # Small delay between requests
                await asyncio.sleep(1)
    
    async def test_categories(self):
        """Test different news categories"""
        print(f"\n📂 Testing news categories...")
        
        categories = ['business', 'technology', 'general']
        
        async with aiohttp.ClientSession() as session:
            for category in categories:
                url = f"{self.base_url}/top-headlines"
                params = {
                    'country': 'us',
                    'category': category,
                    'apiKey': self.api_key,
                    'pageSize': 3
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            print(f"✅ {category.capitalize()}: {len(articles)} articles")
                        else:
                            print(f"❌ {category.capitalize()}: Error {response.status}")
                except Exception as e:
                    print(f"❌ {category.capitalize()}: {e}")
                
                await asyncio.sleep(1)

async def main():
    """Main test function"""
    print("🚀 NewsAPI Integration Test")
    print("="*50)
    
    tester = NewsAPITester()
    
    # Test API key
    if not await tester.test_api_key():
        print("❌ Cannot proceed without valid API key")
        return
    
    # Test symbol search
    await tester.test_symbol_search('AAPL')
    
    # Test rate limits
    await tester.test_rate_limits()
    
    # Test categories
    await tester.test_categories()
    
    print(f"\n🎉 NewsAPI integration test complete!")
    print(f"✅ Ready to integrate with sentiment analysis system")

if __name__ == "__main__":
    asyncio.run(main())
