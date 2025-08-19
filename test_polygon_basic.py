#!/usr/bin/env python3
"""
Test basic Polygon.io API endpoints to check API key and access
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

def test_polygon_api():
    """Test basic Polygon.io API endpoints"""
    api_key = os.getenv('POLYGON_API_KEY')
    base_url = "https://api.polygon.io"
    
    print(f"🔑 Testing Polygon.io API Key: {api_key[:10]}...")
    print(f"🌐 Base URL: {base_url}")
    print("=" * 60)
    
    # Test 1: Check API key with reference data
    print(f"\n📊 Test 1: Reference Data (Tickers)")
    url = f"{base_url}/v3/reference/tickers"
    params = {'apiKey': api_key, 'limit': 1}
    
    response = requests.get(url, params=params)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success: {len(data.get('results', []))} results")
        if data.get('results'):
            print(f"📝 Sample ticker: {data['results'][0]}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test 2: Check current quote
    print(f"\n📊 Test 2: Current Quote (AAPL)")
    url = f"{base_url}/v2/snapshot/locale/us/markets/stocks/tickers/AAPL"
    params = {'apiKey': api_key}
    
    response = requests.get(url, params=params)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success: Got AAPL snapshot")
        print(f"📝 Data keys: {list(data.keys())}")
        if 'ticker' in data:
            ticker_data = data['ticker']
            print(f"📈 Last trade: {ticker_data.get('lastTrade', {})}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test 3: Try different aggregates endpoint
    print(f"\n📊 Test 3: Previous Day Bar (AAPL)")
    url = f"{base_url}/v2/aggs/ticker/AAPL/prev"
    params = {'apiKey': api_key, 'adjusted': 'true'}
    
    response = requests.get(url, params=params)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success: Got previous day data")
        print(f"📝 Results: {data.get('results', [])}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test 4: Try grouped daily bars
    print(f"\n📊 Test 4: Grouped Daily Bars")
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"{base_url}/v2/aggs/grouped/locale/us/market/stocks/{yesterday}"
    params = {'apiKey': api_key, 'adjusted': 'true'}
    
    response = requests.get(url, params=params)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success: Got grouped daily data")
        print(f"📝 Results count: {len(data.get('results', []))}")
        if data.get('results'):
            sample = data['results'][0]
            print(f"📈 Sample: {sample}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test 5: Simple range query with known good format
    print(f"\n📊 Test 5: Simple Range Query (AAPL 1 day)")
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    url = f"{base_url}/v2/aggs/ticker/AAPL/range/1/day/{from_date}/{to_date}"
    params = {'apiKey': api_key, 'adjusted': 'true', 'sort': 'asc'}
    
    response = requests.get(url, params=params)
    print(f"📡 Status: {response.status_code}")
    print(f"🔗 URL: {response.url}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success: Got range data")
        print(f"📝 Results count: {len(data.get('results', []))}")
        print(f"📊 Data keys: {list(data.keys())}")
        if data.get('results'):
            print(f"📈 First result: {data['results'][0]}")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_polygon_api()
