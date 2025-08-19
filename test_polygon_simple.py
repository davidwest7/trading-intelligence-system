#!/usr/bin/env python3
"""
Simple Polygon.io API Test
Debug what data is actually being returned
"""
import os
import asyncio
import requests
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

async def test_polygon_api():
    """Test basic Polygon.io API calls"""
    print("🔍 **POLYGON.IO API DEBUG TEST**")
    print("=" * 40)
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return
    
    print(f"✅ API Key: {api_key[:10]}...")
    
    # Test basic ticker endpoint
    print("\n📊 Testing ticker endpoint...")
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        'apiKey': api_key,
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {len(data.get('results', []))} tickers")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test snapshot endpoint for AAPL
    print("\n📈 Testing snapshot endpoint for AAPL...")
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/AAPL"
    params = {'apiKey': api_key}
    
    try:
        response = requests.get(url, params=params)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! AAPL data retrieved")
            if 'results' in data:
                print(f"Results keys: {list(data['results'].keys())}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test gainers endpoint
    print("\n🏆 Testing gainers endpoint...")
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers"
    params = {
        'apiKey': api_key,
        'limit': 5
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {len(data.get('results', []))} gainers")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_polygon_api())
