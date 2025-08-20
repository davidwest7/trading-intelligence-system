#!/usr/bin/env python3
"""
Simple FRED API Test
Non-async test to avoid hanging issues
"""
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv('env_real_keys.env')

def test_fred_api():
    """Simple test of FRED API"""
    print("🔧 SIMPLE FRED API TEST")
    print("=" * 40)
    
    # Get FRED API key
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        print("❌ FRED API key not found")
        return False
    
    print(f"✅ FRED API key found: {fred_api_key[:10]}...")
    
    # Test URL
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'GDP',
        'api_key': fred_api_key,
        'limit': 3,
        'sort_order': 'desc',
        'file_type': 'json'
    }
    
    try:
        print("Testing FRED API...")
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'observations' in data and data['observations']:
                print("✅ FRED API working!")
                print(f"Observations returned: {len(data['observations'])}")
                
                latest = data['observations'][0]
                print(f"Latest GDP:")
                print(f"  Date: {latest.get('date')}")
                print(f"  Value: {latest.get('value')}")
                
                return True
            else:
                print("❌ No observations in response")
                print(f"Response: {data}")
                return False
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_fred_api()
    
    if success:
        print("\n🎉 FRED API TEST PASSED!")
        print("✅ Ready for macro agent integration")
        print("📈 Alpha Impact: Complete macro agent coverage")
        print("💰 Cost: $0 (free)")
    else:
        print("\n❌ FRED API TEST FAILED")
        print("🔧 Need to fix FRED API integration")
