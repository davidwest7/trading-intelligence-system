#!/usr/bin/env python3
"""
Nasdaq Data Link API Debug
Test connectivity and find working endpoints
"""

import requests
import json
from datetime import datetime

class NasdaqDataLinkDebug:
    """Debug Nasdaq Data Link API connectivity"""
    
    def __init__(self):
        self.api_key = 'fT7ekyy5zz_WJwz3-S9g'
        self.base_url = 'https://data.nasdaq.com/api/v3'
        
    def test_connection(self):
        """Test basic connection to Nasdaq Data Link"""
        print("🔍 TESTING NASDAQ DATA LINK CONNECTION")
        print("="*60)
        
        # Test basic connection
        try:
            url = f"{self.base_url}/databases.json"
            params = {'api_key': self.api_key}
            
            print(f"Testing URL: {url}")
            print(f"API Key: {self.api_key[:10]}...")
            
            response = requests.get(url, params=params)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connection successful!")
                print(f"📊 Response length: {len(str(data))} characters")
                if 'databases' in data:
                    print(f"📊 Available databases: {len(data['databases'])}")
                    for db in data['databases'][:3]:  # Show first 3
                        print(f"   - {db.get('name', 'Unknown')}: {db.get('description', 'No description')[:50]}...")
            else:
                print(f"❌ Connection failed: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    def test_popular_datasets(self):
        """Test popular datasets"""
        print("\n🔍 TESTING POPULAR DATASETS")
        print("="*60)
        
        # Test popular free datasets
        popular_datasets = [
            'FRED/GDP',
            'FRED/UNRATE', 
            'FRED/CPIAUCSL',
            'FRED/FEDFUNDS',
            'OPEC/ORB',
            'LBMA/GOLD',
            'CURRFX/USDEUR',
            'CBOE/VIX',
            'MULTPL/SP500_PE_RATIO_MONTH',
            'MULTPL/SP500_DIV_YIELD_MONTH'
        ]
        
        for dataset in popular_datasets:
            try:
                url = f"{self.base_url}/datasets/{dataset}/data.json"
                params = {
                    'api_key': self.api_key,
                    'limit': 5,
                    'order': 'desc'
                }
                
                print(f"\nTesting dataset: {dataset}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if 'dataset_data' in data and 'data' in data['dataset_data']:
                        data_points = len(data['dataset_data']['data'])
                        print(f"   ✅ Success: {data_points} data points")
                        if data_points > 0:
                            latest = data['dataset_data']['data'][0]
                            print(f"   📊 Latest data: {latest}")
                    else:
                        print(f"   ⚠️ No data available")
                elif response.status_code == 403:
                    print(f"   ❌ Forbidden - API key may need activation")
                elif response.status_code == 404:
                    print(f"   ❌ Dataset not found")
                else:
                    print(f"   ❌ Error: {response.text[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    def test_dataset_search(self):
        """Test dataset search functionality"""
        print("\n🔍 TESTING DATASET SEARCH")
        print("="*60)
        
        search_terms = ['GDP', 'VIX', 'GOLD', 'OIL', 'AAPL']
        
        for term in search_terms:
            try:
                url = f"{self.base_url}/datasets.json"
                params = {
                    'api_key': self.api_key,
                    'query': term,
                    'limit': 5
                }
                
                print(f"\nSearching for: {term}")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if 'datasets' in data:
                        datasets = data['datasets']
                        print(f"   ✅ Found {len(datasets)} datasets")
                        for ds in datasets[:2]:  # Show first 2
                            print(f"      - {ds.get('dataset_code', 'Unknown')}: {ds.get('name', 'No name')[:50]}...")
                    else:
                        print(f"   ⚠️ No datasets found")
                else:
                    print(f"   ❌ Error: {response.text[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    def test_api_limits(self):
        """Test API limits and rate limiting"""
        print("\n🔍 TESTING API LIMITS")
        print("="*60)
        
        try:
            # Test multiple requests to check rate limiting
            for i in range(3):
                url = f"{self.base_url}/databases.json"
                params = {'api_key': self.api_key}
                
                print(f"Request {i+1}:")
                response = requests.get(url, params=params)
                
                print(f"   Status Code: {response.status_code}")
                
                # Check for rate limit headers
                if 'x-ratelimit-remaining' in response.headers:
                    remaining = response.headers['x-ratelimit-remaining']
                    print(f"   📊 Remaining requests: {remaining}")
                
                if response.status_code == 429:
                    print(f"   ⚠️ Rate limited!")
                    break
                    
        except Exception as e:
            print(f"❌ Rate limit test error: {e}")
    
    def test_free_datasets(self):
        """Test specifically free datasets"""
        print("\n🔍 TESTING FREE DATASETS")
        print("="*60)
        
        # FRED datasets are typically free
        fred_datasets = [
            'FRED/GDP',
            'FRED/UNRATE',
            'FRED/CPIAUCSL',
            'FRED/FEDFUNDS',
            'FRED/PAYEMS',
            'FRED/RSAFS',
            'FRED/INDPRO',
            'FRED/HOUST',
            'FRED/UMCSENT'
        ]
        
        working_datasets = []
        
        for dataset in fred_datasets:
            try:
                url = f"{self.base_url}/datasets/{dataset}/data.json"
                params = {
                    'api_key': self.api_key,
                    'limit': 1,
                    'order': 'desc'
                }
                
                print(f"Testing {dataset}...")
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'dataset_data' in data and 'data' in data['dataset_data']:
                        working_datasets.append(dataset)
                        print(f"   ✅ Working")
                    else:
                        print(f"   ⚠️ No data")
                else:
                    print(f"   ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n📊 WORKING DATASETS: {len(working_datasets)}")
        for dataset in working_datasets:
            print(f"   ✅ {dataset}")

def main():
    """Main function to run Nasdaq Data Link debug"""
    debugger = NasdaqDataLinkDebug()
    
    # Run all debug tests
    debugger.test_connection()
    debugger.test_popular_datasets()
    debugger.test_dataset_search()
    debugger.test_api_limits()
    debugger.test_free_datasets()
    
    print("\n" + "="*60)
    print("🎯 NASDAQ DATA LINK DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
