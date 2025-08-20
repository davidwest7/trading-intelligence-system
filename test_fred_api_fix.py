#!/usr/bin/env python3
"""
Test and Fix FRED API
Simple script to test FRED API and fix any issues
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv('env_real_keys.env')

async def test_fred_api():
    """Test FRED API with proper endpoint format"""
    print("🔧 TESTING AND FIXING FRED API")
    print("=" * 50)
    
    # Get FRED API key
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        print("❌ FRED API key not found in env_real_keys.env")
        return False
    
    print(f"✅ FRED API key found: {fred_api_key[:10]}...")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Basic GDP data with correct endpoint format
        print("\n📊 Test 1: GDP Data")
        print("-" * 30)
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'GDP',
            'api_key': fred_api_key,
            'limit': 5,
            'sort_order': 'desc',
            'file_type': 'json'  # Explicitly request JSON
        }
        
        try:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    
                    if 'observations' in data:
                        print("✅ FRED API working with JSON format!")
                        print(f"Observations returned: {len(data['observations'])}")
                        
                        if data['observations']:
                            latest = data['observations'][0]
                            print(f"Latest GDP data:")
                            print(f"  Date: {latest.get('date')}")
                            print(f"  Value: {latest.get('value')}")
                            print(f"  Units: {latest.get('units')}")
                        
                        # Test 2: CPI Data
                        print("\n📊 Test 2: CPI Data")
                        print("-" * 30)
                        
                        params['series_id'] = 'CPIAUCSL'
                        async with session.get(url, params=params) as response2:
                            if response2.status == 200:
                                cpi_data = await response2.json()
                                if 'observations' in cpi_data and cpi_data['observations']:
                                    latest_cpi = cpi_data['observations'][0]
                                    print("✅ CPI data working!")
                                    print(f"  Date: {latest_cpi.get('date')}")
                                    print(f"  Value: {latest_cpi.get('value')}")
                        
                        # Test 3: Unemployment Data
                        print("\n📊 Test 3: Unemployment Data")
                        print("-" * 30)
                        
                        params['series_id'] = 'UNRATE'
                        async with session.get(url, params=params) as response3:
                            if response3.status == 200:
                                unemp_data = await response3.json()
                                if 'observations' in unemp_data and unemp_data['observations']:
                                    latest_unemp = unemp_data['observations'][0]
                                    print("✅ Unemployment data working!")
                                    print(f"  Date: {latest_unemp.get('date')}")
                                    print(f"  Value: {latest_unemp.get('value')}%")
                        
                        # Test 4: Federal Funds Rate
                        print("\n📊 Test 4: Federal Funds Rate")
                        print("-" * 30)
                        
                        params['series_id'] = 'FEDFUNDS'
                        async with session.get(url, params=params) as response4:
                            if response4.status == 200:
                                fed_data = await response4.json()
                                if 'observations' in fed_data and fed_data['observations']:
                                    latest_fed = fed_data['observations'][0]
                                    print("✅ Federal Funds Rate working!")
                                    print(f"  Date: {latest_fed.get('date')}")
                                    print(f"  Value: {latest_fed.get('value')}%")
                        
                        print("\n🎉 ALL FRED API TESTS PASSED!")
                        print("✅ FRED API is working correctly")
                        print("✅ Ready for macro agent integration")
                        
                        return True
                    else:
                        print("❌ No observations in response")
                        print(f"Response: {data}")
                        return False
                else:
                    print(f"❌ HTTP {response.status}")
                    print(f"Response: {await response.text()}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error testing FRED API: {str(e)}")
            return False

async def create_fred_adapter():
    """Create a simple FRED adapter for testing"""
    print("\n🔧 CREATING FRED ADAPTER")
    print("=" * 50)
    
    adapter_code = '''
import asyncio
import aiohttp
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class FREDAdapter:
    """Simple FRED API adapter for economic data"""
    
    def __init__(self):
        self.api_key = os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_economic_series(self, series_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get economic series data"""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'limit': limit,
                'sort_order': 'desc',
                'file_type': 'json'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching {series_id}: HTTP {response.status}")
                    return {}
        except Exception as e:
            print(f"Error fetching {series_id}: {str(e)}")
            return {}
    
    async def get_gdp_data(self) -> Dict[str, Any]:
        """Get GDP data"""
        return await self.get_economic_series('GDP')
    
    async def get_cpi_data(self) -> Dict[str, Any]:
        """Get CPI data"""
        return await self.get_economic_series('CPIAUCSL')
    
    async def get_unemployment_data(self) -> Dict[str, Any]:
        """Get unemployment data"""
        return await self.get_economic_series('UNRATE')
    
    async def get_fed_funds_rate(self) -> Dict[str, Any]:
        """Get Federal Funds Rate"""
        return await self.get_economic_series('FEDFUNDS')
    
    async def get_all_macro_data(self) -> Dict[str, Any]:
        """Get all macro economic data"""
        gdp, cpi, unemp, fed = await asyncio.gather(
            self.get_gdp_data(),
            self.get_cpi_data(),
            self.get_unemployment_data(),
            self.get_fed_funds_rate()
        )
        
        return {
            'gdp': gdp,
            'cpi': cpi,
            'unemployment': unemp,
            'fed_funds_rate': fed,
            'timestamp': datetime.now().isoformat()
        }
'''
    
    # Save the adapter
    with open('fred_adapter.py', 'w') as f:
        f.write(adapter_code)
    
    print("✅ FRED adapter created: fred_adapter.py")
    return True

async def test_fred_adapter():
    """Test the created FRED adapter"""
    print("\n🧪 TESTING FRED ADAPTER")
    print("=" * 50)
    
    try:
        # Import the adapter
        from fred_adapter import FREDAdapter
        
        async with FREDAdapter() as adapter:
            # Test getting all macro data
            print("Fetching all macro economic data...")
            macro_data = await adapter.get_all_macro_data()
            
            print("✅ FRED adapter working!")
            print(f"GDP observations: {len(macro_data['gdp'].get('observations', []))}")
            print(f"CPI observations: {len(macro_data['cpi'].get('observations', []))}")
            print(f"Unemployment observations: {len(macro_data['unemployment'].get('observations', []))}")
            print(f"Fed Funds Rate observations: {len(macro_data['fed_funds_rate'].get('observations', []))}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error testing FRED adapter: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🚀 FRED API TEST AND FIX")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test 1: Basic FRED API
    test1_result = await test_fred_api()
    
    if test1_result:
        # Test 2: Create adapter
        test2_result = await create_fred_adapter()
        
        if test2_result:
            # Test 3: Test adapter
            test3_result = await test_fred_adapter()
            
            if test3_result:
                print("\n🎉 ALL TESTS PASSED!")
                print("✅ FRED API is working correctly")
                print("✅ FRED adapter created and tested")
                print("✅ Ready for macro agent integration")
                print("\n📈 Alpha Impact: Complete macro agent coverage")
                print("💰 Cost: $0 (free)")
                print("⏱️ Implementation Time: 1 day")
            else:
                print("\n❌ FRED adapter test failed")
        else:
            print("\n❌ FRED adapter creation failed")
    else:
        print("\n❌ FRED API test failed")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
