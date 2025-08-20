"""
FRED API Data Adapter
Provides economic indicators data for Macro Agent
"""
import asyncio
import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd

# Add current directory to path
sys.path.append('.')
from common.models import BaseDataAdapter

class FREDAdapter(BaseDataAdapter):
    """FRED API data adapter for economic indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FRED", config)
        self.api_key = config.get('fred_api_key') or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache (economic data changes slowly)
        self.rate_limit_delay = 0.1  # 100ms between calls
        
        if not self.api_key:
            raise ValueError("FRED API key is required")
    
    async def connect(self) -> bool:
        """Connect to FRED API"""
        try:
            # Test connection with a simple request
            test_url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'GDP',
                'api_key': self.api_key,
                'limit': 1,
                'sort_order': 'desc',
                'file_type': 'json'  # Explicitly request JSON
            }
            
            response = requests.get(test_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data:
                    self.is_connected = True
                    return True
            
            return False
        except Exception as e:
            print(f"FRED connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from FRED API"""
        self.is_connected = False
        self.cache.clear()
        return True
    
    async def get_economic_series(self, series_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get economic series data"""
        cache_key = f"{series_id}_{limit}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'limit': limit,
                'sort_order': 'desc',
                'file_type': 'json'  # Explicitly request JSON
            }
            
            # Use requests in executor to avoid blocking
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=10))
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the result
                self.cache[cache_key] = (data, time.time())
                
                return data
            else:
                print(f"Error fetching {series_id}: HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error fetching {series_id}: {str(e)}")
            return {}
    
    async def get_gdp_data(self, limit: int = 50) -> Dict[str, Any]:
        """Get GDP data"""
        return await self.get_economic_series('GDP', limit)
    
    async def get_cpi_data(self, limit: int = 50) -> Dict[str, Any]:
        """Get CPI data"""
        return await self.get_economic_series('CPIAUCSL', limit)
    
    async def get_unemployment_data(self, limit: int = 50) -> Dict[str, Any]:
        """Get unemployment data"""
        return await self.get_economic_series('UNRATE', limit)
    
    async def get_fed_funds_rate(self, limit: int = 50) -> Dict[str, Any]:
        """Get Federal Funds Rate"""
        return await self.get_economic_series('FEDFUNDS', limit)
    
    async def get_inflation_data(self, limit: int = 50) -> Dict[str, Any]:
        """Get inflation data (PCE)"""
        return await self.get_economic_series('PCE', limit)
    
    async def get_consumer_confidence(self, limit: int = 50) -> Dict[str, Any]:
        """Get consumer confidence data"""
        return await self.get_economic_series('UMCSENT', limit)
    
    async def get_all_macro_data(self) -> Dict[str, Any]:
        """Get all macro economic data"""
        try:
            # Fetch all economic indicators concurrently
            gdp, cpi, unemp, fed, inflation, confidence = await asyncio.gather(
                self.get_gdp_data(),
                self.get_cpi_data(),
                self.get_unemployment_data(),
                self.get_fed_funds_rate(),
                self.get_inflation_data(),
                self.get_consumer_confidence()
            )
            
            return {
                'gdp': gdp,
                'cpi': cpi,
                'unemployment': unemp,
                'fed_funds_rate': fed,
                'inflation': inflation,
                'consumer_confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'FRED API'
            }
        except Exception as e:
            print(f"Error fetching all macro data: {str(e)}")
            return {}
    
    async def analyze_macro_environment(self) -> Dict[str, Any]:
        """Analyze macro environment using FRED data"""
        try:
            macro_data = await self.get_all_macro_data()
            
            if not macro_data:
                return self._create_empty_macro_analysis()
            
            # Extract latest values
            latest_values = {}
            for indicator, data in macro_data.items():
                if indicator != 'timestamp' and indicator != 'data_source':
                    if 'observations' in data and data['observations']:
                        latest_values[indicator] = {
                            'date': data['observations'][0].get('date'),
                            'value': data['observations'][0].get('value'),
                            'units': data['observations'][0].get('units', '')
                        }
            
            # Calculate macro signals
            macro_signals = self._calculate_macro_signals(latest_values)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'macro_data': macro_data,
                'latest_values': latest_values,
                'macro_signals': macro_signals,
                'data_source': 'FRED API'
            }
            
        except Exception as e:
            print(f"Error analyzing macro environment: {str(e)}")
            return self._create_empty_macro_analysis()
    
    def _calculate_macro_signals(self, latest_values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate macro economic signals"""
        signals = {
            'gdp_growth': 'neutral',
            'inflation_trend': 'neutral',
            'employment_health': 'neutral',
            'monetary_policy': 'neutral',
            'overall_macro': 'neutral'
        }
        
        try:
            # GDP Analysis
            if 'gdp' in latest_values:
                gdp_value = float(latest_values['gdp']['value'])
                if gdp_value > 20000:  # Strong GDP
                    signals['gdp_growth'] = 'bullish'
                elif gdp_value < 15000:  # Weak GDP
                    signals['gdp_growth'] = 'bearish'
            
            # Inflation Analysis
            if 'cpi' in latest_values:
                cpi_value = float(latest_values['cpi']['value'])
                if cpi_value > 300:  # High inflation
                    signals['inflation_trend'] = 'bearish'
                elif cpi_value < 250:  # Low inflation
                    signals['inflation_trend'] = 'bullish'
            
            # Employment Analysis
            if 'unemployment' in latest_values:
                unemp_value = float(latest_values['unemployment']['value'])
                if unemp_value < 4.0:  # Low unemployment
                    signals['employment_health'] = 'bullish'
                elif unemp_value > 6.0:  # High unemployment
                    signals['employment_health'] = 'bearish'
            
            # Monetary Policy Analysis
            if 'fed_funds_rate' in latest_values:
                fed_value = float(latest_values['fed_funds_rate']['value'])
                if fed_value < 2.0:  # Low rates (accommodative)
                    signals['monetary_policy'] = 'bullish'
                elif fed_value > 5.0:  # High rates (restrictive)
                    signals['monetary_policy'] = 'bearish'
            
            # Overall Macro Assessment
            bullish_count = sum(1 for signal in signals.values() if signal == 'bullish')
            bearish_count = sum(1 for signal in signals.values() if signal == 'bearish')
            
            if bullish_count > bearish_count:
                signals['overall_macro'] = 'bullish'
            elif bearish_count > bullish_count:
                signals['overall_macro'] = 'bearish'
            else:
                signals['overall_macro'] = 'neutral'
                
        except Exception as e:
            print(f"Error calculating macro signals: {str(e)}")
        
        return signals
    
    def _create_empty_macro_analysis(self) -> Dict[str, Any]:
        """Create empty macro analysis when data is unavailable"""
        return {
            'timestamp': datetime.now().isoformat(),
            'macro_data': {},
            'latest_values': {},
            'macro_signals': {
                'gdp_growth': 'neutral',
                'inflation_trend': 'neutral',
                'employment_health': 'neutral',
                'monetary_policy': 'neutral',
                'overall_macro': 'neutral'
            },
            'data_source': 'FRED API (no data)'
        }
