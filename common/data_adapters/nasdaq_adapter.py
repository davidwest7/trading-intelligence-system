"""
Nasdaq Data Link (Quandl) Data Adapter
Provides free access to financial, economic, and alternative data
"""

import asyncio
import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

from .base import BaseDataAdapter

# Load environment variables
load_dotenv('env_real_keys.env')

class NasdaqDataLinkAdapter(BaseDataAdapter):
    """
    Nasdaq Data Link (Quandl) data adapter with comprehensive financial data
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Nasdaq Data Link", config)
        self.api_key = config.get('nasdaq_api_key') or os.getenv('NASDAQ_API_KEY')
        self.base_url = "https://data.nasdaq.com/api/v3"
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache for free tier
        self.rate_limit_delay = 0.1  # 100ms between calls (10 calls per second)
        
        # Free tier limits
        self.free_calls_per_day = 1000
        self.calls_used = 0
        self.last_reset = datetime.now().date()
        
        # Standard headers to avoid Incapsula protection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Popular datasets
        self.datasets = {
            'economic': {
                'FRED_GDP': 'FRED/GDP',
                'FRED_CPI': 'FRED/CPIAUCSL',
                'FRED_UNEMPLOYMENT': 'FRED/UNRATE',
                'FRED_INTEREST_RATE': 'FRED/FEDFUNDS',
                'FRED_CONSUMER_SENTIMENT': 'FRED/UMCSENT'
            },
            'financial': {
                'WIKI_PRICES': 'WIKI/PRICES',  # Historical stock prices
                'OPEC_OIL': 'OPEC/ORB',
                'GOLD_PRICE': 'LBMA/GOLD',
                'SILVER_PRICE': 'LBMA/SILVER'
            },
            'alternative': {
                'ZILLOW_HOME_VALUES': 'ZILLOW/C_SFRCONDO',
                'BITCOIN_PRICE': 'BCHAIN/MKPRU',
                'ETHEREUM_PRICE': 'CRYPTOCURRENCIES/ETHEREUM'
            }
        }
    
    async def connect(self) -> bool:
        """Test connection to Nasdaq Data Link"""
        try:
            # Test with a simple dataset request
            test_url = f"{self.base_url}/datasets/WIKI/PRICES/data.json"
            params = {
                'api_key': self.api_key,
                'limit': 1
            }
            
            response = requests.get(test_url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                self.is_connected = True
                print("✅ Nasdaq Data Link connection successful")
                return True
            else:
                print(f"❌ Nasdaq Data Link connection failed - HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ Nasdaq Data Link connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Nasdaq Data Link"""
        self.is_connected = False
        self.cache.clear()
        return True
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        current_date = datetime.now().date()
        
        # Reset counter if it's a new day
        if current_date != self.last_reset:
            self.calls_used = 0
            self.last_reset = current_date
        
        # Check if we've exceeded daily limit
        if self.calls_used >= self.free_calls_per_day:
            raise Exception(f"Daily API limit exceeded ({self.free_calls_per_day} calls)")
        
        self.calls_used += 1
        await asyncio.sleep(self.rate_limit_delay)
    
    async def get_economic_data(self, indicator: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get economic indicator data"""
        try:
            await self._check_rate_limit()
            
            # Map indicator to dataset
            dataset_map = {
                'gdp': 'FRED/GDP',
                'cpi': 'FRED/CPIAUCSL',
                'unemployment': 'FRED/UNRATE',
                'interest_rate': 'FRED/FEDFUNDS',
                'consumer_sentiment': 'FRED/UMCSENT',
                'inflation': 'FRED/CPIAUCSL',
                'employment': 'FRED/PAYEMS'
            }
            
            dataset = dataset_map.get(indicator.lower())
            if not dataset:
                raise ValueError(f"Unknown economic indicator: {indicator}")
            
            # Build URL
            url = f"{self.base_url}/datasets/{dataset}/data.json"
            
            params = {
                'api_key': self.api_key,
                'limit': 1000
            }
            
            if start_date:
                params['start_date'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['end_date'] = end_date.strftime('%Y-%m-%d')
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['dataset_data']['data'], 
                                columns=data['dataset_data']['column_names'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            else:
                print(f"Error fetching economic data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting economic data: {e}")
            return pd.DataFrame()
    
    async def get_commodity_data(self, commodity: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get commodity price data"""
        try:
            await self._check_rate_limit()
            
            # Map commodity to dataset
            commodity_map = {
                'gold': 'LBMA/GOLD',
                'silver': 'LBMA/SILVER',
                'oil': 'OPEC/ORB',
                'copper': 'LME/PR_CU',
                'aluminum': 'LME/PR_AL'
            }
            
            dataset = commodity_map.get(commodity.lower())
            if not dataset:
                raise ValueError(f"Unknown commodity: {commodity}")
            
            # Build URL
            url = f"{self.base_url}/datasets/{dataset}/data.json"
            
            params = {
                'api_key': self.api_key,
                'limit': 1000
            }
            
            if start_date:
                params['start_date'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['end_date'] = end_date.strftime('%Y-%m-%d')
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['dataset_data']['data'], 
                                columns=data['dataset_data']['column_names'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            else:
                print(f"Error fetching commodity data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting commodity data: {e}")
            return pd.DataFrame()
    
    async def get_crypto_data(self, cryptocurrency: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get cryptocurrency price data"""
        try:
            await self._check_rate_limit()
            
            # Map cryptocurrency to dataset
            crypto_map = {
                'bitcoin': 'BCHAIN/MKPRU',
                'ethereum': 'CRYPTOCURRENCIES/ETHEREUM',
                'litecoin': 'CRYPTOCURRENCIES/LITECOIN',
                'ripple': 'CRYPTOCURRENCIES/RIPPLE'
            }
            
            dataset = crypto_map.get(cryptocurrency.lower())
            if not dataset:
                raise ValueError(f"Unknown cryptocurrency: {cryptocurrency}")
            
            # Build URL
            url = f"{self.base_url}/datasets/{dataset}/data.json"
            
            params = {
                'api_key': self.api_key,
                'limit': 1000
            }
            
            if start_date:
                params['start_date'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['end_date'] = end_date.strftime('%Y-%m-%d')
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['dataset_data']['data'], 
                                columns=data['dataset_data']['column_names'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            else:
                print(f"Error fetching crypto data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting crypto data: {e}")
            return pd.DataFrame()
    
    async def get_alternative_data(self, dataset: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get alternative data (real estate, etc.)"""
        try:
            await self._check_rate_limit()
            
            # Alternative datasets
            alt_datasets = {
                'zillow_home_values': 'ZILLOW/C_SFRCONDO',
                'zillow_rental': 'ZILLOW/M_ZRIMFRR',
                'airbnb_prices': 'AIRBNB/PRICE',
                'uber_rides': 'UBER/UBER_RIDES'
            }
            
            dataset_code = alt_datasets.get(dataset.lower())
            if not dataset_code:
                raise ValueError(f"Unknown alternative dataset: {dataset}")
            
            # Build URL
            url = f"{self.base_url}/datasets/{dataset_code}/data.json"
            
            params = {
                'api_key': self.api_key,
                'limit': 1000
            }
            
            if start_date:
                params['start_date'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['end_date'] = end_date.strftime('%Y-%m-%d')
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['dataset_data']['data'], 
                                columns=data['dataset_data']['column_names'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            else:
                print(f"Error fetching alternative data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting alternative data: {e}")
            return pd.DataFrame()
    
    async def search_datasets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for available datasets"""
        try:
            await self._check_rate_limit()
            
            url = f"{self.base_url}/datasets"
            params = {
                'api_key': self.api_key,
                'query': query,
                'limit': limit
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('datasets', [])
            else:
                print(f"Error searching datasets: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error searching datasets: {e}")
            return []
    
    async def get_dataset_info(self, dataset_code: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        try:
            await self._check_rate_limit()
            
            url = f"{self.base_url}/datasets/{dataset_code}"
            params = {
                'api_key': self.api_key
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting dataset info: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return {}
    
    async def get_ohlcv(self, symbol: str, timeframe: str, since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data (compatibility with base adapter)"""
        try:
            # For Nasdaq Data Link, we'll use WIKI/PRICES for historical stock data
            if symbol.upper() in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']:
                return await self._get_wiki_prices(symbol, since, limit)
            else:
                print(f"Symbol {symbol} not available in WIKI/PRICES dataset")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting OHLCV data: {e}")
            return pd.DataFrame()
    
    async def _get_wiki_prices(self, symbol: str, since: datetime, limit: int) -> pd.DataFrame:
        """Get historical prices from WIKI dataset"""
        try:
            await self._check_rate_limit()
            
            url = f"{self.base_url}/datasets/WIKI/PRICES/data.json"
            params = {
                'api_key': self.api_key,
                'ticker': symbol.upper(),
                'start_date': since.strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'limit': limit
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['dataset_data']['data'], 
                                columns=data['dataset_data']['column_names'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Rename columns to match OHLCV format
                column_mapping = {
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                return df[['open', 'high', 'low', 'close', 'volume']]
            else:
                print(f"Error fetching WIKI prices: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting WIKI prices: {e}")
            return pd.DataFrame()
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote (compatibility with base adapter)"""
        try:
            # Get latest data from WIKI dataset
            df = await self._get_wiki_prices(symbol, datetime.now() - timedelta(days=30), 1)
            
            if not df.empty:
                latest = df.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': latest['close'],
                    'change': 0.0,  # WIKI data doesn't include change
                    'change_percent': 0.0,
                    'volume': latest['volume'],
                    'timestamp': df.index[-1].isoformat()
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting quote: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check adapter health"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "api_calls_used": self.calls_used,
            "api_calls_limit": self.free_calls_per_day,
            "calls_remaining": self.free_calls_per_day - self.calls_used,
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()}
        }
