"""
Alpha Vantage Data Adapter
Provides real market data using Alpha Vantage API
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

class AlphaVantageAdapter(BaseDataAdapter):
    """
    Alpha Vantage data adapter with real market data
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AlphaVantage", config)
        self.api_key = config.get('alpha_vantage_key') or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.rate_limit_delay = 12  # 12 seconds between calls (5 calls per minute free tier)
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
    
    async def connect(self) -> bool:
        """Test connection to Alpha Vantage"""
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
                if 'Global Quote' in data:
                    self.is_connected = True
                    print("✅ Alpha Vantage connection successful")
                    return True
                else:
                    print("❌ Alpha Vantage connection failed - API key may be invalid")
                    return False
            else:
                print(f"❌ Alpha Vantage connection failed - HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Alpha Vantage connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpha Vantage"""
        self.is_connected = False
        self.cache.clear()
        return True
    
    async def get_ohlcv(self, symbol: str, timeframe: str, 
                       since: datetime, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data from Alpha Vantage"""
        
        cache_key = f"{symbol}_{timeframe}_{since.strftime('%Y%m%d')}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Map timeframe to Alpha Vantage function
            function_map = {
                '1d': 'TIME_SERIES_DAILY',
                '1w': 'TIME_SERIES_WEEKLY',
                '1m': 'TIME_SERIES_MONTHLY'
            }
            
            function = function_map.get(timeframe, 'TIME_SERIES_DAILY')
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract time series data
                time_series_key = None
                for key in data.keys():
                    if 'Time Series' in key:
                        time_series_key = key
                        break
                
                if time_series_key and data[time_series_key]:
                    # Convert to DataFrame
                    df_data = []
                    time_series = data[time_series_key]
                    
                    for date, values in time_series.items():
                        if pd.to_datetime(date) >= since:
                            df_data.append({
                                'Date': pd.to_datetime(date),
                                'Open': float(values['1. open']),
                                'High': float(values['2. high']),
                                'Low': float(values['3. low']),
                                'Close': float(values['4. close']),
                                'Volume': int(values['5. volume'])
                            })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df = df.sort_values('Date')
                        
                        # Limit results
                        if len(df) > limit:
                            df = df.tail(limit)
                        
                        # Cache results
                        self.cache[cache_key] = (df, time.time())
                        
                        return df
                    else:
                        print(f"No data found for {symbol} since {since}")
                        return pd.DataFrame()
                else:
                    print(f"No time series data found for {symbol}")
                    return pd.DataFrame()
            else:
                print(f"Error fetching OHLCV for {symbol}: HTTP {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote from Alpha Vantage"""
        
        cache_key = f"quote_{symbol}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < 60:  # 1 minute cache for quotes
                return cached_data
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
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
                    
                    result = {
                        'symbol': symbol,
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%'),
                        'volume': int(quote.get('06. volume', 0)),
                        'open': float(quote.get('02. open', 0)),
                        'high': float(quote.get('03. high', 0)),
                        'low': float(quote.get('04. low', 0)),
                        'previous_close': float(quote.get('08. previous close', 0)),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Cache result
                    self.cache[cache_key] = (result, time.time())
                    
                    return result
                else:
                    print(f"No quote data found for {symbol}")
                    return self._create_empty_quote(symbol)
            else:
                print(f"Error fetching quote for {symbol}: HTTP {response.status_code}")
                return self._create_empty_quote(symbol)
                
        except Exception as e:
            print(f"Error fetching quote for {symbol}: {e}")
            return self._create_empty_quote(symbol)
    
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data"""
        
        cache_key = f"overview_{symbol}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
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
                        'description': data.get('Description', ''),
                        'sector': data.get('Sector'),
                        'industry': data.get('Industry'),
                        'market_cap': data.get('MarketCapitalization'),
                        'pe_ratio': data.get('PERatio'),
                        'dividend_yield': data.get('DividendYield'),
                        'beta': data.get('Beta'),
                        '52_week_high': data.get('52WeekHigh'),
                        '52_week_low': data.get('52WeekLow'),
                        'eps': data.get('EPS'),
                        'revenue': data.get('RevenueTTM'),
                        'profit_margin': data.get('ProfitMargin'),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Cache result
                    self.cache[cache_key] = (overview, time.time())
                    
                    return overview
                else:
                    print(f"No company overview found for {symbol}")
                    return self._create_empty_overview(symbol)
            else:
                print(f"Error fetching company overview for {symbol}: HTTP {response.status_code}")
                return self._create_empty_overview(symbol)
                
        except Exception as e:
            print(f"Error fetching company overview for {symbol}: {e}")
            return self._create_empty_overview(symbol)
    
    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings data"""
        
        cache_key = f"earnings_{symbol}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'quarterlyEarnings' in data:
                    earnings = {
                        'symbol': symbol,
                        'quarterly_earnings': data['quarterlyEarnings'][:4],  # Last 4 quarters
                        'annual_earnings': data.get('annualEarnings', []),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Cache result
                    self.cache[cache_key] = (earnings, time.time())
                    
                    return earnings
                else:
                    print(f"No earnings data found for {symbol}")
                    return self._create_empty_earnings(symbol)
            else:
                print(f"Error fetching earnings for {symbol}: HTTP {response.status_code}")
                return self._create_empty_earnings(symbol)
                
        except Exception as e:
            print(f"Error fetching earnings for {symbol}: {e}")
            return self._create_empty_earnings(symbol)
    
    def _create_empty_quote(self, symbol: str) -> Dict[str, Any]:
        """Create empty quote data"""
        return {
            'symbol': symbol,
            'price': 0.0,
            'change': 0.0,
            'change_percent': '0%',
            'volume': 0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'previous_close': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_empty_overview(self, symbol: str) -> Dict[str, Any]:
        """Create empty company overview"""
        return {
            'symbol': symbol,
            'name': 'Unknown',
            'description': '',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': '0',
            'pe_ratio': '0',
            'dividend_yield': '0',
            'beta': '0',
            '52_week_high': '0',
            '52_week_low': '0',
            'eps': '0',
            'revenue': '0',
            'profit_margin': '0',
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_empty_earnings(self, symbol: str) -> Dict[str, Any]:
        """Create empty earnings data"""
        return {
            'symbol': symbol,
            'quarterly_earnings': [],
            'annual_earnings': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check adapter health"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "api_key_configured": bool(self.api_key),
            "cache_size": len(self.cache),
            "rate_limit_delay": self.rate_limit_delay
        }
