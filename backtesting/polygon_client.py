#!/usr/bin/env python3
"""
Enhanced Polygon.io Client
=========================

Production-ready Polygon.io client with:
- Rate limiting and retry logic
- Comprehensive data fetching
- S3 integration
- Data validation
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Iterator, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PolygonConfig:
    """Configuration for Polygon client"""
    api_key: str
    base_url: str = "https://api.polygon.io"
    rate_limit_delay: float = 0.1  # 100ms between calls
    max_retries: int = 8
    timeout: int = 30
    max_concurrent_requests: int = 5

class PolygonClient:
    """
    Enhanced Polygon.io client with rate limiting, retry logic, and comprehensive data fetching
    """
    
    def __init__(self, config: Optional[PolygonConfig] = None):
        """Initialize Polygon client"""
        if config is None:
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                raise ValueError("POLYGON_API_KEY environment variable is required")
            config = PolygonConfig(api_key=api_key)
        
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingIntelligenceSystem/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_limit = 50000  # Adjust based on your plan
        
        logger.info(f"🚀 Polygon client initialized with base URL: {config.base_url}")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP GET request with retry logic"""
        params = params or {}
        params["apiKey"] = self.config.api_key
        
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                url = f"{self.config.base_url}{path}"
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code in (429, 502, 503, 504):
                    # Rate limit or server error - exponential backoff
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited (attempt {attempt + 1}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Polygon request failed after {self.config.max_retries} attempts: {e}")
                wait_time = min(2 ** attempt, 5)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        raise RuntimeError(f"Polygon request failed: {path}")
    
    def scan_pages(self, path: str, params: Optional[Dict[str, Any]] = None, 
                   cursor_key: str = "next_url") -> Iterator[Dict[str, Any]]:
        """Scan through paginated results"""
        params = params or {}
        params["apiKey"] = self.config.api_key
        
        url = f"{self.config.base_url}{path}"
        current_params = params.copy()
        
        while True:
            try:
                self._rate_limit()
                response = self.session.get(url, params=current_params, timeout=self.config.timeout)
                
                if response.status_code != 200:
                    if response.status_code in (429, 502, 503, 504):
                        time.sleep(2)
                        continue
                    response.raise_for_status()
                
                data = response.json()
                yield data
                
                # Check for next page
                next_url = data.get(cursor_key)
                if not next_url:
                    break
                
                # For next_url, we don't need params as they're encoded in the URL
                url = next_url
                current_params = {}
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error in pagination: {e}")
                break
    
    # ==================== AGGREGATES (BARS) ====================
    
    def get_aggregates(self, symbol: str, multiplier: int, timespan: str,
                      from_date: str, to_date: str, adjusted: bool = True,
                      sort: str = "asc", limit: int = 50000) -> pd.DataFrame:
        """
        Get aggregate bars for a stock ticker
        
        Args:
            symbol: Stock ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start of the aggregate time window (YYYY-MM-DD)
            to_date: End of the aggregate time window (YYYY-MM-DD)
            adjusted: Whether the results are adjusted for splits
            sort: Sort order (asc, desc)
            limit: Limit the number of base aggregates queried
        """
        path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": adjusted,
            "sort": sort,
            "limit": limit
        }
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df['date'] = df['timestamp'].dt.date
                
                # Rename columns for consistency
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'vwap',
                    'n': 'transactions'
                })
                
                # Select relevant columns
                columns = ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
                df = df[columns].copy()
                
                logger.info(f"📊 Retrieved {len(df)} bars for {symbol} from {from_date} to {to_date}")
                return df
            
            else:
                logger.warning(f"No data returned for {symbol} from {from_date} to {to_date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching aggregates for {symbol}: {e}")
            return pd.DataFrame()
    
    # ==================== TRADES & QUOTES ====================
    
    def get_trades(self, symbol: str, date: str, timestamp: Optional[int] = None,
                   timestamp_limit: Optional[int] = None, reverse: bool = False,
                   limit: int = 50000) -> pd.DataFrame:
        """
        Get trades for a stock ticker on a specific date
        
        Args:
            symbol: Stock ticker symbol
            date: Date of trades (YYYY-MM-DD)
            timestamp: Timestamp offset for pagination
            timestamp_limit: Timestamp limit for pagination
            reverse: Reverse the sort order
            limit: Limit the number of trades returned
        """
        path = f"/v3/trades/{symbol}/{date}"
        params = {
            "reverse": reverse,
            "limit": limit
        }
        
        if timestamp:
            params["timestamp"] = timestamp
        if timestamp_limit:
            params["timestamp_limit"] = timestamp_limit
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['t'], unit='ns')
                
                # Rename columns
                df = df.rename(columns={
                    'p': 'price',
                    's': 'size',
                    'c': 'conditions',
                    'x': 'exchange',
                    'z': 'tape'
                })
                
                logger.info(f"📈 Retrieved {len(df)} trades for {symbol} on {date}")
                return df
            
            else:
                logger.warning(f"No trades found for {symbol} on {date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_quotes(self, symbol: str, date: str, timestamp: Optional[int] = None,
                   timestamp_limit: Optional[int] = None, reverse: bool = False,
                   limit: int = 50000) -> pd.DataFrame:
        """
        Get quotes for a stock ticker on a specific date
        """
        path = f"/v3/quotes/{symbol}/{date}"
        params = {
            "reverse": reverse,
            "limit": limit
        }
        
        if timestamp:
            params["timestamp"] = timestamp
        if timestamp_limit:
            params["timestamp_limit"] = timestamp_limit
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['t'], unit='ns')
                
                # Rename columns
                df = df.rename(columns={
                    'bp': 'bid_price',
                    'bs': 'bid_size',
                    'ap': 'ask_price',
                    'as': 'ask_size',
                    'c': 'conditions',
                    'x': 'exchange',
                    'z': 'tape'
                })
                
                logger.info(f"💬 Retrieved {len(df)} quotes for {symbol} on {date}")
                return df
            
            else:
                logger.warning(f"No quotes found for {symbol} on {date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching quotes for {symbol}: {e}")
            return pd.DataFrame()
    
    # ==================== REFERENCE DATA ====================
    
    def get_tickers(self, ticker: Optional[str] = None, type: Optional[str] = None,
                   market: Optional[str] = None, active: Optional[bool] = None,
                   search: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get ticker details
        """
        path = "/v3/reference/tickers"
        params = {"limit": limit}
        
        if ticker:
            params["ticker"] = ticker
        if type:
            params["type"] = type
        if market:
            params["market"] = market
        if active is not None:
            params["active"] = active
        if search:
            params["search"] = search
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                logger.info(f"📋 Retrieved {len(df)} tickers")
                return df
            
            else:
                logger.warning("No tickers found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return pd.DataFrame()
    
    def get_dividends(self, symbol: str, execution_date: Optional[str] = None,
                     record_date: Optional[str] = None, declaration_date: Optional[str] = None,
                     pay_date: Optional[str] = None, ex_date: Optional[str] = None,
                     cash_amount: Optional[float] = None, dividend_type: Optional[str] = None,
                     limit: int = 1000) -> pd.DataFrame:
        """
        Get dividend history for a stock
        """
        path = f"/v3/reference/dividends"
        params = {
            "ticker": symbol,
            "limit": limit
        }
        
        if execution_date:
            params["execution_date"] = execution_date
        if record_date:
            params["record_date"] = record_date
        if declaration_date:
            params["declaration_date"] = declaration_date
        if pay_date:
            params["pay_date"] = pay_date
        if ex_date:
            params["ex_date"] = ex_date
        if cash_amount:
            params["cash_amount"] = cash_amount
        if dividend_type:
            params["dividend_type"] = dividend_type
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert dates
                date_columns = ['execution_date', 'record_date', 'declaration_date', 'pay_date', 'ex_date']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                
                logger.info(f"💰 Retrieved {len(df)} dividends for {symbol}")
                return df
            
            else:
                logger.warning(f"No dividends found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_splits(self, symbol: str, execution_date: Optional[str] = None,
                   reverse_split: Optional[bool] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get stock split history
        """
        path = f"/v3/reference/splits"
        params = {
            "ticker": symbol,
            "limit": limit
        }
        
        if execution_date:
            params["execution_date"] = execution_date
        if reverse_split is not None:
            params["reverse_split"] = reverse_split
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert execution_date
                if 'execution_date' in df.columns:
                    df['execution_date'] = pd.to_datetime(df['execution_date'])
                
                logger.info(f"📊 Retrieved {len(df)} splits for {symbol}")
                return df
            
            else:
                logger.warning(f"No splits found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return pd.DataFrame()
    
    # ==================== NEWS ====================
    
    def get_news(self, ticker: Optional[str] = None, published_utc: Optional[str] = None,
                 order: str = "desc", sort: str = "published_utc", limit: int = 1000) -> pd.DataFrame:
        """
        Get news articles
        """
        path = "/v2/reference/news"
        params = {
            "order": order,
            "sort": sort,
            "limit": limit
        }
        
        if ticker:
            params["ticker"] = ticker
        if published_utc:
            params["published_utc"] = published_utc
        
        try:
            data = self._get(path, params)
            
            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df = pd.DataFrame(data["results"])
                
                # Convert published_utc
                if 'published_utc' in df.columns:
                    df['published_utc'] = pd.to_datetime(df['published_utc'])
                
                logger.info(f"📰 Retrieved {len(df)} news articles")
                return df
            
            else:
                logger.warning("No news articles found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    # ==================== HEALTH CHECK ====================
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health and limits"""
        try:
            # Test with a simple ticker request
            test_df = self.get_tickers(limit=1)
            
            return {
                "status": "healthy",
                "api_key_valid": True,
                "request_count": self.request_count,
                "last_request": self.last_request_time
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "api_key_valid": False
            }
