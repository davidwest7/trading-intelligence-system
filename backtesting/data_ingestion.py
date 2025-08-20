#!/usr/bin/env python3
"""
Data Ingestion System
====================

Comprehensive data ingestion for backtesting:
- Polygon Pro data fetching
- S3 data lake storage
- Data validation with Great Expectations
- Corporate action handling
- Partitioned Parquet storage
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional imports with fallbacks
try:
    import boto3
    import s3fs
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("S3 dependencies not available. S3 functionality will be disabled.")

from .polygon_client import PolygonClient, PolygonConfig

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation using Great Expectations patterns"""
    
    @staticmethod
    def validate_bars_df(df: pd.DataFrame) -> bool:
        """Validate OHLCV bars dataframe"""
        if df.empty:
            return True
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.error("timestamp column must be datetime")
            return False
        
        # Check for nulls in critical columns
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0]}")
        
        # Check price logic
        price_errors = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if price_errors.any():
            logger.error(f"Price logic errors found: {price_errors.sum()} rows")
            return False
        
        # Check volume
        if (df['volume'] < 0).any():
            logger.error("Negative volume found")
            return False
        
        # Check timestamp monotonicity
        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Timestamps are not monotonically increasing")
        
        logger.info(f"✅ Data validation passed for {len(df)} rows")
        return True
    
    @staticmethod
    def validate_trades_df(df: pd.DataFrame) -> bool:
        """Validate trades dataframe"""
        if df.empty:
            return True
        
        required_cols = ['timestamp', 'price', 'size']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check price and size
        if (df['price'] <= 0).any():
            logger.error("Non-positive prices found")
            return False
        
        if (df['size'] <= 0).any():
            logger.error("Non-positive sizes found")
            return False
        
        logger.info(f"✅ Trades validation passed for {len(df)} rows")
        return True

class S3Storage:
    """S3 storage utilities for partitioned Parquet data"""
    
    def __init__(self, bucket: str, prefix: str = "polygon"):
        if not S3_AVAILABLE:
            raise ImportError("S3 dependencies not available. Install s3fs and boto3.")
        
        self.bucket = bucket
        self.prefix = prefix
        self.fs = s3fs.S3FileSystem()
        
        # Create S3 client for metadata operations
        self.s3_client = boto3.client('s3')
        
        logger.info(f"🚀 S3 storage initialized: s3://{bucket}/{prefix}")
    
    def write_parquet_partitioned(self, df: pd.DataFrame, 
                                 s3_prefix: str,
                                 partition_cols: List[str],
                                 compression: str = 'zstd') -> str:
        """
        Write DataFrame to S3 as partitioned Parquet
        
        Args:
            df: DataFrame to write
            s3_prefix: S3 prefix (e.g., 'equities/bars_1m')
            partition_cols: Columns to partition by
            compression: Compression algorithm
        
        Returns:
            S3 path where data was written
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping write")
            return ""
        
        # Create full S3 path
        full_prefix = f"s3://{self.bucket}/{self.prefix}/{s3_prefix}"
        
        try:
            # Write partitioned Parquet
            df.to_parquet(
                full_prefix,
                partition_cols=partition_cols,
                compression=compression,
                filesystem=self.fs,
                index=False
            )
            
            logger.info(f"✅ Wrote {len(df)} rows to {full_prefix}")
            return full_prefix
            
        except Exception as e:
            logger.error(f"❌ Failed to write to S3: {e}")
            raise
    
    def read_parquet_partitioned(self, s3_prefix: str, 
                                filters: Optional[List[Tuple]] = None) -> pd.DataFrame:
        """
        Read partitioned Parquet from S3
        
        Args:
            s3_prefix: S3 prefix to read from
            filters: PyArrow filters for partition pruning
        
        Returns:
            DataFrame with data
        """
        full_prefix = f"s3://{self.bucket}/{self.prefix}/{s3_prefix}"
        
        try:
            df = pd.read_parquet(
                full_prefix,
                filesystem=self.fs,
                filters=filters
            )
            
            logger.info(f"✅ Read {len(df)} rows from {full_prefix}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to read from S3: {e}")
            return pd.DataFrame()
    
    def list_partitions(self, s3_prefix: str) -> List[str]:
        """List available partitions"""
        full_prefix = f"s3://{self.bucket}/{self.prefix}/{s3_prefix}"
        
        try:
            partitions = []
            for path in self.fs.ls(full_prefix, detail=False):
                partitions.append(path)
            return partitions
        except Exception as e:
            logger.error(f"❌ Failed to list partitions: {e}")
            return []

class LocalStorage:
    """Local storage utilities for partitioned Parquet data"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Local storage initialized: {self.base_path}")
    
    def write_parquet_partitioned(self, df: pd.DataFrame, 
                                 prefix: str,
                                 partition_cols: List[str],
                                 compression: str = 'zstd') -> str:
        """Write DataFrame to local storage as partitioned Parquet"""
        if df.empty:
            logger.warning("Empty DataFrame, skipping write")
            return ""
        
        # Create full path
        full_path = self.base_path / prefix
        
        try:
            # Write partitioned Parquet
            df.to_parquet(
                full_path,
                partition_cols=partition_cols,
                compression=compression,
                index=False
            )
            
            logger.info(f"✅ Wrote {len(df)} rows to {full_path}")
            return str(full_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to write to local storage: {e}")
            raise
    
    def read_parquet_partitioned(self, prefix: str, 
                                filters: Optional[List[Tuple]] = None) -> pd.DataFrame:
        """Read partitioned Parquet from local storage"""
        full_path = self.base_path / prefix
        
        try:
            df = pd.read_parquet(
                full_path,
                filters=filters
            )
            
            logger.info(f"✅ Read {len(df)} rows from {full_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to read from local storage: {e}")
            return pd.DataFrame()
    
    def list_partitions(self, prefix: str) -> List[str]:
        """List available partitions"""
        full_path = self.base_path / prefix
        
        try:
            if full_path.exists():
                partitions = []
                for item in full_path.iterdir():
                    if item.is_dir():
                        partitions.append(str(item))
                return partitions
            return []
        except Exception as e:
            logger.error(f"❌ Failed to list partitions: {e}")
            return []

class CorporateActionProcessor:
    """Handle corporate actions (splits, dividends)"""
    
    def __init__(self, polygon_client: PolygonClient):
        self.polygon_client = polygon_client
    
    def get_corporate_actions(self, symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get all corporate actions for a symbol"""
        actions = {}
        
        # Get splits
        splits = self.polygon_client.get_splits(symbol)
        if not splits.empty:
            splits = splits[
                (splits['execution_date'] >= start_date) & 
                (splits['execution_date'] <= end_date)
            ]
            actions['splits'] = splits
        
        # Get dividends
        dividends = self.polygon_client.get_dividends(symbol)
        if not dividends.empty:
            dividends = dividends[
                (dividends['ex_date'] >= start_date) & 
                (dividends['ex_date'] <= end_date)
            ]
            actions['dividends'] = dividends
        
        return actions
    
    def adjust_prices(self, df: pd.DataFrame, symbol: str, 
                     start_date: str, end_date: str) -> pd.DataFrame:
        """
        Adjust prices for corporate actions
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            start_date: Start date for corporate actions
            end_date: End date for corporate actions
        
        Returns:
            Adjusted OHLCV DataFrame
        """
        if df.empty:
            return df
        
        actions = self.get_corporate_actions(symbol, start_date, end_date)
        
        # Create a copy for adjustments
        adjusted_df = df.copy()
        
        # Apply splits (most important)
        if 'splits' in actions and not actions['splits'].empty:
            for _, split in actions['splits'].iterrows():
                split_date = split['execution_date']
                split_ratio = split['split_from'] / split['split_to']
                
                # Adjust all prices before the split
                mask = adjusted_df['timestamp'] < split_date
                price_cols = ['open', 'high', 'low', 'close']
                
                for col in price_cols:
                    adjusted_df.loc[mask, col] = adjusted_df.loc[mask, col] * split_ratio
                
                logger.info(f"Applied {split_ratio:.2f} split for {symbol} on {split_date}")
        
        # Apply dividends (less critical for OHLCV, but good for completeness)
        if 'dividends' in actions and not actions['dividends'].empty:
            for _, dividend in actions['dividends'].iterrows():
                ex_date = dividend['ex_date']
                cash_amount = dividend['cash_amount']
                
                # Adjust close prices before ex-date
                mask = adjusted_df['timestamp'] < ex_date
                adjusted_df.loc[mask, 'close'] = adjusted_df.loc[mask, 'close'] - cash_amount
                
                logger.info(f"Applied ${cash_amount:.2f} dividend for {symbol} on {ex_date}")
        
        return adjusted_df

class PolygonDataIngestion:
    """
    Comprehensive Polygon data ingestion system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data ingestion system
        
        Args:
            config: Configuration dictionary with:
                - polygon_api_key: Polygon API key
                - s3_bucket: S3 bucket name (optional)
                - s3_prefix: S3 prefix (default: 'polygon')
                - max_workers: Max concurrent workers
        """
        self.config = config
        
        # Initialize components
        polygon_config = PolygonConfig(
            api_key=config['polygon_api_key'],
            rate_limit_delay=config.get('rate_limit_delay', 0.1)
        )
        self.polygon_client = PolygonClient(polygon_config)
        
        # Initialize storage (S3 if available, otherwise local)
        if config.get('s3_bucket') and S3_AVAILABLE:
            self.storage = S3Storage(
                bucket=config['s3_bucket'],
                prefix=config.get('s3_prefix', 'polygon')
            )
            self.storage_type = 's3'
        else:
            self.storage = LocalStorage(
                base_path=config.get('local_path', 'data')
            )
            self.storage_type = 'local'
        
        self.validator = DataValidator()
        self.corp_action_processor = CorporateActionProcessor(self.polygon_client)
        
        self.max_workers = config.get('max_workers', 5)
        
        logger.info(f"🚀 Polygon Data Ingestion System initialized with {self.storage_type} storage")
    
    def download_bars_s3(self, symbols: List[str], start_date: str, end_date: str,
                        timeframe: str, adjusted: bool = True) -> Dict[str, str]:
        """
        Download bars data and store
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe (1m, 5m, 1h, 1d)
            adjusted: Whether to use adjusted prices
        
        Returns:
            Dictionary mapping symbols to storage paths
        """
        # Parse timeframe
        multiplier, timespan = self._parse_timeframe(timeframe)
        
        results = {}
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._download_symbol_bars,
                    symbol, start_date, end_date, multiplier, timespan, adjusted
                ): symbol for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    storage_path = future.result()
                    results[symbol] = storage_path
                except Exception as e:
                    logger.error(f"❌ Failed to download bars for {symbol}: {e}")
                    results[symbol] = ""
        
        return results
    
    def _download_symbol_bars(self, symbol: str, start_date: str, end_date: str,
                             multiplier: int, timespan: str, adjusted: bool) -> str:
        """Download bars for a single symbol"""
        try:
            # Get bars from Polygon
            df = self.polygon_client.get_aggregates(
                symbol=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date,
                adjusted=adjusted
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return ""
            
            # Validate data
            if not self.validator.validate_bars_df(df):
                logger.error(f"Data validation failed for {symbol}")
                return ""
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Ensure date column exists
            if 'date' not in df.columns:
                df['date'] = df['timestamp'].dt.date
            
            # Write to storage
            prefix = f"equities/bars_{timespan}"
            storage_path = self.storage.write_parquet_partitioned(
                df=df,
                prefix=prefix,
                partition_cols=['symbol', 'date']
            )
            
            return storage_path
            
        except Exception as e:
            logger.error(f"Error downloading bars for {symbol}: {e}")
            raise
    
    def download_trades_s3(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, str]:
        """Download trades data and store"""
        results = {}
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for symbol in symbols:
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    
                    future = executor.submit(
                        self._download_symbol_trades,
                        symbol, date_str
                    )
                    
                    try:
                        storage_path = future.result()
                        if storage_path:
                            results[f"{symbol}_{date_str}"] = storage_path
                    except Exception as e:
                        logger.error(f"Failed to download trades for {symbol} on {date_str}: {e}")
        
        return results
    
    def _download_symbol_trades(self, symbol: str, date: str) -> str:
        """Download trades for a single symbol on a single date"""
        try:
            df = self.polygon_client.get_trades(symbol, date)
            
            if df.empty:
                return ""
            
            if not self.validator.validate_trades_df(df):
                return ""
            
            # Add metadata
            df['symbol'] = symbol
            df['date'] = date
            
            # Write to storage
            storage_path = self.storage.write_parquet_partitioned(
                df=df,
                prefix="equities/trades",
                partition_cols=['symbol', 'date']
            )
            
            return storage_path
            
        except Exception as e:
            logger.error(f"Error downloading trades for {symbol} on {date}: {e}")
            return ""
    
    def download_reference_data(self, start_date: str, end_date: str) -> Dict[str, str]:
        """Download reference data (tickers, corporate actions, news)"""
        results = {}
        
        try:
            # Get all tickers
            tickers_df = self.polygon_client.get_tickers(type="CS", market="stocks", active=True)
            
            if not tickers_df.empty:
                # Add date for partitioning
                tickers_df['date'] = start_date
                
                storage_path = self.storage.write_parquet_partitioned(
                    df=tickers_df,
                    prefix="reference/tickers",
                    partition_cols=['date']
                )
                results['tickers'] = storage_path
            
            # Get news
            news_df = self.polygon_client.get_news(
                published_utc=f"{start_date}/{end_date}"
            )
            
            if not news_df.empty:
                news_df['date'] = news_df['published_utc'].dt.date.astype(str)
                
                storage_path = self.storage.write_parquet_partitioned(
                    df=news_df,
                    prefix="reference/news",
                    partition_cols=['date']
                )
                results['news'] = storage_path
            
            return results
            
        except Exception as e:
            logger.error(f"Error downloading reference data: {e}")
            return results
    
    def create_adjusted_bars(self, symbols: List[str], start_date: str, end_date: str,
                           timeframe: str) -> Dict[str, str]:
        """
        Create adjusted bars with corporate actions applied
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        
        Returns:
            Dictionary mapping symbols to storage paths of adjusted data
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Read raw bars
                multiplier, timespan = self._parse_timeframe(timeframe)
                prefix = f"equities/bars_{timespan}"
                
                raw_df = self.storage.read_parquet_partitioned(
                    prefix,
                    filters=[('symbol', '=', symbol)]
                )
                
                if raw_df.empty:
                    logger.warning(f"No raw data found for {symbol}")
                    continue
                
                # Apply corporate actions
                adjusted_df = self.corp_action_processor.adjust_prices(
                    raw_df, symbol, start_date, end_date
                )
                
                # Write adjusted data
                storage_path = self.storage.write_parquet_partitioned(
                    df=adjusted_df,
                    prefix=f"equities/adj_bars_{timespan}",
                    partition_cols=['symbol', 'date']
                )
                
                results[symbol] = storage_path
                
            except Exception as e:
                logger.error(f"Error creating adjusted bars for {symbol}: {e}")
                results[symbol] = ""
        
        return results
    
    def _parse_timeframe(self, timeframe: str) -> Tuple[int, str]:
        """Parse timeframe string into multiplier and timespan"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1]), 'minute'
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]), 'hour'
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]), 'day'
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]), 'week'
        elif timeframe.endswith('M'):
            return int(timeframe[:-1]), 'month'
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        summary = {
            'bars': {},
            'trades': {},
            'reference': {}
        }
        
        # List available bar timeframes
        try:
            bar_prefixes = self.storage.list_partitions("equities")
            for prefix in bar_prefixes:
                if 'bars_' in prefix:
                    timeframe = prefix.split('bars_')[-1]
                    summary['bars'][timeframe] = len(self.storage.list_partitions(prefix))
        except Exception as e:
            logger.error(f"Error listing bar data: {e}")
        
        # List available trades
        try:
            summary['trades'] = len(self.storage.list_partitions("equities/trades"))
        except Exception as e:
            logger.error(f"Error listing trades: {e}")
        
        # List reference data
        try:
            ref_prefixes = self.storage.list_partitions("reference")
            for prefix in ref_prefixes:
                data_type = prefix.split('/')[-1]
                summary['reference'][data_type] = len(self.storage.list_partitions(prefix))
        except Exception as e:
            logger.error(f"Error listing reference data: {e}")
        
        return summary
