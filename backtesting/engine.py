#!/usr/bin/env python3
"""
Backtest Engine
==============

Main backtesting engine that orchestrates:
- Data loading and preprocessing
- Strategy execution
- Portfolio management
- Performance tracking
- Error handling and retries
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import json
import yaml
import warnings
from pathlib import Path
import traceback
from dataclasses import dataclass, asdict
import hashlib

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .execution import ExecutionEngine, ExecutionConfig
from .data_ingestion import PolygonDataIngestion
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Data configuration
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: str = "1h"
    
    # Portfolio configuration
    initial_capital: float = 1000000.0
    rebalance_frequency: str = "1d"  # 1d, 1w, 1m
    
    # Execution configuration
    execution_config: ExecutionConfig = None
    
    # Data source configuration
    polygon_api_key: str = None
    s3_bucket: str = None
    s3_prefix: str = "polygon"
    local_path: str = "data"
    
    # Strategy configuration
    strategy_function: Callable = None
    strategy_params: Dict[str, Any] = None
    
    # Risk management
    max_drawdown: float = 0.15
    stop_loss: float = 0.10
    
    # Performance tracking
    benchmark_symbol: str = "SPY"
    
    def __post_init__(self):
        if self.polygon_api_key is None:
            self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        
        if self.execution_config is None:
            self.execution_config = ExecutionConfig()
        
        if self.strategy_params is None:
            self.strategy_params = {}

class BacktestEngine:
    """
    Comprehensive backtesting engine with error handling and retry mechanisms
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.results = {}
        self.portfolio_history = []
        self.trade_history = []
        self.error_log = []
        
        # Initialize components
        self._initialize_components()
        
        # Portfolio state
        self.portfolio = {
            'cash': config.initial_capital,
            'positions': {},
            'total_value': config.initial_capital,
            'weights': pd.Series(dtype=float)
        }
        
        # Performance tracking
        self.metrics = BacktestMetrics()
        
        logger.info("🚀 Backtest Engine initialized")
    
    def _initialize_components(self):
        """Initialize all backtesting components"""
        try:
            # Initialize data ingestion
            data_config = {
                'polygon_api_key': self.config.polygon_api_key,
                's3_bucket': self.config.s3_bucket,
                's3_prefix': self.config.s3_prefix,
                'local_path': self.config.local_path
            }
            self.data_ingestion = PolygonDataIngestion(data_config)
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(self.config.execution_config)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            self.error_log.append({
                'timestamp': datetime.now(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'component': 'initialization'
            })
            raise
    
    def run_backtest(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Run the complete backtest with error handling and retries
        
        Args:
            max_retries: Maximum number of retry attempts
        
        Returns:
            Backtest results
        """
        logger.info("🚀 Starting backtest...")
        
        for attempt in range(max_retries):
            try:
                results = self._run_backtest_internal()
                logger.info("✅ Backtest completed successfully")
                return results
                
            except Exception as e:
                logger.error(f"❌ Backtest failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Log error details
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'attempt': attempt + 1,
                    'component': 'backtest'
                })
                
                # If this is the last attempt, raise the error
                if attempt == max_retries - 1:
                    logger.error("❌ Backtest failed after all retry attempts")
                    raise
                
                # Wait before retrying
                wait_time = 2 ** attempt
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Reset portfolio state for retry
                self._reset_portfolio()
    
    def _run_backtest_internal(self) -> Dict[str, Any]:
        """Internal backtest execution"""
        # Load data
        data = self._load_data()
        if data.empty:
            raise ValueError("No data loaded for backtest")
        
        # Initialize portfolio
        self._initialize_portfolio(data)
        
        # Run backtest loop
        self._run_backtest_loop(data)
        
        # Calculate final results
        results = self._calculate_results()
        
        return results
    
    def _load_data(self) -> pd.DataFrame:
        """Load and prepare data for backtesting"""
        logger.info("📊 Loading data...")
        
        try:
            # Load bars data
            bars_data = {}
            for symbol in self.config.symbols:
                # Try to load from storage first
                prefix = f"equities/bars_{self.config.timeframe}"
                df = self.data_ingestion.storage.read_parquet_partitioned(
                    prefix,
                    filters=[('symbol', '=', symbol)]
                )
                
                if not df.empty:
                    bars_data[symbol] = df
                    continue
                
                # If storage data not available, download from Polygon
                if self.config.polygon_api_key:
                    logger.info(f"Downloading data for {symbol}...")
                    download_results = self.data_ingestion.download_bars_s3(
                        symbols=[symbol],
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        timeframe=self.config.timeframe
                    )
                    
                    if download_results.get(symbol):
                        df = self.data_ingestion.storage.read_parquet_partitioned(
                            prefix,
                            filters=[('symbol', '=', symbol)]
                        )
                        bars_data[symbol] = df
                else:
                    logger.warning(f"No data available for {symbol} and no API key provided")
            
            # Combine all data
            if not bars_data:
                raise ValueError("No data loaded for any symbols")
            
            # Create price matrix
            price_data = self._create_price_matrix(bars_data)
            
            logger.info(f"✅ Loaded data for {len(bars_data)} symbols")
            return price_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _create_price_matrix(self, bars_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create price matrix from bars data"""
        # Get common date range
        all_dates = set()
        for symbol, df in bars_data.items():
            if not df.empty:
                all_dates.update(df['timestamp'].dt.date)
        
        all_dates = sorted(list(all_dates))
        
        # Create price matrix
        price_matrix = pd.DataFrame(index=all_dates)
        
        for symbol, df in bars_data.items():
            if not df.empty:
                # Pivot to get close prices
                symbol_prices = df.set_index('timestamp')['close']
                price_matrix[symbol] = symbol_prices
        
        # Forward fill missing values
        price_matrix = price_matrix.fillna(method='ffill')
        
        return price_matrix
    
    def _initialize_portfolio(self, data: pd.DataFrame):
        """Initialize portfolio with starting positions"""
        logger.info("💰 Initializing portfolio...")
        
        # Set initial weights to zero
        self.portfolio['weights'] = pd.Series(0.0, index=data.columns)
        
        # Record initial state
        self.portfolio_history.append({
            'timestamp': data.index[0],
            'cash': self.portfolio['cash'],
            'positions': self.portfolio['positions'].copy(),
            'total_value': self.portfolio['total_value'],
            'weights': self.portfolio['weights'].copy()
        })
    
    def _run_backtest_loop(self, data: pd.DataFrame):
        """Run the main backtest loop"""
        logger.info("🔄 Running backtest loop...")
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(data.index)
        
        for i, current_date in enumerate(data.index):
            try:
                # Check if it's a rebalance date
                if current_date in rebalance_dates:
                    self._rebalance_portfolio(data, current_date)
                
                # Update portfolio value
                self._update_portfolio_value(data, current_date)
                
                # Check risk limits
                if self._check_risk_limits():
                    logger.warning("⚠️ Risk limits exceeded, stopping backtest")
                    break
                
                # Record portfolio state
                self._record_portfolio_state(current_date)
                
            except Exception as e:
                logger.error(f"❌ Error in backtest loop at {current_date}: {e}")
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'component': 'backtest_loop',
                    'date': current_date
                })
                # Continue with next iteration
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """Get rebalance dates based on frequency"""
        if self.config.rebalance_frequency == "1d":
            return list(dates)
        elif self.config.rebalance_frequency == "1w":
            # Rebalance weekly (every 7 days)
            rebalance_dates = []
            for i in range(0, len(dates), 7):
                rebalance_dates.append(dates[i])
            return rebalance_dates
        elif self.config.rebalance_frequency == "1m":
            # Rebalance monthly
            rebalance_dates = []
            current_month = None
            for date in dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
            return rebalance_dates
        else:
            # Default to daily
            return list(dates)
    
    def _rebalance_portfolio(self, data: pd.DataFrame, current_date: datetime):
        """Rebalance portfolio based on strategy"""
        try:
            # Get current prices
            current_prices = data.loc[current_date]
            
            # Generate target weights from strategy
            if self.config.strategy_function:
                target_weights = self.config.strategy_function(
                    data=data,
                    current_date=current_date,
                    current_prices=current_prices,
                    **self.config.strategy_params
                )
            else:
                # Default to equal weight
                target_weights = pd.Series(1.0 / len(self.config.symbols), index=self.config.symbols)
            
            # Execute trades
            execution_results = self._execute_trades(target_weights, current_prices, current_date)
            
            # Update portfolio
            self._update_portfolio(execution_results)
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio rebalancing: {e}")
            self.error_log.append({
                'timestamp': datetime.now(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'component': 'rebalancing',
                'date': current_date
            })
    
    def _execute_trades(self, target_weights: pd.Series, 
                       current_prices: pd.Series, 
                       current_date: datetime) -> Dict[str, Any]:
        """Execute trades using execution engine"""
        try:
            # Convert to DataFrames for execution engine
            target_weights_df = pd.DataFrame([target_weights], index=[current_date])
            current_weights_df = pd.DataFrame([self.portfolio['weights']], index=[current_date])
            prices_df = pd.DataFrame([current_prices], index=[current_date])
            
            # Create volume data (simplified - could be enhanced)
            volumes_df = pd.DataFrame(1000000, index=[current_date], columns=current_prices.index)
            
            # Execute trades
            execution_results = self.execution_engine.execute_trades(
                target_weights=target_weights_df,
                current_weights=current_weights_df,
                prices=prices_df,
                volumes=volumes_df,
                portfolio_value=self.portfolio['total_value']
            )
            
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ Error executing trades: {e}")
            # Return empty results
            return {
                'executed_weights': pd.DataFrame(),
                'trades': pd.DataFrame(),
                'costs': 0.0,
                'slippage': 0.0,
                'impact': 0.0
            }
    
    def _update_portfolio(self, execution_results: Dict[str, Any]):
        """Update portfolio based on execution results"""
        try:
            # Update weights
            if not execution_results['executed_weights'].empty:
                self.portfolio['weights'] = execution_results['executed_weights'].iloc[0]
            
            # Update cash (subtract costs)
            self.portfolio['cash'] -= execution_results['costs']
            
            # Update total value
            self.portfolio['total_value'] -= execution_results['costs']
            
            # Record trades
            if not execution_results['trades'].empty:
                self.trade_history.extend(execution_results['trades'].to_dict('records'))
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio: {e}")
    
    def _update_portfolio_value(self, data: pd.DataFrame, current_date: datetime):
        """Update portfolio value based on current prices"""
        try:
            current_prices = data.loc[current_date]
            
            # Calculate position values
            position_values = {}
            total_position_value = 0
            
            for symbol, weight in self.portfolio['weights'].items():
                if symbol in current_prices and weight != 0:
                    position_value = weight * self.portfolio['total_value']
                    position_values[symbol] = position_value
                    total_position_value += position_value
            
            # Update portfolio
            self.portfolio['positions'] = position_values
            self.portfolio['total_value'] = self.portfolio['cash'] + total_position_value
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio value: {e}")
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are exceeded"""
        try:
            # Check drawdown
            if len(self.portfolio_history) > 1:
                initial_value = self.portfolio_history[0]['total_value']
                current_value = self.portfolio['total_value']
                drawdown = (initial_value - current_value) / initial_value
                
                if drawdown > self.config.max_drawdown:
                    logger.warning(f"⚠️ Maximum drawdown exceeded: {drawdown:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return False
    
    def _record_portfolio_state(self, current_date: datetime):
        """Record current portfolio state"""
        try:
            self.portfolio_history.append({
                'timestamp': current_date,
                'cash': self.portfolio['cash'],
                'positions': self.portfolio['positions'].copy(),
                'total_value': self.portfolio['total_value'],
                'weights': self.portfolio['weights'].copy()
            })
        except Exception as e:
            logger.error(f"❌ Error recording portfolio state: {e}")
    
    def _reset_portfolio(self):
        """Reset portfolio state for retry"""
        self.portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},
            'total_value': self.config.initial_capital,
            'weights': pd.Series(dtype=float)
        }
        self.portfolio_history = []
        self.trade_history = []
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtest results"""
        logger.info("📊 Calculating results...")
        
        try:
            # Create portfolio DataFrame
            portfolio_df = pd.DataFrame(self.portfolio_history)
            
            if portfolio_df.empty:
                raise ValueError("No portfolio history available")
            
            # Calculate performance metrics
            performance_metrics = self.metrics.calculate_metrics(portfolio_df)
            
            # Calculate execution metrics
            execution_metrics = self.execution_engine.get_execution_summary()
            
            # Create results
            results = {
                'portfolio_history': portfolio_df,
                'trade_history': pd.DataFrame(self.trade_history),
                'performance_metrics': performance_metrics,
                'execution_metrics': execution_metrics,
                'error_log': self.error_log,
                'config': asdict(self.config),
                'summary': {
                    'total_return': performance_metrics.get('total_return', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0),
                    'total_trades': execution_metrics.get('total_trades', 0),
                    'total_costs': execution_metrics.get('total_volume', 0) * 0.0015  # Estimate
                }
            }
            
            logger.info("✅ Results calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error calculating results: {e}")
            raise
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        try:
            # Convert DataFrames to dict for JSON serialization
            results_copy = self.results.copy()
            
            if 'portfolio_history' in results_copy:
                results_copy['portfolio_history'] = results_copy['portfolio_history'].to_dict()
            
            if 'trade_history' in results_copy:
                results_copy['trade_history'] = results_copy['trade_history'].to_dict()
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            logger.info(f"✅ Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            raise
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        if not self.error_log:
            return {'total_errors': 0}
        
        error_types = {}
        for error in self.error_log:
            error_type = error.get('component', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'latest_error': self.error_log[-1] if self.error_log else None
        }
