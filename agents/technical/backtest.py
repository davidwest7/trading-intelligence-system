"""
Purged Cross-Validation Backtesting for Technical Strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

from .models import TechnicalOpportunity


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio
        }


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str
    quantity: float
    pnl: float
    pnl_pct: float
    strategy: str
    
    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600


class PurgedCrossValidationBacktester:
    """
    Purged Cross-Validation Backtester for financial strategies
    
    Implements purging and embargo to avoid look-ahead bias in cross-validation
    """
    
    def __init__(self, 
                 purge_pct: float = 0.02,
                 embargo_pct: float = 0.01,
                 n_splits: int = 5):
        """
        Initialize backtester
        
        Args:
            purge_pct: Percentage of data to purge after each training set
            embargo_pct: Percentage of data to embargo after purging  
            n_splits: Number of cross-validation splits
        """
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.n_splits = n_splits
    
    async def run_backtest(self, strategy, data: Dict[str, pd.DataFrame], 
                          start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run purged cross-validation backtest
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Comprehensive backtest results
        """
        # Prepare data
        price_data = self._prepare_data(data, start_date, end_date)
        
        if price_data.empty:
            return {"error": "No data available for backtesting"}
        
        # Run purged cross-validation
        cv_results = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(price_data)):
            # Apply purging and embargo
            train_idx, test_idx = self._apply_purging_embargo(
                train_idx, test_idx, len(price_data)
            )
            
            if len(test_idx) == 0:
                continue
                
            # Split data
            train_data = price_data.iloc[train_idx]
            test_data = price_data.iloc[test_idx]
            
            # Run strategy on test data
            fold_result = await self._run_strategy_on_period(
                strategy, test_data, f"fold_{fold}"
            )
            
            cv_results.append(fold_result)
        
        # Aggregate results across folds
        aggregated_results = self._aggregate_cv_results(cv_results)
        
        return {
            "overall_results": aggregated_results.to_dict(),
            "cv_results": [result.to_dict() for result in cv_results],
            "n_folds": len(cv_results),
            "purge_pct": self.purge_pct,
            "embargo_pct": self.embargo_pct
        }
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame], 
                     start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare and combine data for backtesting"""
        if not data:
            return pd.DataFrame()
        
        # For simplicity, use first symbol's data
        # TODO: Implement multi-asset backtesting
        first_symbol = list(data.keys())[0]
        df = data[first_symbol].copy()
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        return df
    
    def _apply_purging_embargo(self, train_idx: np.ndarray, 
                              test_idx: np.ndarray, 
                              total_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply purging and embargo to avoid data leakage"""
        
        # Calculate purge and embargo lengths
        purge_length = int(total_length * self.purge_pct)
        embargo_length = int(total_length * self.embargo_pct)
        
        # Get the end of training period
        train_end = train_idx[-1]
        
        # Apply purging: remove data after training end
        purge_start = train_end + 1
        purge_end = min(purge_start + purge_length, total_length - 1)
        
        # Apply embargo: further remove data after purge
        embargo_start = purge_end + 1
        embargo_end = min(embargo_start + embargo_length, total_length - 1)
        
        # Remove purged and embargoed indices from test set
        test_idx_filtered = test_idx[test_idx > embargo_end]
        
        return train_idx, test_idx_filtered
    
    async def _run_strategy_on_period(self, strategy, data: pd.DataFrame, 
                                     period_name: str) -> BacktestResult:
        """Run strategy on a specific time period"""
        
        # Generate signals (simplified)
        # TODO: Integrate with actual strategy.analyze() method
        signals = self._generate_mock_signals(data)
        
        # Convert signals to trades
        trades = self._signals_to_trades(signals, data)
        
        # Calculate performance metrics
        if not trades:
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                num_trades=0,
                avg_trade_duration=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0
            )
        
        return self._calculate_performance_metrics(trades, data)
    
    def _generate_mock_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock trading signals for testing"""
        signals = pd.DataFrame(index=data.index)
        
        # Simple moving average crossover
        data['sma_fast'] = data['close'].rolling(10).mean()
        data['sma_slow'] = data['close'].rolling(30).mean()
        
        signals['signal'] = 0
        signals.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
        signals.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
        
        # Only trade on signal changes
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def _signals_to_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> List[Trade]:
        """Convert signals to individual trades"""
        trades = []
        current_position = None
        entry_price = None
        entry_time = None
        
        for timestamp, row in signals.iterrows():
            if current_position is None and row['position'] != 0:
                # Enter position
                current_position = row['signal']
                entry_price = data.loc[timestamp, 'close']
                entry_time = timestamp
                
            elif current_position is not None and (
                row['position'] != 0 or timestamp == signals.index[-1]
            ):
                # Exit position
                exit_price = data.loc[timestamp, 'close']
                exit_time = timestamp
                
                # Calculate PnL
                if current_position == 1:  # Long position
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:  # Short position
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                trade = Trade(
                    symbol="TEST",
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction="long" if current_position == 1 else "short",
                    quantity=1.0,
                    pnl=pnl_pct * 10000,  # Assume $10k position
                    pnl_pct=pnl_pct,
                    strategy="mock"
                )
                
                trades.append(trade)
                current_position = row['signal'] if row['position'] != 0 else None
                
                if current_position is not None:
                    entry_price = exit_price
                    entry_time = exit_time
        
        return trades
    
    def _calculate_performance_metrics(self, trades: List[Trade], 
                                     data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic metrics
        returns = [trade.pnl_pct for trade in trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        total_return = sum(returns)
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # Risk metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(min(drawdowns)) if drawdowns.size > 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duration
        durations = [trade.duration_hours for trade in trades]
        avg_duration = np.mean(durations) if durations else 0
        
        # Advanced ratios
        calmar_ratio = (total_return * 252) / max_drawdown if max_drawdown > 0 else 0
        
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0
        sortino_ratio = (np.mean(returns) * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_duration=avg_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    def _aggregate_cv_results(self, cv_results: List[BacktestResult]) -> BacktestResult:
        """Aggregate results across CV folds"""
        
        if not cv_results:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate means across folds
        total_return = np.mean([r.total_return for r in cv_results])
        sharpe_ratio = np.mean([r.sharpe_ratio for r in cv_results])
        max_drawdown = np.mean([r.max_drawdown for r in cv_results])
        win_rate = np.mean([r.win_rate for r in cv_results])
        profit_factor = np.mean([r.profit_factor for r in cv_results if r.profit_factor != float('inf')])
        num_trades = int(np.mean([r.num_trades for r in cv_results]))
        avg_trade_duration = np.mean([r.avg_trade_duration for r in cv_results])
        volatility = np.mean([r.volatility for r in cv_results])
        calmar_ratio = np.mean([r.calmar_ratio for r in cv_results])
        sortino_ratio = np.mean([r.sortino_ratio for r in cv_results])
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_duration=avg_trade_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
