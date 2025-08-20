#!/usr/bin/env python3
"""
Backtest Metrics
===============

Comprehensive performance and risk metrics calculation:
- Return metrics (total, annualized, etc.)
- Risk metrics (volatility, drawdown, VaR, etc.)
- Risk-adjusted metrics (Sharpe, Sortino, etc.)
- Attribution analysis
- Benchmark comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestMetrics:
    """
    Comprehensive backtest metrics calculator
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        logger.info("📊 Backtest Metrics initialized")
    
    def calculate_metrics(self, portfolio_df: pd.DataFrame, 
                         benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_df: Portfolio history DataFrame
            benchmark_df: Benchmark history DataFrame (optional)
        
        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Calculate returns
            returns = self._calculate_returns(portfolio_df)
            
            # Basic metrics
            metrics = {
                'total_return': self._calculate_total_return(portfolio_df),
                'annualized_return': self._calculate_annualized_return(returns),
                'volatility': self._calculate_volatility(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(portfolio_df),
                'calmar_ratio': self._calculate_calmar_ratio(returns, portfolio_df),
                'var_95': self._calculate_var(returns, 0.95),
                'cvar_95': self._calculate_cvar(returns, 0.95),
                'win_rate': self._calculate_win_rate(returns),
                'profit_factor': self._calculate_profit_factor(returns),
                'avg_win': self._calculate_avg_win(returns),
                'avg_loss': self._calculate_avg_loss(returns),
                'max_consecutive_wins': self._calculate_max_consecutive_wins(returns),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(returns),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns),
                'information_ratio': None,  # Will be calculated if benchmark provided
                'beta': None,  # Will be calculated if benchmark provided
                'alpha': None,  # Will be calculated if benchmark provided
                'tracking_error': None,  # Will be calculated if benchmark provided
                'up_capture': None,  # Will be calculated if benchmark provided
                'down_capture': None,  # Will be calculated if benchmark provided
                'monthly_returns': self._calculate_monthly_returns(returns),
                'rolling_metrics': self._calculate_rolling_metrics(returns),
                'drawdown_periods': self._calculate_drawdown_periods(portfolio_df)
            }
            
            # Calculate benchmark metrics if provided
            if benchmark_df is not None:
                benchmark_returns = self._calculate_returns(benchmark_df)
                metrics.update({
                    'information_ratio': self._calculate_information_ratio(returns, benchmark_returns),
                    'beta': self._calculate_beta(returns, benchmark_returns),
                    'alpha': self._calculate_alpha(returns, benchmark_returns),
                    'tracking_error': self._calculate_tracking_error(returns, benchmark_returns),
                    'up_capture': self._calculate_up_capture(returns, benchmark_returns),
                    'down_capture': self._calculate_down_capture(returns, benchmark_returns),
                    'correlation': self._calculate_correlation(returns, benchmark_returns)
                })
            
            logger.info("✅ Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def _calculate_returns(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns"""
        if 'total_value' not in portfolio_df.columns:
            raise ValueError("Portfolio DataFrame must contain 'total_value' column")
        
        # Calculate percentage returns
        returns = portfolio_df['total_value'].pct_change().dropna()
        return returns
    
    def _calculate_total_return(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate total return"""
        if len(portfolio_df) < 2:
            return 0.0
        
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]
        
        return (final_value - initial_value) / initial_value
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Assuming 252 trading days per year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        volatility = returns.std()
        
        if volatility == 0:
            return 0.0
        
        return (excess_returns.mean() / volatility) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        return (excess_returns.mean() / downside_deviation) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_df) < 2:
            return 0.0
        
        cumulative = portfolio_df['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, returns: pd.Series, portfolio_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0
        
        annualized_return = self._calculate_annualized_return(returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_df)
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate"""
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).mean()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def _calculate_avg_win(self, returns: pd.Series) -> float:
        """Calculate average winning return"""
        winning_returns = returns[returns > 0]
        return winning_returns.mean() if len(winning_returns) > 0 else 0.0
    
    def _calculate_avg_loss(self, returns: pd.Series) -> float:
        """Calculate average losing return"""
        losing_returns = returns[returns < 0]
        return losing_returns.mean() if len(losing_returns) > 0 else 0.0
    
    def _calculate_max_consecutive_wins(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive wins"""
        if len(returns) == 0:
            return 0
        
        wins = (returns > 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for win in wins:
            if win == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        if len(returns) == 0:
            return 0
        
        losses = (returns < 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness"""
        if len(returns) == 0:
            return 0.0
        
        return stats.skew(returns)
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate return kurtosis"""
        if len(returns) == 0:
            return 0.0
        
        return stats.kurtosis(returns)
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return excess_returns.mean() / tracking_error
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate alpha"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        beta = self._calculate_beta(returns, benchmark_returns)
        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        
        return portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        excess_returns = aligned_returns - aligned_benchmark
        return excess_returns.std() * np.sqrt(252)
    
    def _calculate_up_capture(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate up capture ratio"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        # Up periods
        up_mask = aligned_benchmark > 0
        if not up_mask.any():
            return 0.0
        
        portfolio_up_return = aligned_returns[up_mask].sum()
        benchmark_up_return = aligned_benchmark[up_mask].sum()
        
        if benchmark_up_return == 0:
            return 0.0
        
        return portfolio_up_return / benchmark_up_return
    
    def _calculate_down_capture(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate down capture ratio"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        # Down periods
        down_mask = aligned_benchmark < 0
        if not down_mask.any():
            return 0.0
        
        portfolio_down_return = aligned_returns[down_mask].sum()
        benchmark_down_return = aligned_benchmark[down_mask].sum()
        
        if benchmark_down_return == 0:
            return 0.0
        
        return portfolio_down_return / benchmark_down_return
    
    def _calculate_correlation(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate correlation with benchmark"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        return aligned_returns.corr(aligned_benchmark)
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate monthly returns"""
        if len(returns) == 0:
            return pd.Series()
        
        # Resample to monthly frequency
        monthly_returns = (1 + returns).resample('M').prod() - 1
        return monthly_returns
    
    def _calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> Dict[str, pd.Series]:
        """Calculate rolling metrics"""
        if len(returns) == 0:
            return {}
        
        rolling_metrics = {
            'rolling_sharpe': returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252),
            'rolling_volatility': returns.rolling(window).std() * np.sqrt(252),
            'rolling_max_drawdown': self._calculate_rolling_drawdown(returns, window)
        }
        
        return rolling_metrics
    
    def _calculate_rolling_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        if len(returns) == 0:
            return pd.Series()
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return drawdown.rolling(window).min()
    
    def _calculate_drawdown_periods(self, portfolio_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate drawdown periods"""
        if len(portfolio_df) < 2:
            return []
        
        cumulative = portfolio_df['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd == 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_idx is not None:
                    period = {
                        'start_date': portfolio_df.index[start_idx],
                        'end_date': portfolio_df.index[i],
                        'duration': i - start_idx,
                        'max_drawdown': drawdown.iloc[start_idx:i+1].min(),
                        'recovery_time': i - start_idx
                    }
                    drawdown_periods.append(period)
                start_idx = None
        
        return drawdown_periods
    
    def generate_tearsheet(self, portfolio_df: pd.DataFrame, 
                          benchmark_df: Optional[pd.DataFrame] = None) -> str:
        """Generate a comprehensive tearsheet report"""
        try:
            metrics = self.calculate_metrics(portfolio_df, benchmark_df)
            
            tearsheet = """
            ========================================
            BACKTEST PERFORMANCE TEARSHEET
            ========================================
            
            PERFORMANCE METRICS:
            --------------------
            Total Return: {total_return:.2%}
            Annualized Return: {annualized_return:.2%}
            Volatility: {volatility:.2%}
            Sharpe Ratio: {sharpe_ratio:.2f}
            Sortino Ratio: {sortino_ratio:.2f}
            Calmar Ratio: {calmar_ratio:.2f}
            
            RISK METRICS:
            -------------
            Maximum Drawdown: {max_drawdown:.2%}
            VaR (95%): {var_95:.2%}
            CVaR (95%): {cvar_95:.2%}
            
            TRADING METRICS:
            ----------------
            Win Rate: {win_rate:.2%}
            Profit Factor: {profit_factor:.2f}
            Average Win: {avg_win:.2%}
            Average Loss: {avg_loss:.2%}
            Max Consecutive Wins: {max_consecutive_wins}
            Max Consecutive Losses: {max_consecutive_losses}
            
            DISTRIBUTION METRICS:
            ---------------------
            Skewness: {skewness:.2f}
            Kurtosis: {kurtosis:.2f}
            """.format(**metrics)
            
            if benchmark_df is not None:
                tearsheet += """
                BENCHMARK COMPARISON:
                ---------------------
                Information Ratio: {information_ratio:.2f}
                Beta: {beta:.2f}
                Alpha: {alpha:.2%}
                Tracking Error: {tracking_error:.2%}
                Up Capture: {up_capture:.2f}
                Down Capture: {down_capture:.2f}
                Correlation: {correlation:.2f}
                """.format(**metrics)
            
            return tearsheet
            
        except Exception as e:
            logger.error(f"❌ Error generating tearsheet: {e}")
            return f"Error generating tearsheet: {e}"
