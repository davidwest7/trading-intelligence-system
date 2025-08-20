"""
Backtest Engine for Trading Strategy Evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'initial_capital': 1000000,  # $1M initial capital
            'commission_rate': 0.001,    # 0.1% commission
            'slippage': 0.0005,          # 0.05% slippage
            'risk_free_rate': 0.02,      # 2% annual risk-free rate
            'benchmark': 'SPY'           # Default benchmark
        }
        
        self.portfolio_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
    def run_backtest(self, strategy_returns: List[float], benchmark_returns: List[float] = None) -> Dict[str, Any]:
        """
        Run backtest on strategy returns
        
        Args:
            strategy_returns: List of strategy return values
            benchmark_returns: List of benchmark return values (optional)
            
        Returns:
            Backtest results dictionary
        """
        try:
            print("🔬 Running backtest...")
            
            # Convert to numpy arrays
            strategy_array = np.array(strategy_returns)
            
            # Calculate portfolio values
            portfolio_values = self._calculate_portfolio_values(strategy_array)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(strategy_array, benchmark_returns)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(strategy_array, benchmark_returns)
            
            # Calculate drawdown analysis
            drawdown_analysis = self._calculate_drawdown_analysis(portfolio_values)
            
            # Compile results
            results = {
                'total_return': performance_metrics['total_return'],
                'annualized_return': performance_metrics['annualized_return'],
                'volatility': risk_metrics['volatility'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown'],
                'calmar_ratio': performance_metrics['calmar_ratio'],
                'win_rate': performance_metrics['win_rate'],
                'profit_factor': performance_metrics['profit_factor'],
                'portfolio_values': portfolio_values,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'drawdown_analysis': drawdown_analysis,
                'trade_count': len(self.trade_history),
                'backtest_duration': len(strategy_returns)
            }
            
            self.performance_metrics = results
            
            return results
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'error': str(e)
            }
    
    def _calculate_portfolio_values(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio values over time
        
        Args:
            returns: Array of return values
            
        Returns:
            Array of portfolio values
        """
        try:
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns)
            
            # Calculate portfolio values
            portfolio_values = self.config['initial_capital'] * cumulative_returns
            
            return portfolio_values
            
        except Exception as e:
            print(f"Error calculating portfolio values: {e}")
            return np.array([self.config['initial_capital']])
    
    def _calculate_performance_metrics(self, returns: np.ndarray, benchmark_returns: List[float] = None) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            returns: Array of return values
            benchmark_returns: Benchmark return values (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}
            
            # Total return
            total_return = np.prod(1 + returns) - 1
            metrics['total_return'] = total_return
            
            # Annualized return (assuming daily data)
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            metrics['annualized_return'] = annualized_return
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            metrics['volatility'] = volatility
            
            # Sharpe ratio
            excess_returns = returns - self.config['risk_free_rate'] / 252
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0.0
            metrics['sortino_ratio'] = sortino_ratio
            
            # Win rate
            winning_days = np.sum(returns > 0)
            total_days = len(returns)
            win_rate = winning_days / total_days if total_days > 0 else 0.0
            metrics['win_rate'] = win_rate
            
            # Profit factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            metrics['max_drawdown'] = abs(max_drawdown)
            
            # Calmar ratio
            if abs(max_drawdown) > 0:
                calmar_ratio = annualized_return / abs(max_drawdown)
            else:
                calmar_ratio = 0.0
            metrics['calmar_ratio'] = calmar_ratio
            
            # Information ratio (if benchmark provided)
            if benchmark_returns is not None:
                benchmark_array = np.array(benchmark_returns)
                min_length = min(len(returns), len(benchmark_array))
                active_returns = returns[:min_length] - benchmark_array[:min_length]
                
                if np.std(active_returns) > 0:
                    information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
                else:
                    information_ratio = 0.0
                metrics['information_ratio'] = information_ratio
                
                # Alpha and Beta
                covariance = np.cov(returns[:min_length], benchmark_array[:min_length])[0, 1]
                benchmark_variance = np.var(benchmark_array[:min_length])
                
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                    alpha = np.mean(returns[:min_length]) - beta * np.mean(benchmark_array[:min_length])
                else:
                    beta = 0.0
                    alpha = 0.0
                
                metrics['alpha'] = alpha * 252  # Annualized alpha
                metrics['beta'] = beta
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }
    
    def _calculate_risk_metrics(self, returns: np.ndarray, benchmark_returns: List[float] = None) -> Dict[str, float]:
        """
        Calculate risk metrics
        
        Args:
            returns: Array of return values
            benchmark_returns: Benchmark return values (optional)
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            metrics = {}
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            metrics['var_95'] = abs(var_95)
            metrics['var_99'] = abs(var_99)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns_95 = returns[returns <= var_95]
            tail_returns_99 = returns[returns <= var_99]
            
            if len(tail_returns_95) > 0:
                expected_shortfall_95 = np.mean(tail_returns_95)
            else:
                expected_shortfall_95 = var_95
                
            if len(tail_returns_99) > 0:
                expected_shortfall_99 = np.mean(tail_returns_99)
            else:
                expected_shortfall_99 = var_99
                
            metrics['expected_shortfall_95'] = abs(expected_shortfall_95)
            metrics['expected_shortfall_99'] = abs(expected_shortfall_99)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            metrics['max_drawdown'] = abs(max_drawdown)
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            metrics['volatility'] = volatility
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
            else:
                downside_deviation = 0.0
            metrics['downside_deviation'] = downside_deviation
            
            # Skewness and Kurtosis
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)
            metrics['skewness'] = skewness
            metrics['kurtosis'] = kurtosis
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'expected_shortfall_95': 0.0,
                'expected_shortfall_99': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'downside_deviation': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
    
    def _calculate_drawdown_analysis(self, portfolio_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed drawdown analysis
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Dictionary of drawdown analysis
        """
        try:
            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)
            
            # Calculate drawdown
            drawdown = (portfolio_values - running_max) / running_max
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, dd in enumerate(drawdown):
                if dd < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_idx = i
                elif dd >= 0 and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    drawdown_periods.append({
                        'start': start_idx,
                        'end': i,
                        'duration': i - start_idx,
                        'max_drawdown': np.min(drawdown[start_idx:i])
                    })
            
            # Handle ongoing drawdown
            if in_drawdown:
                drawdown_periods.append({
                    'start': start_idx,
                    'end': len(drawdown) - 1,
                    'duration': len(drawdown) - 1 - start_idx,
                    'max_drawdown': np.min(drawdown[start_idx:])
                })
            
            # Calculate statistics
            if drawdown_periods:
                durations = [period['duration'] for period in drawdown_periods]
                max_drawdowns = [abs(period['max_drawdown']) for period in drawdown_periods]
                
                analysis = {
                    'total_periods': len(drawdown_periods),
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'avg_drawdown': np.mean(max_drawdowns),
                    'max_drawdown': np.max(max_drawdowns),
                    'drawdown_periods': drawdown_periods
                }
            else:
                analysis = {
                    'total_periods': 0,
                    'avg_duration': 0,
                    'max_duration': 0,
                    'avg_drawdown': 0,
                    'max_drawdown': 0,
                    'drawdown_periods': []
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error calculating drawdown analysis: {e}")
            return {
                'total_periods': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'avg_drawdown': 0,
                'max_drawdown': 0,
                'drawdown_periods': []
            }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                skewness = np.mean(((returns - mean_return) / std_return) ** 3)
                return skewness
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
                return kurtosis
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def add_trade(self, trade: Dict[str, Any]):
        """
        Add a trade to the trade history
        
        Args:
            trade: Trade dictionary with details
        """
        self.trade_history.append(trade)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary
        
        Returns:
            Summary dictionary
        """
        if not self.performance_metrics:
            return {'error': 'No backtest results available'}
        
        return {
            'total_return': self.performance_metrics.get('total_return', 0.0),
            'annualized_return': self.performance_metrics.get('annualized_return', 0.0),
            'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0),
            'win_rate': self.performance_metrics.get('win_rate', 0.0),
            'trade_count': self.performance_metrics.get('trade_count', 0),
            'backtest_duration': self.performance_metrics.get('backtest_duration', 0)
        }
