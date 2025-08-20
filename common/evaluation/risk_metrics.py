"""
Risk Metrics Calculation Module
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskMetrics:
    """
    Comprehensive risk metrics calculation system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'confidence_levels': [0.95, 0.99],
            'historical_window': 252,
            'simulation_days': 1000,
            'risk_free_rate': 0.02
        }
        
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: List of return values
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            VaR value
        """
        try:
            returns_array = np.array(returns)
            
            # Historical VaR
            var_percentile = (1 - confidence_level) * 100
            historical_var = np.percentile(returns_array, var_percentile)
            
            return abs(historical_var)
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_expected_shortfall(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: List of return values
            confidence_level: Confidence level for ES calculation (default: 0.95)
            
        Returns:
            Expected Shortfall value
        """
        try:
            returns_array = np.array(returns)
            
            # Calculate VaR threshold
            var_threshold = self.calculate_var(returns, confidence_level)
            
            # Calculate expected shortfall
            tail_returns = returns_array[returns_array <= -var_threshold]
            
            if len(tail_returns) > 0:
                expected_shortfall = np.mean(tail_returns)
                return abs(expected_shortfall)
            else:
                return var_threshold
                
        except Exception as e:
            print(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            returns: List of return values
            
        Returns:
            Maximum drawdown value
        """
        try:
            returns_array = np.array(returns)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns_array)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Return maximum drawdown
            max_drawdown = np.min(drawdown)
            
            return abs(max_drawdown)
            
        except Exception as e:
            print(f"Error calculating Max Drawdown: {e}")
            return 0.0
    
    def calculate_beta(self, asset_returns: List[float], market_returns: List[float]) -> float:
        """
        Calculate Beta relative to market
        
        Args:
            asset_returns: Asset return values
            market_returns: Market return values
            
        Returns:
            Beta value
        """
        try:
            asset_array = np.array(asset_returns)
            market_array = np.array(market_returns)
            
            # Ensure same length
            min_length = min(len(asset_array), len(market_array))
            asset_array = asset_array[:min_length]
            market_array = market_array[:min_length]
            
            # Calculate covariance and variance
            covariance = np.cov(asset_array, market_array)[0, 1]
            market_variance = np.var(market_array)
            
            if market_variance > 0:
                beta = covariance / market_variance
                return beta
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Beta: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (default: from config)
            
        Returns:
            Sharpe ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            returns_array = np.array(returns)
            
            # Calculate excess returns
            excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
            
            # Calculate Sharpe ratio
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                return sharpe_ratio
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Sharpe Ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """
        Calculate Sortino Ratio
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (default: from config)
            
        Returns:
            Sortino ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            returns_array = np.array(returns)
            
            # Calculate excess returns
            excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
                return sortino_ratio
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Sortino Ratio: {e}")
            return 0.0
    
    def calculate_information_ratio(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """
        Calculate Information Ratio
        
        Args:
            portfolio_returns: Portfolio return values
            benchmark_returns: Benchmark return values
            
        Returns:
            Information ratio
        """
        try:
            portfolio_array = np.array(portfolio_returns)
            benchmark_array = np.array(benchmark_returns)
            
            # Ensure same length
            min_length = min(len(portfolio_array), len(benchmark_array))
            portfolio_array = portfolio_array[:min_length]
            benchmark_array = benchmark_array[:min_length]
            
            # Calculate active returns
            active_returns = portfolio_array - benchmark_array
            
            # Calculate information ratio
            if np.std(active_returns) > 0:
                information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
                return information_ratio
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Information Ratio: {e}")
            return 0.0
    
    def calculate_treynor_ratio(self, returns: List[float], market_returns: List[float], risk_free_rate: float = None) -> float:
        """
        Calculate Treynor Ratio
        
        Args:
            returns: Asset return values
            market_returns: Market return values
            risk_free_rate: Risk-free rate (default: from config)
            
        Returns:
            Treynor ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            # Calculate beta
            beta = self.calculate_beta(returns, market_returns)
            
            if beta != 0:
                returns_array = np.array(returns)
                excess_return = np.mean(returns_array) - risk_free_rate / 252
                treynor_ratio = excess_return / beta * 252
                return treynor_ratio
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Treynor Ratio: {e}")
            return 0.0
    
    def calculate_calmar_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """
        Calculate Calmar Ratio
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (default: from config)
            
        Returns:
            Calmar ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            returns_array = np.array(returns)
            
            # Calculate annualized return
            annual_return = np.mean(returns_array) * 252 - risk_free_rate
            
            # Calculate maximum drawdown
            max_dd = self.calculate_max_drawdown(returns)
            
            if max_dd > 0:
                calmar_ratio = annual_return / max_dd
                return calmar_ratio
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating Calmar Ratio: {e}")
            return 0.0
    
    def calculate_all_metrics(self, returns: List[float], market_returns: List[float] = None) -> Dict[str, float]:
        """
        Calculate all risk metrics
        
        Args:
            returns: List of return values
            market_returns: Market return values (optional)
            
        Returns:
            Dictionary of all risk metrics
        """
        try:
            metrics = {}
            
            # Basic risk metrics
            metrics['var_95'] = self.calculate_var(returns, 0.95)
            metrics['var_99'] = self.calculate_var(returns, 0.99)
            metrics['expected_shortfall_95'] = self.calculate_expected_shortfall(returns, 0.95)
            metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
            
            # Market-dependent metrics
            if market_returns is not None:
                metrics['beta'] = self.calculate_beta(returns, market_returns)
                metrics['information_ratio'] = self.calculate_information_ratio(returns, market_returns)
                metrics['treynor_ratio'] = self.calculate_treynor_ratio(returns, market_returns)
            
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating all metrics: {e}")
            return {}
