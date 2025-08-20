"""
Performance Metrics and Evaluation Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

class PerformanceMetrics:
    """
    Comprehensive performance metrics and evaluation system
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'benchmark_return': 0.10,  # 10% annual benchmark return
            'confidence_level': 0.95,
            'max_drawdown_threshold': 0.20,  # 20% maximum drawdown
            'sharpe_ratio_threshold': 1.0
        }
        
        self.performance_history = []
        self.risk_metrics = {}
        
    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        try:
            if len(prices) < 2:
                return []
            
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    return_val = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(return_val)
                else:
                    returns.append(0.0)
            
            return returns
            
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return []
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns:
                return 0.0
            
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            # Convert annual risk-free rate to period rate
            period_risk_free_rate = risk_free_rate / 252  # Assuming daily data
            
            # Calculate excess returns
            excess_returns = [r - period_risk_free_rate for r in returns]
            
            # Calculate Sharpe ratio
            avg_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            
            if std_excess_return == 0:
                return 0.0
            
            sharpe_ratio = avg_excess_return / std_excess_return
            
            # Annualize (assuming daily data)
            sharpe_ratio *= math.sqrt(252)
            
            return sharpe_ratio
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if not returns:
                return 0.0
            
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            # Convert annual risk-free rate to period rate
            period_risk_free_rate = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = [r - period_risk_free_rate for r in returns]
            
            # Calculate downside deviation
            downside_returns = [r for r in excess_returns if r < 0]
            
            if not downside_returns:
                return float('inf')  # No downside risk
            
            avg_excess_return = np.mean(excess_returns)
            downside_deviation = np.std(downside_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = avg_excess_return / downside_deviation
            
            # Annualize
            sortino_ratio *= math.sqrt(252)
            
            return sortino_ratio
            
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, prices: List[float]) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 2:
                return {'max_drawdown': 0.0, 'drawdown_duration': 0}
            
            peak = prices[0]
            max_drawdown = 0.0
            drawdown_start = 0
            drawdown_end = 0
            max_drawdown_start = 0
            max_drawdown_end = 0
            
            for i, price in enumerate(prices):
                if price > peak:
                    peak = price
                    drawdown_start = i
                else:
                    drawdown = (peak - price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        max_drawdown_start = drawdown_start
                        max_drawdown_end = i
            
            drawdown_duration = max_drawdown_end - max_drawdown_start
            
            return {
                'max_drawdown': max_drawdown,
                'drawdown_duration': drawdown_duration,
                'drawdown_start': max_drawdown_start,
                'drawdown_end': max_drawdown_end
            }
            
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0}
    
    def calculate_var(self, returns: List[float], confidence_level: float = None) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            if not returns:
                return 0.0
            
            if confidence_level is None:
                confidence_level = self.config['confidence_level']
            
            # Calculate VaR using historical simulation
            sorted_returns = sorted(returns)
            var_index = int((1 - confidence_level) * len(sorted_returns))
            
            if var_index >= len(sorted_returns):
                var_index = len(sorted_returns) - 1
            
            var = sorted_returns[var_index]
            
            return abs(var)  # Return positive VaR
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = None) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        try:
            if not returns:
                return 0.0
            
            if confidence_level is None:
                confidence_level = self.config['confidence_level']
            
            # Calculate VaR first
            var = self.calculate_var(returns, confidence_level)
            
            # Calculate CVaR (average of returns below VaR)
            tail_returns = [r for r in returns if r <= -var]
            
            if not tail_returns:
                return var
            
            cvar = np.mean(tail_returns)
            
            return abs(cvar)  # Return positive CVaR
            
        except Exception as e:
            print(f"Error calculating CVaR: {e}")
            return 0.0
    
    def calculate_beta(self, portfolio_returns: List[float], market_returns: List[float]) -> float:
        """Calculate beta relative to market"""
        try:
            if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
                return 1.0
            
            # Calculate covariance and variance
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_alpha(self, portfolio_returns: List[float], market_returns: List[float], 
                       risk_free_rate: float = None) -> float:
        """Calculate alpha (excess return)"""
        try:
            if len(portfolio_returns) != len(market_returns):
                return 0.0
            
            if risk_free_rate is None:
                risk_free_rate = self.config['risk_free_rate']
            
            # Calculate average returns
            avg_portfolio_return = np.mean(portfolio_returns)
            avg_market_return = np.mean(market_returns)
            
            # Calculate beta
            beta = self.calculate_beta(portfolio_returns, market_returns)
            
            # Convert annual risk-free rate to period rate
            period_risk_free_rate = risk_free_rate / 252
            
            # Calculate alpha
            alpha = avg_portfolio_return - (period_risk_free_rate + beta * (avg_market_return - period_risk_free_rate))
            
            # Annualize
            alpha *= 252
            
            return alpha
            
        except Exception as e:
            print(f"Error calculating alpha: {e}")
            return 0.0
    
    def calculate_information_ratio(self, portfolio_returns: List[float], 
                                  benchmark_returns: List[float]) -> float:
        """Calculate information ratio"""
        try:
            if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
                return 0.0
            
            # Calculate active returns
            active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            
            # Calculate information ratio
            avg_active_return = np.mean(active_returns)
            tracking_error = np.std(active_returns)
            
            if tracking_error == 0:
                return 0.0
            
            information_ratio = avg_active_return / tracking_error
            
            # Annualize
            information_ratio *= math.sqrt(252)
            
            return information_ratio
            
        except Exception as e:
            print(f"Error calculating information ratio: {e}")
            return 0.0
    
    def calculate_calmar_ratio(self, returns: List[float], max_drawdown: float = None) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        try:
            if not returns:
                return 0.0
            
            # Calculate annual return
            avg_return = np.mean(returns)
            annual_return = avg_return * 252
            
            # Calculate max drawdown if not provided
            if max_drawdown is None:
                prices = self._returns_to_prices(returns)
                drawdown_info = self.calculate_max_drawdown(prices)
                max_drawdown = drawdown_info['max_drawdown']
            
            if max_drawdown == 0:
                return 0.0
            
            calmar_ratio = annual_return / max_drawdown
            
            return calmar_ratio
            
        except Exception as e:
            print(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _returns_to_prices(self, returns: List[float], initial_price: float = 100.0) -> List[float]:
        """Convert returns to price series"""
        try:
            prices = [initial_price]
            for ret in returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            return prices
            
        except Exception as e:
            print(f"Error converting returns to prices: {e}")
            return [initial_price]
    
    def calculate_comprehensive_metrics(self, portfolio_returns: List[float], 
                                      market_returns: List[float] = None,
                                      benchmark_returns: List[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not portfolio_returns:
                return {}
            
            # Convert returns to prices for drawdown calculation
            prices = self._returns_to_prices(portfolio_returns)
            
            # Calculate basic metrics
            total_return = (prices[-1] - prices[0]) / prices[0]
            annual_return = total_return * 252 / len(portfolio_returns)
            volatility = np.std(portfolio_returns) * math.sqrt(252)
            
            # Calculate risk metrics
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self.calculate_sortino_ratio(portfolio_returns)
            max_drawdown_info = self.calculate_max_drawdown(prices)
            var = self.calculate_var(portfolio_returns)
            cvar = self.calculate_cvar(portfolio_returns)
            
            # Calculate relative metrics if market data available
            beta = 1.0
            alpha = 0.0
            information_ratio = 0.0
            
            if market_returns and len(market_returns) == len(portfolio_returns):
                beta = self.calculate_beta(portfolio_returns, market_returns)
                alpha = self.calculate_alpha(portfolio_returns, market_returns)
            
            if benchmark_returns and len(benchmark_returns) == len(portfolio_returns):
                information_ratio = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            # Calculate additional ratios
            calmar_ratio = self.calculate_calmar_ratio(portfolio_returns, max_drawdown_info['max_drawdown'])
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown_info['max_drawdown'],
                'drawdown_duration': max_drawdown_info['drawdown_duration'],
                'var_95': var,
                'cvar_95': cvar,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'return_per_drawdown': calmar_ratio,
                'risk_adjusted_return': sharpe_ratio,
                'downside_deviation': np.std([r for r in portfolio_returns if r < 0]) * math.sqrt(252),
                'win_rate': len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns),
                'avg_win': np.mean([r for r in portfolio_returns if r > 0]) if any(r > 0 for r in portfolio_returns) else 0,
                'avg_loss': np.mean([r for r in portfolio_returns if r < 0]) if any(r < 0 for r in portfolio_returns) else 0,
                'profit_factor': abs(sum([r for r in portfolio_returns if r > 0]) / sum([r for r in portfolio_returns if r < 0])) if any(r < 0 for r in portfolio_returns) else float('inf'),
                'timestamp': datetime.now()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating comprehensive metrics: {e}")
            return {}
    
    def evaluate_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance against thresholds"""
        try:
            evaluation = {
                'overall_rating': 'Neutral',
                'risk_rating': 'Neutral',
                'return_rating': 'Neutral',
                'recommendations': []
            }
            
            # Evaluate Sharpe ratio
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio >= self.config['sharpe_ratio_threshold']:
                evaluation['risk_rating'] = 'Good'
            elif sharpe_ratio < 0:
                evaluation['risk_rating'] = 'Poor'
            
            # Evaluate max drawdown
            max_drawdown = metrics.get('max_drawdown', 0)
            if max_drawdown <= self.config['max_drawdown_threshold']:
                evaluation['risk_rating'] = 'Good'
            elif max_drawdown > 0.3:
                evaluation['risk_rating'] = 'Poor'
            
            # Evaluate returns
            annual_return = metrics.get('annual_return', 0)
            if annual_return >= self.config['benchmark_return']:
                evaluation['return_rating'] = 'Good'
            elif annual_return < 0:
                evaluation['return_rating'] = 'Poor'
            
            # Overall rating
            if evaluation['risk_rating'] == 'Good' and evaluation['return_rating'] == 'Good':
                evaluation['overall_rating'] = 'Excellent'
            elif evaluation['risk_rating'] == 'Poor' or evaluation['return_rating'] == 'Poor':
                evaluation['overall_rating'] = 'Poor'
            
            # Generate recommendations
            if sharpe_ratio < 1.0:
                evaluation['recommendations'].append("Consider improving risk-adjusted returns")
            if max_drawdown > 0.2:
                evaluation['recommendations'].append("Implement better risk management to reduce drawdowns")
            if annual_return < 0.1:
                evaluation['recommendations'].append("Focus on improving absolute returns")
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating performance: {e}")
            return {'overall_rating': 'Error', 'recommendations': [f"Error in evaluation: {e}"]}
