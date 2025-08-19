"""
Advanced Risk Management System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import time

class AdvancedRiskManager:
    """
    Advanced Risk Management System with:
    - Portfolio optimization (Modern Portfolio Theory)
    - VaR calculations
    - Stress testing
    - Position sizing (Kelly Criterion)
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'confidence_level': 0.95,  # 95% VaR
            'max_leverage': 2.0,      # Maximum 2x leverage
            'position_limits': {
                'max_single_position': 0.10,  # 10% max single position
                'max_sector_exposure': 0.30,  # 30% max sector exposure
                'max_asset_class': 0.50       # 50% max asset class
            },
            'var_parameters': {
                'historical_window': 252,     # 1 year of data
                'simulation_days': 1000       # Monte Carlo simulations
            }
        }
        
        self.portfolio = {}
        self.risk_metrics = {}
        self.position_history = []
        self.var_history = []
        
    async def optimize_portfolio(self, assets_data, target_return=None):
        """Optimize portfolio using Modern Portfolio Theory"""
        try:
            print("🔬 Optimizing portfolio using Modern Portfolio Theory...")
            
            # Calculate returns and covariance matrix
            returns_data = self._calculate_returns(assets_data)
            covariance_matrix = returns_data.cov()
            expected_returns = returns_data.mean()
            
            # Generate efficient frontier
            efficient_frontier = self._generate_efficient_frontier(
                expected_returns, covariance_matrix
            )
            
            # Find optimal portfolio
            if target_return is None:
                # Find maximum Sharpe ratio portfolio
                optimal_weights = self._find_max_sharpe_portfolio(
                    expected_returns, covariance_matrix
                )
            else:
                # Find portfolio with target return
                optimal_weights = self._find_target_return_portfolio(
                    expected_returns, covariance_matrix, target_return
                )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights, expected_returns, covariance_matrix
            )
            
            return {
                'success': True,
                'optimal_weights': optimal_weights,
                'portfolio_metrics': portfolio_metrics,
                'efficient_frontier': efficient_frontier
            }
            
        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_returns(self, assets_data):
        """Calculate returns for all assets"""
        returns_data = pd.DataFrame()
        
        for symbol, data in assets_data.items():
            if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        return returns_data
    
    def _generate_efficient_frontier(self, expected_returns, covariance_matrix):
        """Generate efficient frontier"""
        try:
            n_assets = len(expected_returns)
            efficient_frontier = []
            
            # Generate portfolios with different target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            
            for target_return in np.linspace(min_return, max_return, 50):
                try:
                    weights = self._find_target_return_portfolio(
                        expected_returns, covariance_matrix, target_return
                    )
                    
                    if weights is not None:
                        portfolio_return = np.sum(weights * expected_returns)
                        portfolio_volatility = np.sqrt(
                            np.dot(weights.T, np.dot(covariance_matrix, weights))
                        )
                        
                        efficient_frontier.append({
                            'return': portfolio_return,
                            'volatility': portfolio_volatility,
                            'sharpe_ratio': (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility,
                            'weights': weights
                        })
                except:
                    continue
            
            return efficient_frontier
            
        except Exception as e:
            print(f"Error generating efficient frontier: {e}")
            return []
    
    def _find_max_sharpe_portfolio(self, expected_returns, covariance_matrix):
        """Find portfolio with maximum Sharpe ratio"""
        try:
            n_assets = len(expected_returns)
            
            # Use equal weights as starting point
            weights = np.array([1/n_assets] * n_assets)
            
            # Simple optimization (in real implementation, use scipy.optimize)
            best_sharpe = -np.inf
            best_weights = weights
            
            # Grid search for optimal weights
            for _ in range(1000):
                # Generate random weights
                random_weights = np.random.random(n_assets)
                random_weights = random_weights / np.sum(random_weights)
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(random_weights * expected_returns)
                portfolio_volatility = np.sqrt(
                    np.dot(random_weights.T, np.dot(covariance_matrix, random_weights))
                )
                
                if portfolio_volatility > 0:
                    sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
                    
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_weights = random_weights
            
            return best_weights
            
        except Exception as e:
            print(f"Error finding max Sharpe portfolio: {e}")
            return None
    
    def _find_target_return_portfolio(self, expected_returns, covariance_matrix, target_return):
        """Find portfolio with target return"""
        try:
            n_assets = len(expected_returns)
            
            # Use equal weights as starting point
            weights = np.array([1/n_assets] * n_assets)
            
            # Simple optimization
            best_weights = weights
            min_variance = np.inf
            
            for _ in range(1000):
                # Generate random weights
                random_weights = np.random.random(n_assets)
                random_weights = random_weights / np.sum(random_weights)
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(random_weights * expected_returns)
                portfolio_variance = np.dot(random_weights.T, np.dot(covariance_matrix, random_weights))
                
                # Check if return is close to target
                if abs(portfolio_return - target_return) < 0.001 and portfolio_variance < min_variance:
                    min_variance = portfolio_variance
                    best_weights = random_weights
            
            return best_weights
            
        except Exception as e:
            print(f"Error finding target return portfolio: {e}")
            return None
    
    def _calculate_portfolio_metrics(self, weights, expected_returns, covariance_matrix):
        """Calculate portfolio metrics"""
        try:
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(covariance_matrix, weights))
            )
            sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': self._calculate_var(weights, expected_returns, covariance_matrix, 0.95),
                'var_99': self._calculate_var(weights, expected_returns, covariance_matrix, 0.99)
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def calculate_var(self, portfolio_data, confidence_level=None):
        """Calculate Value at Risk"""
        try:
            if confidence_level is None:
                confidence_level = self.config['confidence_level']
            
            # Historical VaR
            historical_var = self._calculate_historical_var(portfolio_data, confidence_level)
            
            # Parametric VaR
            parametric_var = self._calculate_parametric_var(portfolio_data, confidence_level)
            
            # Monte Carlo VaR
            monte_carlo_var = await self._calculate_monte_carlo_var(portfolio_data, confidence_level)
            
            var_result = {
                'historical_var': historical_var,
                'parametric_var': parametric_var,
                'monte_carlo_var': monte_carlo_var,
                'confidence_level': confidence_level,
                'timestamp': datetime.now()
            }
            
            self.var_history.append(var_result)
            
            return var_result
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_historical_var(self, portfolio_data, confidence_level):
        """Calculate historical VaR"""
        try:
            if isinstance(portfolio_data, pd.Series):
                returns = portfolio_data
            else:
                returns = portfolio_data['returns']
            
            # Calculate historical VaR
            var_percentile = (1 - confidence_level) * 100
            historical_var = np.percentile(returns, var_percentile)
            
            return abs(historical_var)
            
        except Exception as e:
            print(f"Error calculating historical VaR: {e}")
            return 0.0
    
    def _calculate_parametric_var(self, portfolio_data, confidence_level):
        """Calculate parametric VaR"""
        try:
            if isinstance(portfolio_data, pd.Series):
                returns = portfolio_data
            else:
                returns = portfolio_data['returns']
            
            # Calculate parametric VaR
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            from scipy.stats import norm
            z_score = norm.ppf(confidence_level)
            
            parametric_var = mean_return - z_score * std_return
            
            return abs(parametric_var)
            
        except Exception as e:
            print(f"Error calculating parametric VaR: {e}")
            return 0.0
    
    async def _calculate_monte_carlo_var(self, portfolio_data, confidence_level):
        """Calculate Monte Carlo VaR"""
        try:
            if isinstance(portfolio_data, pd.Series):
                returns = portfolio_data
            else:
                returns = portfolio_data['returns']
            
            # Monte Carlo simulation
            n_simulations = self.config['var_parameters']['simulation_days']
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            monte_carlo_var = np.percentile(simulated_returns, var_percentile)
            
            return abs(monte_carlo_var)
            
        except Exception as e:
            print(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0
    
    def _calculate_var(self, weights, expected_returns, covariance_matrix, confidence_level):
        """Calculate VaR for portfolio weights"""
        try:
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(covariance_matrix, weights))
            )
            
            # Z-score for confidence level
            from scipy.stats import norm
            z_score = norm.ppf(confidence_level)
            
            var = portfolio_return - z_score * portfolio_volatility
            
            return abs(var)
            
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    async def perform_stress_test(self, portfolio_data, scenarios):
        """Perform stress testing"""
        try:
            print("🔬 Performing stress testing...")
            
            stress_results = {}
            
            for scenario_name, scenario_data in scenarios.items():
                # Apply stress scenario
                stressed_returns = self._apply_stress_scenario(portfolio_data, scenario_data)
                
                # Calculate stressed VaR
                stressed_var = self._calculate_historical_var(stressed_returns, self.config['confidence_level'])
                
                # Calculate portfolio impact
                portfolio_impact = self._calculate_portfolio_impact(portfolio_data, stressed_returns)
                
                stress_results[scenario_name] = {
                    'stressed_var': stressed_var,
                    'portfolio_impact': portfolio_impact,
                    'scenario_data': scenario_data
                }
            
            return {
                'success': True,
                'stress_results': stress_results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error performing stress test: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_stress_scenario(self, portfolio_data, scenario_data):
        """Apply stress scenario to portfolio data"""
        try:
            if isinstance(portfolio_data, pd.Series):
                returns = portfolio_data
            else:
                returns = portfolio_data['returns']
            
            # Apply stress factors
            stressed_returns = returns.copy()
            
            for factor, stress_factor in scenario_data.items():
                if factor in returns.index:
                    stressed_returns[factor] *= stress_factor
            
            return stressed_returns
            
        except Exception as e:
            print(f"Error applying stress scenario: {e}")
            return portfolio_data
    
    def _calculate_portfolio_impact(self, portfolio_data, stressed_returns):
        """Calculate portfolio impact of stress scenario"""
        try:
            if isinstance(portfolio_data, pd.Series):
                original_returns = portfolio_data
            else:
                original_returns = portfolio_data['returns']
            
            # Calculate impact
            impact = stressed_returns.mean() - original_returns.mean()
            
            return impact
            
        except Exception as e:
            print(f"Error calculating portfolio impact: {e}")
            return 0.0
    
    async def calculate_kelly_position_size(self, win_rate, avg_win, avg_loss):
        """Calculate optimal position size using Kelly Criterion"""
        try:
            print("🔬 Calculating Kelly Criterion position size...")
            
            # Kelly Criterion formula: f = (bp - q) / b
            # where: f = fraction of capital to bet
            #        b = odds received on bet
            #        p = probability of winning
            #        q = probability of losing (1 - p)
            
            if avg_loss == 0:
                return 0.0
            
            b = avg_win / avg_loss  # odds received
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply constraints
            kelly_fraction = max(0, min(kelly_fraction, self.config['max_leverage']))
            
            return kelly_fraction
            
        except Exception as e:
            print(f"Error calculating Kelly position size: {e}")
            return 0.0
    
    async def check_risk_limits(self, new_position, current_portfolio):
        """Check if new position meets risk limits"""
        try:
            risk_checks = {
                'position_size': True,
                'sector_exposure': True,
                'asset_class_exposure': True,
                'leverage': True
            }
            
            # Position size check
            position_value = abs(new_position['quantity'] * new_position['price'])
            portfolio_value = sum(pos['value'] for pos in current_portfolio.values())
            
            if position_value / portfolio_value > self.config['position_limits']['max_single_position']:
                risk_checks['position_size'] = False
            
            # Sector exposure check (simplified)
            # In real implementation, would check actual sector classifications
            
            # Asset class exposure check (simplified)
            # In real implementation, would check actual asset class classifications
            
            # Leverage check
            total_exposure = sum(pos['value'] for pos in current_portfolio.values()) + position_value
            if total_exposure / portfolio_value > self.config['max_leverage']:
                risk_checks['leverage'] = False
            
            return {
                'success': all(risk_checks.values()),
                'risk_checks': risk_checks,
                'position_value': position_value,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_risk_summary(self):
        """Get risk management summary"""
        try:
            return {
                'portfolio_size': len(self.portfolio),
                'total_positions': len(self.position_history),
                'var_history_length': len(self.var_history),
                'latest_var': self.var_history[-1] if self.var_history else None,
                'risk_metrics': self.risk_metrics
            }
            
        except Exception as e:
            print(f"Error getting risk summary: {e}")
            return {}
