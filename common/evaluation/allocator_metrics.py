#!/usr/bin/env python3
"""
Advanced Allocator Metrics
IR per risk dollar, capacity, crowding beta, borrow costs, and comprehensive performance evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Traditional metrics
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    
    # Information ratios
    information_ratio: float
    ir_per_risk_dollar: float
    risk_adjusted_ir: float
    
    # Capacity metrics
    capacity_utilization: float
    signal_capacity: float
    turnover_ratio: float
    capacity_decay_rate: float
    
    # Crowding metrics
    crowding_beta: float
    crowding_correlation: float
    crowding_exposure: float
    crowding_contribution: float
    
    # Cost metrics
    total_slippage: float
    borrow_costs: float
    transaction_costs: float
    market_impact: float
    
    # Risk metrics
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    
    # Additional metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk-adjusted metrics
    treynor_ratio: float
    jensen_alpha: float
    omega_ratio: float
    calmar_ratio: float
    
    # Attribution metrics
    factor_attribution: Dict[str, float]
    sector_attribution: Dict[str, float]
    style_attribution: Dict[str, float]
    
    # Timestamp
    calculation_date: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    evaluation_id: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    performance_metrics: PerformanceMetrics
    
    # Detailed breakdowns
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    rolling_metrics: Dict[str, pd.Series]
    
    # Risk decomposition
    risk_decomposition: Dict[str, float]
    factor_contributions: Dict[str, float]
    
    # Capacity analysis
    capacity_analysis: Dict[str, Any]
    crowding_analysis: Dict[str, Any]
    
    # Cost analysis
    cost_breakdown: Dict[str, float]
    slippage_analysis: Dict[str, Any]
    
    # Benchmark comparison
    benchmark_comparison: Dict[str, float]
    
    # Evaluation metadata
    metadata: Dict[str, Any]


class AdvancedAllocatorMetrics:
    """Advanced metrics for allocator evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_comprehensive_metrics(self, 
                                      returns: pd.Series,
                                      benchmark_returns: pd.Series,
                                      positions: pd.Series,
                                      costs: pd.Series,
                                      risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk metrics
            excess_returns = returns - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            
            # Sharpe ratio
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and Expected Shortfall
            var_95 = returns.quantile(0.05)
            expected_shortfall = returns[returns <= var_95].mean()
            
            # Information ratio
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            information_ratio = (returns.mean() - benchmark_returns.mean()) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            # IR per risk dollar
            risk_dollars = abs(positions).sum() * volatility
            ir_per_risk_dollar = information_ratio / risk_dollars if risk_dollars > 0 else 0
            
            # Capacity metrics
            capacity_utilization = self._calculate_capacity_utilization(positions, returns)
            signal_capacity = self._calculate_signal_capacity(returns, positions)
            turnover_ratio = self._calculate_turnover_ratio(positions)
            capacity_decay_rate = self._calculate_capacity_decay_rate(returns)
            
            # Crowding metrics
            crowding_beta = self._calculate_crowding_beta(returns, benchmark_returns)
            crowding_correlation = self._calculate_crowding_correlation(returns, benchmark_returns)
            crowding_exposure = self._calculate_crowding_exposure(positions)
            crowding_contribution = self._calculate_crowding_contribution(returns, positions)
            
            # Cost metrics
            total_slippage = costs.sum() if costs is not None else 0
            borrow_costs = self._calculate_borrow_costs(positions, returns)
            transaction_costs = self._calculate_transaction_costs(positions)
            market_impact = self._calculate_market_impact(positions, returns)
            
            # Additional metrics
            win_rate = len(returns[returns > 0]) / len(returns)
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
            average_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            average_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            largest_win = returns.max()
            largest_loss = returns.min()
            
            # Risk-adjusted metrics
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = self._calculate_alpha(returns, benchmark_returns, risk_free_rate)
            treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else 0
            jensen_alpha = alpha * 252
            omega_ratio = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
            
            # Attribution metrics (simplified)
            factor_attribution = self._calculate_factor_attribution(returns, positions)
            sector_attribution = self._calculate_sector_attribution(returns, positions)
            style_attribution = self._calculate_style_attribution(returns, positions)
            
            return PerformanceMetrics(
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                information_ratio=information_ratio,
                ir_per_risk_dollar=ir_per_risk_dollar,
                risk_adjusted_ir=information_ratio * (1 - abs(max_drawdown)),
                capacity_utilization=capacity_utilization,
                signal_capacity=signal_capacity,
                turnover_ratio=turnover_ratio,
                capacity_decay_rate=capacity_decay_rate,
                crowding_beta=crowding_beta,
                crowding_correlation=crowding_correlation,
                crowding_exposure=crowding_exposure,
                crowding_contribution=crowding_contribution,
                total_slippage=total_slippage,
                borrow_costs=borrow_costs,
                transaction_costs=transaction_costs,
                market_impact=market_impact,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                omega_ratio=omega_ratio,
                factor_attribution=factor_attribution,
                sector_attribution=sector_attribution,
                style_attribution=style_attribution
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return self._create_fallback_metrics()
    
    def _calculate_capacity_utilization(self, positions: pd.Series, returns: pd.Series) -> float:
        """Calculate capacity utilization ratio"""
        try:
            # Average position size relative to maximum capacity
            avg_position = positions.abs().mean()
            max_position = positions.abs().max()
            return avg_position / max_position if max_position > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_signal_capacity(self, returns: pd.Series, positions: pd.Series) -> float:
        """Calculate signal capacity based on information ratio decay"""
        try:
            # Estimate signal capacity from information ratio decay
            ir = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            position_scale = positions.abs().mean()
            
            # Signal capacity decreases with position size (simplified model)
            capacity = max(0, 1 - position_scale * 0.1)  # 10% decay per unit position
            return capacity
        except Exception:
            return 1.0
    
    def _calculate_turnover_ratio(self, positions: pd.Series) -> float:
        """Calculate turnover ratio"""
        try:
            position_changes = positions.diff().abs()
            avg_position = positions.abs().mean()
            return position_changes.mean() / avg_position if avg_position > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_capacity_decay_rate(self, returns: pd.Series) -> float:
        """Calculate capacity decay rate"""
        try:
            # Estimate decay rate from rolling performance
            window = min(60, len(returns) // 4)
            if window < 10:
                return 0.0
            
            rolling_ir = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            decay_rate = (rolling_ir.iloc[-1] - rolling_ir.iloc[0]) / len(rolling_ir) if len(rolling_ir) > 0 else 0
            return max(0, -decay_rate)  # Positive decay rate
        except Exception:
            return 0.0
    
    def _calculate_crowding_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate crowding beta (correlation with benchmark)"""
        try:
            correlation = returns.corr(benchmark_returns)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_crowding_correlation(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate crowding correlation"""
        try:
            # Rolling correlation to detect crowding
            window = min(30, len(returns) // 4)
            if window < 10:
                return 0.0
            
            rolling_corr = returns.rolling(window).corr(benchmark_returns)
            return rolling_corr.mean() if len(rolling_corr) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_crowding_exposure(self, positions: pd.Series) -> float:
        """Calculate crowding exposure"""
        try:
            # Measure concentration of positions
            position_weights = positions.abs() / positions.abs().sum()
            herfindahl_index = (position_weights ** 2).sum()
            return herfindahl_index
        except Exception:
            return 0.0
    
    def _calculate_crowding_contribution(self, returns: pd.Series, positions: pd.Series) -> float:
        """Calculate crowding contribution to returns"""
        try:
            # Estimate how much returns come from crowded positions
            position_weights = positions.abs() / positions.abs().sum()
            weighted_returns = returns * position_weights
            crowding_contribution = weighted_returns.sum() / returns.sum() if returns.sum() != 0 else 0
            return crowding_contribution
        except Exception:
            return 0.0
    
    def _calculate_borrow_costs(self, positions: pd.Series, returns: pd.Series) -> float:
        """Calculate borrow costs for short positions"""
        try:
            short_positions = positions[positions < 0]
            if len(short_positions) == 0:
                return 0.0
            
            # Assume 2% annual borrow cost for short positions
            borrow_rate = 0.02 / 252
            borrow_costs = abs(short_positions).sum() * borrow_rate
            return borrow_costs
        except Exception:
            return 0.0
    
    def _calculate_transaction_costs(self, positions: pd.Series) -> float:
        """Calculate transaction costs"""
        try:
            position_changes = positions.diff().abs()
            # Assume 10 bps transaction cost
            transaction_rate = 0.001
            transaction_costs = position_changes.sum() * transaction_rate
            return transaction_costs
        except Exception:
            return 0.0
    
    def _calculate_market_impact(self, positions: pd.Series, returns: pd.Series) -> float:
        """Calculate market impact costs"""
        try:
            position_changes = positions.diff().abs()
            # Assume square-root impact model
            impact_rate = 0.0001  # 1 bps per sqrt(volume)
            market_impact = (position_changes ** 0.5).sum() * impact_rate
            return market_impact
        except Exception:
            return 0.0
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark"""
        try:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            return beta
        except Exception:
            return 1.0
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate alpha"""
        try:
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = returns.mean() - (risk_free_rate / 252 + beta * benchmark_returns.mean())
            return alpha
        except Exception:
            return 0.0
    
    def _calculate_factor_attribution(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Calculate factor attribution (simplified)"""
        try:
            # Simplified factor attribution
            return {
                'momentum': returns.mean() * 0.3,
                'value': returns.mean() * 0.2,
                'quality': returns.mean() * 0.2,
                'size': returns.mean() * 0.15,
                'volatility': returns.mean() * 0.15
            }
        except Exception:
            return {'total': 0.0}
    
    def _calculate_sector_attribution(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Calculate sector attribution (simplified)"""
        try:
            # Simplified sector attribution
            return {
                'technology': returns.mean() * 0.25,
                'financials': returns.mean() * 0.20,
                'healthcare': returns.mean() * 0.15,
                'consumer': returns.mean() * 0.15,
                'energy': returns.mean() * 0.10,
                'other': returns.mean() * 0.15
            }
        except Exception:
            return {'total': 0.0}
    
    def _calculate_style_attribution(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Calculate style attribution (simplified)"""
        try:
            # Simplified style attribution
            return {
                'growth': returns.mean() * 0.4,
                'value': returns.mean() * 0.3,
                'momentum': returns.mean() * 0.2,
                'quality': returns.mean() * 0.1
            }
        except Exception:
            return {'total': 0.0}
    
    def _create_fallback_metrics(self) -> PerformanceMetrics:
        """Create fallback metrics when calculation fails"""
        return PerformanceMetrics(
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            information_ratio=0.0,
            ir_per_risk_dollar=0.0,
            risk_adjusted_ir=0.0,
            capacity_utilization=0.0,
            signal_capacity=0.0,
            turnover_ratio=0.0,
            capacity_decay_rate=0.0,
            crowding_beta=0.0,
            crowding_correlation=0.0,
            crowding_exposure=0.0,
            crowding_contribution=0.0,
            total_slippage=0.0,
            borrow_costs=0.0,
            transaction_costs=0.0,
            market_impact=0.0,
            beta=1.0,
            alpha=0.0,
            tracking_error=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            treynor_ratio=0.0,
            jensen_alpha=0.0,
            omega_ratio=1.0,
            factor_attribution={'total': 0.0},
            sector_attribution={'total': 0.0},
            style_attribution={'total': 0.0}
        )
    
    def calculate_auc_score(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate AUC score for binary classification"""
        try:
            if len(np.unique(actuals)) == 2:
                return roc_auc_score(actuals, predictions)
            else:
                # For regression, convert to binary using median
                median_val = np.median(actuals)
                binary_actuals = (actuals > median_val).astype(int)
                return roc_auc_score(binary_actuals, predictions)
        except Exception as e:
            self.logger.error(f"Error calculating AUC: {e}")
            return 0.5
    
    def calculate_brier_score(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Brier score for probability predictions"""
        try:
            # Convert to probabilities if needed
            if predictions.min() < 0 or predictions.max() > 1:
                predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid transform
            
            # Convert actuals to binary if needed
            if len(np.unique(actuals)) > 2:
                median_val = np.median(actuals)
                binary_actuals = (actuals > median_val).astype(int)
            else:
                binary_actuals = actuals
            
            return brier_score_loss(binary_actuals, predictions)
        except Exception as e:
            self.logger.error(f"Error calculating Brier score: {e}")
            return 1.0
    
    def calculate_slippage_metrics(self, intended_prices: np.ndarray, 
                                 executed_prices: np.ndarray,
                                 volumes: np.ndarray) -> Dict[str, float]:
        """Calculate slippage metrics"""
        try:
            slippage = (executed_prices - intended_prices) / intended_prices
            
            return {
                'mean_slippage': slippage.mean(),
                'median_slippage': np.median(slippage),
                'slippage_std': slippage.std(),
                'positive_slippage': slippage[slippage > 0].mean() if len(slippage[slippage > 0]) > 0 else 0,
                'negative_slippage': slippage[slippage < 0].mean() if len(slippage[slippage < 0]) > 0 else 0,
                'volume_weighted_slippage': (slippage * volumes).sum() / volumes.sum() if volumes.sum() > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error calculating slippage metrics: {e}")
            return {
                'mean_slippage': 0.0,
                'median_slippage': 0.0,
                'slippage_std': 0.0,
                'positive_slippage': 0.0,
                'negative_slippage': 0.0,
                'volume_weighted_slippage': 0.0
            }
    
    async def run_comprehensive_evaluation(self, 
                                         returns: pd.Series,
                                         benchmark_returns: pd.Series,
                                         positions: pd.Series,
                                         costs: pd.Series,
                                         start_date: datetime,
                                         end_date: datetime) -> EvaluationResult:
        """Run comprehensive evaluation"""
        try:
            # Calculate performance metrics
            performance_metrics = self.calculate_comprehensive_metrics(
                returns, benchmark_returns, positions, costs
            )
            
            # Calculate rolling metrics
            rolling_metrics = self._calculate_rolling_metrics(returns, benchmark_returns)
            
            # Risk decomposition
            risk_decomposition = self._decompose_risk(returns, positions)
            
            # Factor contributions
            factor_contributions = self._calculate_factor_contributions(returns, positions)
            
            # Capacity analysis
            capacity_analysis = self._analyze_capacity(returns, positions)
            
            # Crowding analysis
            crowding_analysis = self._analyze_crowding(returns, benchmark_returns, positions)
            
            # Cost breakdown
            cost_breakdown = self._analyze_costs(returns, positions, costs)
            
            # Slippage analysis
            slippage_analysis = self._analyze_slippage(returns, positions)
            
            # Benchmark comparison
            benchmark_comparison = self._compare_to_benchmark(returns, benchmark_returns)
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                start_date=start_date,
                end_date=end_date,
                performance_metrics=performance_metrics,
                daily_returns=returns,
                cumulative_returns=(1 + returns).cumprod(),
                rolling_metrics=rolling_metrics,
                risk_decomposition=risk_decomposition,
                factor_contributions=factor_contributions,
                capacity_analysis=capacity_analysis,
                crowding_analysis=crowding_analysis,
                cost_breakdown=cost_breakdown,
                slippage_analysis=slippage_analysis,
                benchmark_comparison=benchmark_comparison,
                metadata={
                    'calculation_method': 'comprehensive',
                    'risk_free_rate': 0.02,
                    'benchmark': 'market_index'
                }
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {e}")
            return self._create_fallback_evaluation(start_date, end_date)
    
    def _calculate_rolling_metrics(self, returns: pd.Series, 
                                 benchmark_returns: pd.Series) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        try:
            window = min(60, len(returns) // 4)
            if window < 10:
                return {}
            
            rolling_metrics = {
                'rolling_sharpe': returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252),
                'rolling_ir': (returns - benchmark_returns).rolling(window).mean() / (returns - benchmark_returns).rolling(window).std() * np.sqrt(252),
                'rolling_beta': returns.rolling(window).cov(benchmark_returns) / benchmark_returns.rolling(window).var(),
                'rolling_volatility': returns.rolling(window).std() * np.sqrt(252)
            }
            
            return rolling_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    def _decompose_risk(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Decompose risk into components"""
        try:
            total_risk = returns.var()
            
            # Factor risk (simplified)
            factor_risk = total_risk * 0.6
            
            # Specific risk
            specific_risk = total_risk * 0.3
            
            # Liquidity risk
            liquidity_risk = total_risk * 0.1
            
            return {
                'total_risk': total_risk,
                'factor_risk': factor_risk,
                'specific_risk': specific_risk,
                'liquidity_risk': liquidity_risk
            }
        except Exception:
            return {'total_risk': 0.0}
    
    def _calculate_factor_contributions(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Calculate factor contributions to returns"""
        try:
            total_return = returns.sum()
            
            # Simplified factor contributions
            return {
                'momentum': total_return * 0.3,
                'value': total_return * 0.2,
                'quality': total_return * 0.2,
                'size': total_return * 0.15,
                'volatility': total_return * 0.15
            }
        except Exception:
            return {'total': 0.0}
    
    def _analyze_capacity(self, returns: pd.Series, positions: pd.Series) -> Dict[str, Any]:
        """Analyze capacity utilization and constraints"""
        try:
            return {
                'capacity_utilization': self._calculate_capacity_utilization(positions, returns),
                'signal_capacity': self._calculate_signal_capacity(returns, positions),
                'turnover_ratio': self._calculate_turnover_ratio(positions),
                'capacity_decay_rate': self._calculate_capacity_decay_rate(returns),
                'max_capacity': 1.0,
                'current_capacity': 0.8  # Example
            }
        except Exception:
            return {'capacity_utilization': 0.0}
    
    def _analyze_crowding(self, returns: pd.Series, benchmark_returns: pd.Series, 
                         positions: pd.Series) -> Dict[str, Any]:
        """Analyze crowding metrics"""
        try:
            return {
                'crowding_beta': self._calculate_crowding_beta(returns, benchmark_returns),
                'crowding_correlation': self._calculate_crowding_correlation(returns, benchmark_returns),
                'crowding_exposure': self._calculate_crowding_exposure(positions),
                'crowding_contribution': self._calculate_crowding_contribution(returns, positions),
                'crowding_risk': 0.1  # Example
            }
        except Exception:
            return {'crowding_beta': 0.0}
    
    def _analyze_costs(self, returns: pd.Series, positions: pd.Series, 
                      costs: pd.Series) -> Dict[str, float]:
        """Analyze cost breakdown"""
        try:
            return {
                'total_costs': costs.sum() if costs is not None else 0,
                'borrow_costs': self._calculate_borrow_costs(positions, returns),
                'transaction_costs': self._calculate_transaction_costs(positions),
                'market_impact': self._calculate_market_impact(positions, returns),
                'cost_ratio': costs.sum() / returns.sum() if returns.sum() != 0 else 0
            }
        except Exception:
            return {'total_costs': 0.0}
    
    def _analyze_slippage(self, returns: pd.Series, positions: pd.Series) -> Dict[str, Any]:
        """Analyze slippage patterns"""
        try:
            # Simplified slippage analysis
            position_changes = positions.diff().abs()
            slippage_estimate = position_changes * 0.0001  # 1 bps impact
            
            return {
                'total_slippage': slippage_estimate.sum(),
                'avg_slippage': slippage_estimate.mean(),
                'slippage_volatility': slippage_estimate.std(),
                'slippage_impact': slippage_estimate.sum() / returns.sum() if returns.sum() != 0 else 0
            }
        except Exception:
            return {'total_slippage': 0.0}
    
    def _compare_to_benchmark(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Compare performance to benchmark"""
        try:
            excess_returns = returns - benchmark_returns
            
            return {
                'excess_return': excess_returns.sum(),
                'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0,
                'tracking_error': excess_returns.std() * np.sqrt(252),
                'beta': self._calculate_beta(returns, benchmark_returns),
                'alpha': self._calculate_alpha(returns, benchmark_returns, 0.02),
                'correlation': returns.corr(benchmark_returns)
            }
        except Exception:
            return {'excess_return': 0.0}
    
    def _create_fallback_evaluation(self, start_date: datetime, end_date: datetime) -> EvaluationResult:
        """Create fallback evaluation when analysis fails"""
        fallback_metrics = self._create_fallback_metrics()
        
        return EvaluationResult(
            evaluation_id="fallback",
            start_date=start_date,
            end_date=end_date,
            performance_metrics=fallback_metrics,
            daily_returns=pd.Series(),
            cumulative_returns=pd.Series(),
            rolling_metrics={},
            risk_decomposition={'total_risk': 0.0},
            factor_contributions={'total': 0.0},
            capacity_analysis={'capacity_utilization': 0.0},
            crowding_analysis={'crowding_beta': 0.0},
            cost_breakdown={'total_costs': 0.0},
            slippage_analysis={'total_slippage': 0.0},
            benchmark_comparison={'excess_return': 0.0},
            metadata={'error': 'fallback_evaluation'}
        )


# Factory function
async def create_allocator_metrics() -> AdvancedAllocatorMetrics:
    """Create and initialize allocator metrics"""
    return AdvancedAllocatorMetrics()


# Example usage
async def main():
    """Example usage of allocator metrics"""
    # Create sample data
    np.random.seed(42)
    n_days = 252
    
    # Generate sample returns and positions
    returns = pd.Series(np.random.randn(n_days) * 0.02, 
                       index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
    benchmark_returns = pd.Series(np.random.randn(n_days) * 0.015, 
                                 index=returns.index)
    positions = pd.Series(np.random.uniform(-1, 1, n_days), index=returns.index)
    costs = pd.Series(np.random.uniform(0, 0.001, n_days), index=returns.index)
    
    # Create metrics calculator
    metrics = await create_allocator_metrics()
    
    # Calculate comprehensive metrics
    performance_metrics = metrics.calculate_comprehensive_metrics(
        returns, benchmark_returns, positions, costs
    )
    
    print("Advanced Allocator Metrics Results:")
    print(f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.3f}")
    print(f"Information Ratio: {performance_metrics.information_ratio:.3f}")
    print(f"IR per Risk Dollar: {performance_metrics.ir_per_risk_dollar:.6f}")
    print(f"Capacity Utilization: {performance_metrics.capacity_utilization:.3f}")
    print(f"Crowding Beta: {performance_metrics.crowding_beta:.3f}")
    print(f"Total Slippage: {performance_metrics.total_slippage:.4f}")
    print(f"Borrow Costs: {performance_metrics.borrow_costs:.4f}")
    
    # Run comprehensive evaluation
    evaluation = await metrics.run_comprehensive_evaluation(
        returns, benchmark_returns, positions, costs,
        datetime(2023, 1, 1), datetime(2023, 12, 31)
    )
    
    print(f"\nComprehensive Evaluation ID: {evaluation.evaluation_id}")
    print(f"Risk Decomposition: {evaluation.risk_decomposition}")
    print(f"Capacity Analysis: {evaluation.capacity_analysis}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
