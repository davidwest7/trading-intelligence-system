#!/usr/bin/env python3
"""
Signal Capacity Study
Simulate increasing ADV%, re-optimize turnover penalty, capacity curves, decay vs. capital analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CapacityConfig:
    """Configuration for capacity study"""
    min_adv_pct: float = 0.01  # Minimum ADV% to test (1%)
    max_adv_pct: float = 0.20  # Maximum ADV% to test (20%)
    adv_steps: int = 20        # Number of ADV% steps to test
    
    # Simulation parameters
    n_simulations: int = 1000  # Number of Monte Carlo simulations
    simulation_days: int = 252 # Trading days per simulation
    
    # Market parameters
    daily_volatility: float = 0.02    # Daily return volatility
    daily_volume_mean: float = 1e6    # Average daily volume
    daily_volume_std: float = 0.3     # Volume volatility (CV)
    
    # Signal parameters
    signal_decay_halflife: int = 5    # Signal decay half-life (days)
    signal_volatility: float = 0.1    # Signal noise level
    base_information_ratio: float = 1.5  # Base IR before capacity constraints
    
    # Cost parameters
    spread_bps: float = 10            # Bid-ask spread (bps)
    linear_impact_coeff: float = 0.1  # Linear impact coefficient
    sqrt_impact_coeff: float = 0.5    # Square-root impact coefficient
    turnover_penalty_base: float = 2  # Base turnover penalty (bps)


@dataclass
class CapacityResult:
    """Result of capacity analysis"""
    adv_pct: float
    optimal_capital: float
    optimal_turnover: float
    net_information_ratio: float
    gross_information_ratio: float
    transaction_costs: float
    market_impact: float
    turnover_penalty: float
    capacity_utilization: float
    signal_decay_impact: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Detailed breakdown
    returns_before_costs: np.ndarray
    returns_after_costs: np.ndarray
    trading_volumes: np.ndarray
    position_sizes: np.ndarray
    
    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class CapacityStudyResult:
    """Complete capacity study results"""
    study_id: str
    timestamp: datetime
    config: CapacityConfig
    
    # Capacity curve data
    adv_percentages: np.ndarray
    optimal_capitals: np.ndarray
    information_ratios: np.ndarray
    transaction_costs: np.ndarray
    
    # Key findings
    max_capacity_adv_pct: float
    max_capacity_capital: float
    efficient_frontier: List[Tuple[float, float]]  # (risk, return) pairs
    
    # Detailed results
    detailed_results: List[CapacityResult]
    
    # Analysis metrics
    capacity_decay_rate: float
    cost_impact_elasticity: float
    optimal_turnover_curve: np.ndarray


class SignalGenerator:
    """Generate realistic trading signals with decay"""
    
    def __init__(self, config: CapacityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_signal_series(self, n_days: int, 
                             signal_strength: float = 1.0) -> np.ndarray:
        """Generate signal time series with realistic properties"""
        # Base signal (mean-reverting with trend component)
        base_signal = np.random.randn(n_days) * self.config.signal_volatility
        
        # Add persistence (AR(1) process)
        ar_coeff = 0.3
        for i in range(1, n_days):
            base_signal[i] += ar_coeff * base_signal[i-1]
        
        # Scale by signal strength
        signal = base_signal * signal_strength
        
        # Add signal decay over time (alpha decay)
        decay_rate = np.log(2) / self.config.signal_decay_halflife
        decay_factors = np.exp(-decay_rate * np.arange(n_days))
        signal *= decay_factors
        
        return signal
    
    def generate_market_returns(self, signal: np.ndarray, 
                              signal_loading: float = 1.0) -> np.ndarray:
        """Generate market returns based on signal"""
        n_days = len(signal)
        
        # Market noise
        market_noise = np.random.randn(n_days) * self.config.daily_volatility
        
        # Signal contribution to returns
        signal_returns = signal * signal_loading
        
        # Total returns
        returns = signal_returns + market_noise
        
        return returns


class TransactionCostModel:
    """Model transaction costs and market impact"""
    
    def __init__(self, config: CapacityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_transaction_costs(self, trade_volume: np.ndarray,
                                  market_volume: np.ndarray,
                                  adv_pct: float) -> Dict[str, np.ndarray]:
        """Calculate various components of transaction costs"""
        # Participation rate
        participation_rate = np.abs(trade_volume) / market_volume
        
        # Bid-ask spread cost (always paid)
        spread_cost = np.abs(trade_volume) * self.config.spread_bps * 1e-4
        
        # Linear market impact
        linear_impact = (
            self.config.linear_impact_coeff * participation_rate * 
            np.abs(trade_volume) * 1e-4
        )
        
        # Square-root market impact (Almgren-Chriss style)
        sqrt_impact = (
            self.config.sqrt_impact_coeff * np.sqrt(participation_rate) * 
            np.abs(trade_volume) * 1e-4
        )
        
        # Turnover penalty (higher for frequent trading)
        turnover_penalty = (
            self.config.turnover_penalty_base * 
            (1 + adv_pct * 10) *  # Penalty increases with ADV%
            np.abs(trade_volume) * 1e-4
        )
        
        # Cross-impact (impact on other positions)
        cross_impact = 0.1 * linear_impact  # 10% of linear impact
        
        total_costs = spread_cost + linear_impact + sqrt_impact + turnover_penalty + cross_impact
        
        return {
            'spread_cost': spread_cost,
            'linear_impact': linear_impact,
            'sqrt_impact': sqrt_impact,
            'turnover_penalty': turnover_penalty,
            'cross_impact': cross_impact,
            'total_costs': total_costs
        }


class CapacityOptimizer:
    """Optimize portfolio parameters for capacity analysis"""
    
    def __init__(self, config: CapacityConfig):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.cost_model = TransactionCostModel(config)
        self.logger = logging.getLogger(__name__)
    
    def optimize_capital_allocation(self, adv_pct: float,
                                  target_turnover: Optional[float] = None) -> CapacityResult:
        """Optimize capital allocation for given ADV%"""
        try:
            # Define optimization objective
            def objective(capital_scale):
                return -self._simulate_performance(adv_pct, capital_scale, target_turnover)
            
            # Optimize capital scale
            result = minimize_scalar(
                objective,
                bounds=(0.1, 10.0),
                method='bounded'
            )
            
            optimal_capital_scale = result.x
            optimal_performance = -result.fun
            
            # Run detailed simulation with optimal parameters
            detailed_result = self._detailed_simulation(adv_pct, optimal_capital_scale, target_turnover)
            
            return detailed_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing capital allocation: {e}")
            return self._create_fallback_result(adv_pct)
    
    def _simulate_performance(self, adv_pct: float, capital_scale: float,
                            target_turnover: Optional[float] = None) -> float:
        """Simulate portfolio performance for given parameters"""
        try:
            # Generate signals and returns
            signal = self.signal_generator.generate_signal_series(self.config.simulation_days)
            
            # Calculate signal loading based on capacity constraints
            base_loading = self.config.base_information_ratio / np.std(signal)
            capacity_adjusted_loading = base_loading / (1 + adv_pct * 5)  # Capacity constraint
            
            # Generate market returns
            market_returns = self.signal_generator.generate_market_returns(
                signal, capacity_adjusted_loading
            )
            
            # Calculate positions
            positions = signal * capital_scale
            
            # Apply turnover constraint if specified
            if target_turnover is not None:
                positions = self._apply_turnover_constraint(positions, target_turnover)
            
            # Calculate trading volumes
            trade_volumes = np.diff(positions, prepend=0)
            
            # Generate market volumes
            market_volumes = np.random.lognormal(
                np.log(self.config.daily_volume_mean),
                self.config.daily_volume_std,
                self.config.simulation_days
            )
            
            # Calculate transaction costs
            cost_components = self.cost_model.calculate_transaction_costs(
                trade_volumes, market_volumes, adv_pct
            )
            
            # Portfolio returns
            portfolio_returns = positions[:-1] * market_returns[1:]  # Lag positions
            net_returns = portfolio_returns - cost_components['total_costs'][1:]
            
            # Performance metric (Sharpe ratio)
            if np.std(net_returns) > 0:
                sharpe_ratio = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error in performance simulation: {e}")
            return 0.0
    
    def _apply_turnover_constraint(self, positions: np.ndarray, 
                                 target_turnover: float) -> np.ndarray:
        """Apply turnover constraint to position changes"""
        # Calculate current turnover
        trade_volumes = np.abs(np.diff(positions, prepend=0))
        current_turnover = np.mean(trade_volumes)
        
        if current_turnover > target_turnover:
            # Scale down position changes
            scale_factor = target_turnover / current_turnover
            scaled_trades = np.diff(positions, prepend=0) * scale_factor
            
            # Reconstruct positions
            constrained_positions = np.cumsum(scaled_trades)
            return constrained_positions
        
        return positions
    
    def _detailed_simulation(self, adv_pct: float, capital_scale: float,
                           target_turnover: Optional[float] = None) -> CapacityResult:
        """Run detailed simulation with all metrics"""
        try:
            # Run multiple simulations for robustness
            n_sims = min(100, self.config.n_simulations)
            simulation_results = []
            
            for sim in range(n_sims):
                sim_result = self._single_simulation(adv_pct, capital_scale, target_turnover)
                simulation_results.append(sim_result)
            
            # Aggregate results
            aggregated_result = self._aggregate_simulation_results(
                simulation_results, adv_pct, capital_scale
            )
            
            return aggregated_result
            
        except Exception as e:
            self.logger.error(f"Error in detailed simulation: {e}")
            return self._create_fallback_result(adv_pct)
    
    def _single_simulation(self, adv_pct: float, capital_scale: float,
                         target_turnover: Optional[float] = None) -> Dict[str, Any]:
        """Run single simulation"""
        # Generate data
        signal = self.signal_generator.generate_signal_series(self.config.simulation_days)
        capacity_adjusted_loading = self.config.base_information_ratio / (1 + adv_pct * 5)
        market_returns = self.signal_generator.generate_market_returns(
            signal, capacity_adjusted_loading
        )
        
        # Calculate positions
        positions = signal * capital_scale
        if target_turnover is not None:
            positions = self._apply_turnover_constraint(positions, target_turnover)
        
        # Trading volumes and market volumes
        trade_volumes = np.diff(positions, prepend=0)
        market_volumes = np.random.lognormal(
            np.log(self.config.daily_volume_mean),
            self.config.daily_volume_std,
            self.config.simulation_days
        )
        
        # Transaction costs
        cost_components = self.cost_model.calculate_transaction_costs(
            trade_volumes, market_volumes, adv_pct
        )
        
        # Returns
        portfolio_returns = positions[:-1] * market_returns[1:]
        net_returns = portfolio_returns - cost_components['total_costs'][1:]
        
        return {
            'positions': positions,
            'trade_volumes': trade_volumes,
            'market_volumes': market_volumes,
            'portfolio_returns': portfolio_returns,
            'net_returns': net_returns,
            'cost_components': cost_components,
            'signal': signal,
            'market_returns': market_returns
        }
    
    def _aggregate_simulation_results(self, simulation_results: List[Dict[str, Any]],
                                    adv_pct: float, capital_scale: float) -> CapacityResult:
        """Aggregate multiple simulation results"""
        # Combine all returns
        all_portfolio_returns = np.concatenate([sim['portfolio_returns'] for sim in simulation_results])
        all_net_returns = np.concatenate([sim['net_returns'] for sim in simulation_results])
        all_costs = np.concatenate([sim['cost_components']['total_costs'] for sim in simulation_results])
        
        # Performance metrics
        gross_ir = np.mean(all_portfolio_returns) / np.std(all_portfolio_returns) * np.sqrt(252) if np.std(all_portfolio_returns) > 0 else 0
        net_ir = np.mean(all_net_returns) / np.std(all_net_returns) * np.sqrt(252) if np.std(all_net_returns) > 0 else 0
        sharpe_ratio = net_ir
        
        # Risk metrics
        returns_series = pd.Series(all_net_returns)
        var_95 = returns_series.quantile(0.05)
        expected_shortfall = returns_series[returns_series <= var_95].mean()
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()
        
        # Drawdown calculation
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Cost breakdown
        avg_total_costs = np.mean(all_costs)
        market_impact = np.mean([
            np.mean(sim['cost_components']['linear_impact'] + sim['cost_components']['sqrt_impact'])
            for sim in simulation_results
        ])
        turnover_penalty = np.mean([
            np.mean(sim['cost_components']['turnover_penalty'])
            for sim in simulation_results
        ])
        
        # Capacity metrics
        avg_position_size = np.mean([np.mean(np.abs(sim['positions'])) for sim in simulation_results])
        avg_turnover = np.mean([np.mean(np.abs(sim['trade_volumes'])) for sim in simulation_results])
        capacity_utilization = adv_pct / 0.20  # Relative to 20% max
        
        # Signal decay impact
        signal_decay_impact = 1 - gross_ir / self.config.base_information_ratio if self.config.base_information_ratio > 0 else 0
        
        return CapacityResult(
            adv_pct=adv_pct,
            optimal_capital=capital_scale,
            optimal_turnover=avg_turnover,
            net_information_ratio=net_ir,
            gross_information_ratio=gross_ir,
            transaction_costs=avg_total_costs,
            market_impact=market_impact,
            turnover_penalty=turnover_penalty,
            capacity_utilization=capacity_utilization,
            signal_decay_impact=signal_decay_impact,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            returns_before_costs=all_portfolio_returns,
            returns_after_costs=all_net_returns,
            trading_volumes=np.concatenate([sim['trade_volumes'] for sim in simulation_results]),
            position_sizes=np.concatenate([sim['positions'] for sim in simulation_results]),
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def _create_fallback_result(self, adv_pct: float) -> CapacityResult:
        """Create fallback result when simulation fails"""
        return CapacityResult(
            adv_pct=adv_pct,
            optimal_capital=1.0,
            optimal_turnover=0.0,
            net_information_ratio=0.0,
            gross_information_ratio=0.0,
            transaction_costs=0.0,
            market_impact=0.0,
            turnover_penalty=0.0,
            capacity_utilization=adv_pct / 0.20,
            signal_decay_impact=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            returns_before_costs=np.array([0.0]),
            returns_after_costs=np.array([0.0]),
            trading_volumes=np.array([0.0]),
            position_sizes=np.array([0.0])
        )


class CapacityStudy:
    """Main class for running comprehensive capacity studies"""
    
    def __init__(self, config: Optional[CapacityConfig] = None):
        self.config = config or CapacityConfig()
        self.optimizer = CapacityOptimizer(self.config)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized Capacity Study")
    
    async def run_capacity_study(self) -> CapacityStudyResult:
        """Run comprehensive capacity study"""
        try:
            self.logger.info("Running capacity study...")
            
            # ADV percentage range
            adv_percentages = np.linspace(
                self.config.min_adv_pct,
                self.config.max_adv_pct,
                self.config.adv_steps
            )
            
            # Run optimization for each ADV%
            detailed_results = []
            optimal_capitals = []
            information_ratios = []
            transaction_costs = []
            
            for i, adv_pct in enumerate(adv_percentages):
                self.logger.info(f"Processing ADV% {adv_pct:.1%} ({i+1}/{len(adv_percentages)})")
                
                # Optimize for this ADV%
                result = self.optimizer.optimize_capital_allocation(adv_pct)
                detailed_results.append(result)
                
                optimal_capitals.append(result.optimal_capital)
                information_ratios.append(result.net_information_ratio)
                transaction_costs.append(result.transaction_costs)
            
            # Find maximum capacity
            max_capacity_idx = np.argmax(information_ratios)
            max_capacity_adv_pct = adv_percentages[max_capacity_idx]
            max_capacity_capital = optimal_capitals[max_capacity_idx]
            
            # Calculate efficient frontier
            efficient_frontier = self._calculate_efficient_frontier(detailed_results)
            
            # Analyze capacity decay
            capacity_decay_rate = self._calculate_capacity_decay_rate(
                adv_percentages, information_ratios
            )
            
            # Analyze cost impact elasticity
            cost_impact_elasticity = self._calculate_cost_elasticity(
                adv_percentages, transaction_costs
            )
            
            # Optimal turnover curve
            optimal_turnover_curve = np.array([result.optimal_turnover for result in detailed_results])
            
            study_result = CapacityStudyResult(
                study_id=f"capacity_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                config=self.config,
                adv_percentages=adv_percentages,
                optimal_capitals=np.array(optimal_capitals),
                information_ratios=np.array(information_ratios),
                transaction_costs=np.array(transaction_costs),
                max_capacity_adv_pct=max_capacity_adv_pct,
                max_capacity_capital=max_capacity_capital,
                efficient_frontier=efficient_frontier,
                detailed_results=detailed_results,
                capacity_decay_rate=capacity_decay_rate,
                cost_impact_elasticity=cost_impact_elasticity,
                optimal_turnover_curve=optimal_turnover_curve
            )
            
            self.logger.info(f"Capacity study completed. Max capacity at {max_capacity_adv_pct:.1%} ADV")
            return study_result
            
        except Exception as e:
            self.logger.error(f"Error running capacity study: {e}")
            # Return empty result
            return CapacityStudyResult(
                study_id="error",
                timestamp=datetime.now(),
                config=self.config,
                adv_percentages=np.array([]),
                optimal_capitals=np.array([]),
                information_ratios=np.array([]),
                transaction_costs=np.array([]),
                max_capacity_adv_pct=0.0,
                max_capacity_capital=0.0,
                efficient_frontier=[],
                detailed_results=[],
                capacity_decay_rate=0.0,
                cost_impact_elasticity=0.0,
                optimal_turnover_curve=np.array([])
            )
    
    def _calculate_efficient_frontier(self, results: List[CapacityResult]) -> List[Tuple[float, float]]:
        """Calculate risk-return efficient frontier"""
        efficient_points = []
        
        for result in results:
            if len(result.returns_after_costs) > 0:
                risk = np.std(result.returns_after_costs) * np.sqrt(252)
                return_value = np.mean(result.returns_after_costs) * 252
                efficient_points.append((risk, return_value))
        
        # Sort by risk
        efficient_points.sort(key=lambda x: x[0])
        
        return efficient_points
    
    def _calculate_capacity_decay_rate(self, adv_percentages: np.ndarray,
                                     information_ratios: np.ndarray) -> float:
        """Calculate the decay rate of capacity with ADV%"""
        try:
            # Fit exponential decay model: IR = IR_0 * exp(-decay_rate * ADV%)
            if len(information_ratios) > 2 and np.max(information_ratios) > 0:
                # Find peak and use points after peak
                peak_idx = np.argmax(information_ratios)
                if peak_idx < len(information_ratios) - 2:
                    x = adv_percentages[peak_idx:]
                    y = information_ratios[peak_idx:]
                    
                    # Log-linear regression
                    log_y = np.log(np.maximum(y, 1e-10))  # Avoid log(0)
                    
                    if len(x) > 1:
                        slope, _, _, _, _ = stats.linregress(x, log_y)
                        return -slope  # Decay rate
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_cost_elasticity(self, adv_percentages: np.ndarray,
                                 transaction_costs: np.ndarray) -> float:
        """Calculate elasticity of transaction costs to ADV%"""
        try:
            if len(transaction_costs) > 2:
                # Log-log regression: log(Cost) = α + β * log(ADV%)
                x = np.log(np.maximum(adv_percentages, 1e-10))
                y = np.log(np.maximum(transaction_costs, 1e-10))
                
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope  # Elasticity
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def analyze_turnover_optimization(self, adv_pct: float,
                                          turnover_range: Tuple[float, float] = (0.1, 2.0),
                                          n_points: int = 20) -> Dict[str, Any]:
        """Analyze optimal turnover for given ADV%"""
        try:
            turnover_levels = np.linspace(turnover_range[0], turnover_range[1], n_points)
            performance_results = []
            
            for turnover in turnover_levels:
                result = self.optimizer.optimize_capital_allocation(adv_pct, target_turnover=turnover)
                performance_results.append({
                    'turnover': turnover,
                    'information_ratio': result.net_information_ratio,
                    'transaction_costs': result.transaction_costs,
                    'sharpe_ratio': result.sharpe_ratio
                })
            
            # Find optimal turnover
            performance_df = pd.DataFrame(performance_results)
            optimal_idx = performance_df['information_ratio'].idxmax()
            optimal_turnover = performance_df.loc[optimal_idx, 'turnover']
            
            return {
                'optimal_turnover': optimal_turnover,
                'turnover_analysis': performance_results,
                'performance_curve': performance_df
            }
            
        except Exception as e:
            self.logger.error(f"Error in turnover optimization: {e}")
            return {'optimal_turnover': 1.0, 'turnover_analysis': [], 'performance_curve': pd.DataFrame()}
    
    async def run_sensitivity_analysis(self, base_result: CapacityStudyResult,
                                     parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Run sensitivity analysis on key parameters"""
        try:
            sensitivity_results = {}
            
            for param_name, (min_val, max_val) in parameter_ranges.items():
                param_values = np.linspace(min_val, max_val, 10)
                param_results = []
                
                for param_val in param_values:
                    # Create modified config
                    modified_config = CapacityConfig(**self.config.__dict__)
                    setattr(modified_config, param_name, param_val)
                    
                    # Run mini study with modified parameter
                    mini_optimizer = CapacityOptimizer(modified_config)
                    result = mini_optimizer.optimize_capital_allocation(base_result.max_capacity_adv_pct)
                    
                    param_results.append({
                        'parameter_value': param_val,
                        'information_ratio': result.net_information_ratio,
                        'optimal_capital': result.optimal_capital,
                        'transaction_costs': result.transaction_costs
                    })
                
                sensitivity_results[param_name] = param_results
            
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {e}")
            return {}
    
    def generate_capacity_report(self, study_result: CapacityStudyResult) -> str:
        """Generate comprehensive capacity study report"""
        report = f"""
# SIGNAL CAPACITY STUDY REPORT
Study ID: {study_result.study_id}
Generated: {study_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- Maximum Capacity: {study_result.max_capacity_adv_pct:.1%} of ADV
- Optimal Capital Scale: {study_result.max_capacity_capital:.2f}x
- Peak Information Ratio: {np.max(study_result.information_ratios):.2f}
- Capacity Decay Rate: {study_result.capacity_decay_rate:.3f}
- Cost Elasticity: {study_result.cost_impact_elasticity:.2f}

## CAPACITY CURVE ANALYSIS
ADV% Range: {study_result.config.min_adv_pct:.1%} - {study_result.config.max_adv_pct:.1%}
Data Points: {len(study_result.adv_percentages)}

Key Findings:
"""
        
        # Add performance at key ADV levels
        for i, adv_pct in enumerate([0.01, 0.05, 0.10, 0.15, 0.20]):
            if adv_pct <= study_result.config.max_adv_pct:
                # Find closest result
                closest_idx = np.argmin(np.abs(study_result.adv_percentages - adv_pct))
                result = study_result.detailed_results[closest_idx]
                
                report += f"""
- {adv_pct:.0%} ADV: IR={result.net_information_ratio:.2f}, Costs={result.transaction_costs:.1%}, MaxDD={result.max_drawdown:.1%}"""
        
        report += f"""

## TRANSACTION COST BREAKDOWN
Average cost components across capacity range:
- Market Impact: {np.mean([r.market_impact for r in study_result.detailed_results]):.1%}
- Turnover Penalty: {np.mean([r.turnover_penalty for r in study_result.detailed_results]):.1%}
- Total Transaction Costs: {np.mean(study_result.transaction_costs):.1%}

## RISK ANALYSIS
- Average VaR (95%): {np.mean([r.var_95 for r in study_result.detailed_results]):.2%}
- Average Max Drawdown: {np.mean([r.max_drawdown for r in study_result.detailed_results]):.1%}
- Average Skewness: {np.mean([r.skewness for r in study_result.detailed_results]):.2f}

## RECOMMENDATIONS
1. Optimal operating point: {study_result.max_capacity_adv_pct:.1%} ADV
2. Consider capacity limits beyond {study_result.max_capacity_adv_pct + 0.05:.1%} ADV
3. Monitor signal decay impact: {np.mean([r.signal_decay_impact for r in study_result.detailed_results]):.1%}
4. Regular re-optimization recommended due to market evolution

## MODEL PARAMETERS
- Base Information Ratio: {study_result.config.base_information_ratio:.2f}
- Signal Decay Half-life: {study_result.config.signal_decay_halflife} days
- Simulation Days: {study_result.config.simulation_days}
- Monte Carlo Runs: {study_result.config.n_simulations}
"""
        
        return report
    
    async def get_study_summary(self) -> Dict[str, Any]:
        """Get summary of capacity study capabilities"""
        return {
            "capacity_study": {
                "adv_range": f"{self.config.min_adv_pct:.1%} - {self.config.max_adv_pct:.1%}",
                "simulation_parameters": {
                    "days": self.config.simulation_days,
                    "monte_carlo_runs": self.config.n_simulations,
                    "steps": self.config.adv_steps
                },
                "signal_model": {
                    "base_ir": self.config.base_information_ratio,
                    "decay_halflife": self.config.signal_decay_halflife,
                    "volatility": self.config.signal_volatility
                },
                "cost_model": {
                    "spread_bps": self.config.spread_bps,
                    "linear_impact": self.config.linear_impact_coeff,
                    "sqrt_impact": self.config.sqrt_impact_coeff,
                    "turnover_penalty": self.config.turnover_penalty_base
                }
            }
        }


# Factory function
async def create_capacity_study(config: Optional[CapacityConfig] = None) -> CapacityStudy:
    """Create and initialize capacity study"""
    return CapacityStudy(config)


# Example usage
async def main():
    """Example usage of capacity study"""
    # Create configuration
    config = CapacityConfig(
        min_adv_pct=0.01,
        max_adv_pct=0.15,
        adv_steps=15,
        n_simulations=500,
        base_information_ratio=2.0,
        signal_decay_halflife=7
    )
    
    # Create capacity study
    study = await create_capacity_study(config)
    
    # Run capacity study
    print("Running capacity study...")
    result = await study.run_capacity_study()
    
    print("\nCapacity Study Results:")
    print(f"Study ID: {result.study_id}")
    print(f"Maximum Capacity: {result.max_capacity_adv_pct:.1%} ADV")
    print(f"Optimal Capital Scale: {result.max_capacity_capital:.2f}x")
    print(f"Peak Information Ratio: {np.max(result.information_ratios):.2f}")
    print(f"Capacity Decay Rate: {result.capacity_decay_rate:.3f}")
    print(f"Cost Elasticity: {result.cost_impact_elasticity:.2f}")
    
    # Show capacity curve points
    print("\nCapacity Curve (selected points):")
    for i in range(0, len(result.adv_percentages), 3):
        adv_pct = result.adv_percentages[i]
        ir = result.information_ratios[i]
        costs = result.transaction_costs[i]
        print(f"  {adv_pct:.1%} ADV: IR={ir:.2f}, Costs={costs:.3f}")
    
    # Analyze turnover optimization at peak capacity
    print(f"\nTurnover Optimization at {result.max_capacity_adv_pct:.1%} ADV:")
    turnover_analysis = await study.analyze_turnover_optimization(result.max_capacity_adv_pct)
    print(f"Optimal Turnover: {turnover_analysis['optimal_turnover']:.2f}")
    
    # Generate report
    report = study.generate_capacity_report(result)
    print("\nGenerated capacity study report")
    
    # Get study summary
    summary = await study.get_study_summary()
    print(f"\nStudy Configuration:")
    print(f"ADV Range: {summary['capacity_study']['adv_range']}")
    print(f"Base IR: {summary['capacity_study']['signal_model']['base_ir']}")
    print(f"Simulations: {summary['capacity_study']['simulation_parameters']['monte_carlo_runs']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
