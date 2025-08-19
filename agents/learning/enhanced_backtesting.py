"""
Enhanced Backtesting System

Implements:
- Monte Carlo Simulation
- Regime Detection
- Stress Testing
- Transaction Costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    simulation_id: str
    total_return_mean: float
    total_return_std: float
    sharpe_ratio_mean: float
    sharpe_ratio_std: float
    max_drawdown_mean: float
    max_drawdown_std: float
    win_rate_mean: float
    win_rate_std: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    worst_case_scenario: Dict[str, float]
    best_case_scenario: Dict[str, float]
    simulation_paths: List[List[float]]

@dataclass
class MarketRegime:
    """Market regime information"""
    regime_id: int
    regime_name: str  # 'bull', 'bear', 'sideways', 'volatile'
    start_date: datetime
    end_date: datetime
    volatility: float
    trend_strength: float
    volume_profile: str
    regime_probability: float
    transition_probabilities: Dict[str, float]

@dataclass
class StressTestResult:
    """Results from stress testing"""
    stress_test_id: str
    scenario_name: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    recovery_time: int  # days
    worst_day_loss: float
    consecutive_losses: int
    scenario_severity: str  # 'mild', 'moderate', 'severe', 'extreme'

class MonteCarloSimulator:
    """Monte Carlo simulation for probabilistic performance analysis"""
    
    def __init__(self, n_simulations=1000, confidence_level=0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
    def simulate_returns(self, historical_returns: pd.Series, 
                        simulation_days: int = 252) -> List[List[float]]:
        """Simulate returns using historical distribution"""
        # Calculate historical statistics
        mean_return = historical_returns.mean()
        std_return = historical_returns.std()
        
        # Generate random returns
        simulated_paths = []
        for _ in range(self.n_simulations):
            # Use normal distribution with historical parameters
            daily_returns = np.random.normal(mean_return, std_return, simulation_days)
            simulated_paths.append(daily_returns.tolist())
        
        return simulated_paths
    
    def simulate_with_regime_switching(self, historical_data: pd.DataFrame, 
                                     regimes: List[MarketRegime], 
                                     simulation_days: int = 252) -> List[List[float]]:
        """Simulate returns with regime switching"""
        simulated_paths = []
        
        for _ in range(self.n_simulations):
            path = []
            current_regime = np.random.choice(regimes, p=[r.regime_probability for r in regimes])
            
            for day in range(simulation_days):
                # Generate return based on current regime
                regime_return = np.random.normal(
                    current_regime.trend_strength * 0.01,  # Convert to daily return
                    current_regime.volatility,
                    1
                )[0]
                
                path.append(regime_return)
                
                # Check for regime transition
                if np.random.random() < 0.05:  # 5% chance of regime change
                    current_regime = np.random.choice(regimes, p=[r.regime_probability for r in regimes])
            
            simulated_paths.append(path)
        
        return simulated_paths
    
    def calculate_portfolio_metrics(self, return_paths: List[List[float]], 
                                  initial_capital: float = 100000) -> MonteCarloResult:
        """Calculate portfolio metrics from simulated paths"""
        portfolio_paths = []
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for path in return_paths:
            # Calculate portfolio value path
            portfolio_values = [initial_capital]
            for daily_return in path:
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
            
            portfolio_paths.append(portfolio_values)
            
            # Calculate metrics
            total_return = (portfolio_values[-1] / initial_capital) - 1
            total_returns.append(total_return)
            
            # Sharpe ratio (assuming risk-free rate = 0)
            returns_series = pd.Series(path)
            sharpe_ratio = returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
            sharpe_ratios.append(sharpe_ratio)
            
            # Max drawdown
            peak = pd.Series(portfolio_values).expanding().max()
            drawdown = (pd.Series(portfolio_values) - peak) / peak
            max_drawdown = drawdown.min()
            max_drawdowns.append(max_drawdown)
            
            # Win rate
            win_rate = (returns_series > 0).mean()
            win_rates.append(win_rate)
        
        # Calculate statistics
        total_return_mean = np.mean(total_returns)
        total_return_std = np.std(total_returns)
        sharpe_ratio_mean = np.mean(sharpe_ratios)
        sharpe_ratio_std = np.std(sharpe_ratios)
        max_drawdown_mean = np.mean(max_drawdowns)
        max_drawdown_std = np.std(max_drawdowns)
        win_rate_mean = np.mean(win_rates)
        win_rate_std = np.std(win_rates)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        confidence_intervals = {
            'total_return': (np.percentile(total_returns, alpha/2*100), 
                           np.percentile(total_returns, (1-alpha/2)*100)),
            'sharpe_ratio': (np.percentile(sharpe_ratios, alpha/2*100), 
                           np.percentile(sharpe_ratios, (1-alpha/2)*100)),
            'max_drawdown': (np.percentile(max_drawdowns, alpha/2*100), 
                           np.percentile(max_drawdowns, (1-alpha/2)*100))
        }
        
        # Worst and best case scenarios
        worst_case_scenario = {
            'total_return': np.min(total_returns),
            'sharpe_ratio': np.min(sharpe_ratios),
            'max_drawdown': np.min(max_drawdowns),
            'win_rate': np.min(win_rates)
        }
        
        best_case_scenario = {
            'total_return': np.max(total_returns),
            'sharpe_ratio': np.max(sharpe_ratios),
            'max_drawdown': np.max(max_drawdowns),
            'win_rate': np.max(win_rates)
        }
        
        return MonteCarloResult(
            simulation_id=f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_return_mean=total_return_mean,
            total_return_std=total_return_std,
            sharpe_ratio_mean=sharpe_ratio_mean,
            sharpe_ratio_std=sharpe_ratio_std,
            max_drawdown_mean=max_drawdown_mean,
            max_drawdown_std=max_drawdown_std,
            win_rate_mean=win_rate_mean,
            win_rate_std=win_rate_std,
            confidence_intervals=confidence_intervals,
            worst_case_scenario=worst_case_scenario,
            best_case_scenario=best_case_scenario,
            simulation_paths=portfolio_paths
        )

class RegimeDetector:
    """Detect market regimes using clustering and statistical methods"""
    
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.regimes = []
        self.regime_model = None
        
    def detect_regimes(self, market_data: pd.DataFrame) -> List[MarketRegime]:
        """Detect market regimes from historical data"""
        if not ML_AVAILABLE:
            return self._detect_regimes_simple(market_data)
        
        # Prepare features for regime detection
        features = self._extract_regime_features(market_data)
        
        # Use Gaussian Mixture Model for regime detection
        self.regime_model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        regime_labels = self.regime_model.fit_predict(features)
        
        # Create regime objects
        self.regimes = []
        for regime_id in range(self.n_regimes):
            regime_mask = regime_labels == regime_id
            regime_data = market_data[regime_mask]
            
            if len(regime_data) > 0:
                regime = self._create_regime_object(regime_id, regime_data, market_data.index[regime_mask])
                self.regimes.append(regime)
        
        return self.regimes
    
    def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features = []
        
        for i in range(len(market_data)):
            # Use rolling windows for feature calculation
            window_size = 20
            if i >= window_size:
                window_data = market_data.iloc[i-window_size:i+1]
                
                # Calculate features
                returns = window_data['close'].pct_change().dropna()
                volatility = returns.std()
                trend = (window_data['close'].iloc[-1] / window_data['close'].iloc[0]) - 1
                volume_ratio = window_data['volume'].iloc[-1] / window_data['volume'].rolling(20).mean().iloc[-1]
                
                features.append([volatility, trend, volume_ratio])
            else:
                # Use default values for early data points
                features.append([0.02, 0.0, 1.0])
        
        return np.array(features)
    
    def _create_regime_object(self, regime_id: int, regime_data: pd.DataFrame, 
                            regime_dates: pd.DatetimeIndex) -> MarketRegime:
        """Create MarketRegime object from regime data"""
        returns = regime_data['close'].pct_change().dropna()
        
        # Determine regime characteristics
        volatility = returns.std()
        trend_strength = (regime_data['close'].iloc[-1] / regime_data['close'].iloc[0]) - 1
        
        # Classify regime type
        if trend_strength > 0.1 and volatility < 0.02:
            regime_name = 'bull'
        elif trend_strength < -0.1 and volatility < 0.02:
            regime_name = 'bear'
        elif volatility > 0.03:
            regime_name = 'volatile'
        else:
            regime_name = 'sideways'
        
        # Calculate regime probability
        regime_probability = len(regime_data) / len(regime_data)
        
        # Calculate transition probabilities (simplified)
        transition_probabilities = {
            'bull': 0.25,
            'bear': 0.25,
            'sideways': 0.25,
            'volatile': 0.25
        }
        
        return MarketRegime(
            regime_id=regime_id,
            regime_name=regime_name,
            start_date=regime_dates[0],
            end_date=regime_dates[-1],
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile='normal',
            regime_probability=regime_probability,
            transition_probabilities=transition_probabilities
        )
    
    def _detect_regimes_simple(self, market_data: pd.DataFrame) -> List[MarketRegime]:
        """Simple regime detection without ML"""
        # Split data into quarters and classify based on performance
        quarterly_returns = market_data['close'].resample('Q').last().pct_change().dropna()
        
        regimes = []
        for i, (date, ret) in enumerate(quarterly_returns.items()):
            if ret > 0.05:
                regime_name = 'bull'
            elif ret < -0.05:
                regime_name = 'bear'
            else:
                regime_name = 'sideways'
            
            regime = MarketRegime(
                regime_id=i,
                regime_name=regime_name,
                start_date=date,
                end_date=date + pd.DateOffset(months=3),
                volatility=0.02,
                trend_strength=ret,
                volume_profile='normal',
                regime_probability=0.25,
                transition_probabilities={'bull': 0.25, 'bear': 0.25, 'sideways': 0.5}
            )
            regimes.append(regime)
        
        return regimes
    
    def predict_current_regime(self, recent_data: pd.DataFrame) -> MarketRegime:
        """Predict current market regime"""
        if not self.regimes:
            return None
        
        # Use most recent regime as current
        return self.regimes[-1]

class StressTester:
    """Stress testing for extreme market scenarios"""
    
    def __init__(self):
        self.scenarios = {
            'market_crash': {
                'daily_return_shock': -0.05,  # 5% daily loss
                'volatility_multiplier': 3.0,
                'duration': 10,  # days
                'severity': 'severe'
            },
            'flash_crash': {
                'daily_return_shock': -0.10,  # 10% daily loss
                'volatility_multiplier': 5.0,
                'duration': 1,  # day
                'severity': 'extreme'
            },
            'recession': {
                'daily_return_shock': -0.02,  # 2% daily loss
                'volatility_multiplier': 2.0,
                'duration': 60,  # days
                'severity': 'moderate'
            },
            'liquidity_crisis': {
                'daily_return_shock': -0.03,  # 3% daily loss
                'volatility_multiplier': 4.0,
                'duration': 20,  # days
                'severity': 'severe'
            }
        }
    
    def run_stress_tests(self, strategy_returns: pd.Series, 
                        market_data: pd.DataFrame) -> List[StressTestResult]:
        """Run stress tests on strategy"""
        results = []
        
        for scenario_name, scenario_params in self.scenarios.items():
            result = self._apply_stress_scenario(strategy_returns, scenario_params, scenario_name)
            results.append(result)
        
        return results
    
    def _apply_stress_scenario(self, strategy_returns: pd.Series, 
                             scenario_params: Dict[str, Any], 
                             scenario_name: str) -> StressTestResult:
        """Apply stress scenario to strategy returns"""
        # Create stressed returns
        stressed_returns = strategy_returns.copy()
        
        # Apply shock
        shock_return = scenario_params['daily_return_shock']
        duration = scenario_params['duration']
        volatility_multiplier = scenario_params['volatility_multiplier']
        
        # Apply shock to random period
        shock_start = np.random.randint(0, len(stressed_returns) - duration)
        for i in range(duration):
            if shock_start + i < len(stressed_returns):
                # Apply shock and increase volatility
                stressed_returns.iloc[shock_start + i] = shock_return * volatility_multiplier
        
        # Calculate stressed metrics
        total_return = (1 + stressed_returns).prod() - 1
        sharpe_ratio = stressed_returns.mean() / stressed_returns.std() if stressed_returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + stressed_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate recovery time
        recovery_time = self._calculate_recovery_time(cumulative_returns, peak)
        
        # Calculate worst day loss
        worst_day_loss = stressed_returns.min()
        
        # Calculate consecutive losses
        consecutive_losses = self._calculate_consecutive_losses(stressed_returns)
        
        return StressTestResult(
            stress_test_id=f"stress_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scenario_name=scenario_name,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            recovery_time=recovery_time,
            worst_day_loss=worst_day_loss,
            consecutive_losses=consecutive_losses,
            scenario_severity=scenario_params['severity']
        )
    
    def _calculate_recovery_time(self, cumulative_returns: pd.Series, peak: pd.Series) -> int:
        """Calculate time to recover from drawdown"""
        drawdown = (cumulative_returns - peak) / peak
        recovery_threshold = -0.01  # 1% recovery threshold
        
        # Find last drawdown period
        in_drawdown = drawdown < recovery_threshold
        if not in_drawdown.any():
            return 0
        
        # Find recovery point
        recovery_point = None
        for i in range(len(drawdown) - 1, -1, -1):
            if not in_drawdown.iloc[i]:
                recovery_point = i
                break
        
        if recovery_point is None:
            return len(drawdown)
        
        # Find drawdown start
        drawdown_start = None
        for i in range(recovery_point, -1, -1):
            if in_drawdown.iloc[i]:
                drawdown_start = i
                break
        
        if drawdown_start is None:
            return 0
        
        return recovery_point - drawdown_start
    
    def _calculate_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        losses = returns < 0
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

class TransactionCostCalculator:
    """Calculate realistic transaction costs"""
    
    def __init__(self, commission_rate=0.001, slippage_rate=0.0005, 
                 market_impact_rate=0.0001):
        self.commission_rate = commission_rate  # 0.1% commission
        self.slippage_rate = slippage_rate      # 0.05% slippage
        self.market_impact_rate = market_impact_rate  # 0.01% market impact
    
    def calculate_transaction_costs(self, trade_size: float, 
                                  trade_price: float, 
                                  market_volume: float) -> float:
        """Calculate total transaction costs"""
        # Commission
        commission = trade_size * trade_price * self.commission_rate
        
        # Slippage
        slippage = trade_size * trade_price * self.slippage_rate
        
        # Market impact (increases with trade size relative to market volume)
        volume_ratio = (trade_size * trade_price) / market_volume
        market_impact = trade_size * trade_price * self.market_impact_rate * volume_ratio
        
        total_costs = commission + slippage + market_impact
        return total_costs
    
    def apply_transaction_costs_to_returns(self, strategy_returns: pd.Series, 
                                         trade_sizes: pd.Series, 
                                         prices: pd.Series, 
                                         volumes: pd.Series) -> pd.Series:
        """Apply transaction costs to strategy returns"""
        adjusted_returns = strategy_returns.copy()
        
        for i in range(len(strategy_returns)):
            if trade_sizes.iloc[i] > 0:  # If there's a trade
                costs = self.calculate_transaction_costs(
                    trade_sizes.iloc[i],
                    prices.iloc[i],
                    volumes.iloc[i]
                )
                
                # Reduce return by transaction costs
                trade_value = trade_sizes.iloc[i] * prices.iloc[i]
                cost_ratio = costs / trade_value
                adjusted_returns.iloc[i] -= cost_ratio
        
        return adjusted_returns

class EnhancedBacktestingEngine:
    """Enhanced backtesting engine with all advanced features"""
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.regime_detector = RegimeDetector()
        self.stress_tester = StressTester()
        self.cost_calculator = TransactionCostCalculator()
    
    def run_enhanced_backtest(self, strategy_returns: pd.Series, 
                            market_data: pd.DataFrame,
                            trade_sizes: pd.Series = None,
                            prices: pd.Series = None,
                            volumes: pd.Series = None) -> Dict[str, Any]:
        """Run comprehensive enhanced backtest"""
        results = {
            'basic_metrics': {},
            'monte_carlo': None,
            'regime_analysis': [],
            'stress_tests': [],
            'transaction_costs': {},
            'recommendations': []
        }
        
        # 1. Basic metrics
        results['basic_metrics'] = self._calculate_basic_metrics(strategy_returns)
        
        # 2. Monte Carlo simulation
        mc_result = self.monte_carlo.calculate_portfolio_metrics(
            self.monte_carlo.simulate_returns(strategy_returns)
        )
        results['monte_carlo'] = mc_result
        
        # 3. Regime detection
        regimes = self.regime_detector.detect_regimes(market_data)
        results['regime_analysis'] = regimes
        
        # 4. Stress testing
        stress_results = self.stress_tester.run_stress_tests(strategy_returns, market_data)
        results['stress_tests'] = stress_results
        
        # 5. Transaction costs (if trade data available)
        if trade_sizes is not None and prices is not None and volumes is not None:
            adjusted_returns = self.cost_calculator.apply_transaction_costs_to_returns(
                strategy_returns, trade_sizes, prices, volumes
            )
            results['transaction_costs'] = {
                'original_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                'adjusted_sharpe': adjusted_returns.mean() / adjusted_returns.std() if adjusted_returns.std() > 0 else 0,
                'cost_impact': (strategy_returns.mean() - adjusted_returns.mean()) / strategy_returns.mean() if strategy_returns.mean() > 0 else 0
            }
        
        # 6. Generate recommendations
        results['recommendations'] = self._generate_backtest_recommendations(results)
        
        return results
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        win_rate = (returns > 0).mean()
        volatility = returns.std()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'sortino_ratio': returns.mean() / returns[returns < 0].std() if returns[returns < 0].std() > 0 else 0
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def _generate_backtest_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from backtest results"""
        recommendations = []
        
        # Monte Carlo recommendations
        mc_result = results.get('monte_carlo')
        if mc_result:
            if mc_result.worst_case_scenario['total_return'] < -0.3:
                recommendations.append("⚠️ High downside risk: Consider risk management improvements")
            
            if mc_result.confidence_intervals['sharpe_ratio'][0] < 0.5:
                recommendations.append("📉 Low Sharpe ratio confidence: Strategy may be unstable")
        
        # Regime analysis recommendations
        regimes = results.get('regime_analysis', [])
        if regimes:
            regime_names = [r.regime_name for r in regimes]
            if 'volatile' in regime_names:
                recommendations.append("🌊 Volatile regime detected: Consider volatility-based position sizing")
        
        # Stress test recommendations
        stress_tests = results.get('stress_tests', [])
        for test in stress_tests:
            if test.scenario_severity in ['severe', 'extreme'] and test.max_drawdown < -0.4:
                recommendations.append(f"🚨 {test.scenario_name}: Extreme drawdown risk")
        
        # Transaction cost recommendations
        transaction_costs = results.get('transaction_costs', {})
        if transaction_costs:
            cost_impact = transaction_costs.get('cost_impact', 0)
            if cost_impact > 0.1:
                recommendations.append("💰 High transaction costs: Consider reducing trade frequency")
        
        return recommendations
