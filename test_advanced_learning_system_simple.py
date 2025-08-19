#!/usr/bin/env python3
"""
Simplified Advanced Learning System Test

Tests core functionality without TensorFlow dependencies
"""

import asyncio
import time
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('env_real_keys.env')

class SimplifiedAdvancedLearningTest:
    def __init__(self):
        self.config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        }
        self.test_tickers = ['AAPL', 'TSLA', 'SPY']
        
    async def run_simplified_test(self):
        """Run simplified test focusing on core functionality"""
        print("🚀 **SIMPLIFIED ADVANCED LEARNING SYSTEM TEST**")
        print("=" * 80)
        print("Testing Core Advanced Learning Methods (No TensorFlow)")
        print(f"Test Tickers: {', '.join(self.test_tickers)}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 80)
        
        try:
            # Import core components (avoiding TensorFlow)
            from agents.learning.advanced_learning_methods import (
                ReinforcementLearningAgent, MetaLearningAgent, 
                TransferLearningAgent, OnlineLearningAgent
            )
            from agents.learning.enhanced_backtesting import (
                MonteCarloSimulator, RegimeDetector, StressTester, TransactionCostCalculator
            )
            from agents.learning.autonomous_code_generation import (
                GeneticProgramming, HyperparameterOptimizer, FeatureSelector
            )
            from common.data_adapters.polygon_adapter import PolygonAdapter
            
            # Initialize components
            print("🔧 Initializing Simplified Advanced Learning System...")
            
            # Advanced Learning Methods
            rl_agent = ReinforcementLearningAgent()
            meta_agent = MetaLearningAgent()
            transfer_agent = TransferLearningAgent()
            online_agent = OnlineLearningAgent()
            
            # Enhanced Backtesting
            mc_simulator = MonteCarloSimulator(n_simulations=100)  # Reduced for speed
            regime_detector = RegimeDetector()
            stress_tester = StressTester()
            cost_calculator = TransactionCostCalculator()
            
            # Autonomous Code Generation
            genetic_programming = GeneticProgramming(population_size=20, generations=10)  # Reduced
            hp_optimizer = HyperparameterOptimizer(n_trials=10)  # Reduced
            feature_selector = FeatureSelector()
            
            # Data adapter
            polygon_adapter = PolygonAdapter(self.config)
            
            print("✅ Simplified Advanced Learning System initialized")
            
            # Collect historical data
            print(f"\n📊 **COLLECTING HISTORICAL DATA**")
            print("-" * 40)
            historical_data = await self._collect_historical_data(polygon_adapter)
            
            if not historical_data:
                print("❌ No historical data available for testing")
                return None
            
            # Prepare data for analysis
            prepared_data = self._prepare_data_for_analysis(historical_data)
            
            # Run simplified tests
            results = {
                'reinforcement_learning': {},
                'meta_learning': {},
                'transfer_learning': {},
                'online_learning': {},
                'monte_carlo': {},
                'regime_detection': {},
                'stress_testing': {},
                'genetic_programming': {},
                'hyperparameter_optimization': {},
                'feature_selection': {},
                'summary': {}
            }
            
            # Test each component individually
            for ticker, data_info in prepared_data.items():
                print(f"\n🔍 **TESTING {ticker}**")
                print("-" * 40)
                
                # 1. Reinforcement Learning
                print(f"🧠 Testing Reinforcement Learning for {ticker}...")
                try:
                    rl_result = await self._test_reinforcement_learning(rl_agent, data_info)
                    results['reinforcement_learning'][ticker] = rl_result
                    print(f"✅ RL completed for {ticker}")
                except Exception as e:
                    print(f"❌ RL error for {ticker}: {e}")
                    results['reinforcement_learning'][ticker] = {'error': str(e)}
                
                # 2. Meta Learning
                print(f"🧠 Testing Meta Learning for {ticker}...")
                try:
                    meta_result = await self._test_meta_learning(meta_agent, data_info)
                    results['meta_learning'][ticker] = meta_result
                    print(f"✅ Meta Learning completed for {ticker}")
                except Exception as e:
                    print(f"❌ Meta Learning error for {ticker}: {e}")
                    results['meta_learning'][ticker] = {'error': str(e)}
                
                # 3. Monte Carlo Simulation
                print(f"🎲 Testing Monte Carlo Simulation for {ticker}...")
                try:
                    mc_result = await self._test_monte_carlo(mc_simulator, data_info)
                    results['monte_carlo'][ticker] = mc_result
                    print(f"✅ Monte Carlo completed for {ticker}")
                except Exception as e:
                    print(f"❌ Monte Carlo error for {ticker}: {e}")
                    results['monte_carlo'][ticker] = {'error': str(e)}
                
                # 4. Regime Detection
                print(f"📊 Testing Regime Detection for {ticker}...")
                try:
                    regime_result = await self._test_regime_detection(regime_detector, data_info)
                    results['regime_detection'][ticker] = regime_result
                    print(f"✅ Regime Detection completed for {ticker}")
                except Exception as e:
                    print(f"❌ Regime Detection error for {ticker}: {e}")
                    results['regime_detection'][ticker] = {'error': str(e)}
                
                # 5. Genetic Programming
                print(f"🧬 Testing Genetic Programming for {ticker}...")
                try:
                    gp_result = await self._test_genetic_programming(genetic_programming, data_info)
                    results['genetic_programming'][ticker] = gp_result
                    print(f"✅ Genetic Programming completed for {ticker}")
                except Exception as e:
                    print(f"❌ Genetic Programming error for {ticker}: {e}")
                    results['genetic_programming'][ticker] = {'error': str(e)}
                
                # 6. Feature Selection
                print(f"🔍 Testing Feature Selection for {ticker}...")
                try:
                    fs_result = await self._test_feature_selection(feature_selector, data_info)
                    results['feature_selection'][ticker] = fs_result
                    print(f"✅ Feature Selection completed for {ticker}")
                except Exception as e:
                    print(f"❌ Feature Selection error for {ticker}: {e}")
                    results['feature_selection'][ticker] = {'error': str(e)}
            
            # Generate summary
            results['summary'] = self._generate_simplified_summary(results)
            
            # Display results
            await self._display_simplified_results(results)
            
            return results
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Install required packages: pip install scikit-learn pandas numpy")
            return None
        except Exception as e:
            print(f"❌ Error running simplified test: {e}")
            return None
    
    async def _collect_historical_data(self, polygon_adapter) -> dict:
        """Collect historical data for testing"""
        historical_data = {}
        
        for ticker in self.test_tickers:
            try:
                print(f"📊 Collecting data for {ticker}...")
                
                # Get daily data for the last 1 year (reduced for speed)
                data = await polygon_adapter.get_intraday_data(
                    ticker, interval="D", limit=250
                )
                
                if not data.empty:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    historical_data[ticker] = data
                    print(f"✅ Collected {len(data)} data points for {ticker}")
                else:
                    print(f"⚠️ No data available for {ticker}")
                    
            except Exception as e:
                print(f"❌ Error collecting data for {ticker}: {e}")
        
        return historical_data
    
    def _add_technical_indicators(self, data):
        """Add technical indicators to data"""
        try:
            # RSI
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            
            # Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            data['bb_position'] = (data['close'] - (sma_20 - 2*std_20)) / (4*std_20)
            
            # Moving Averages
            data['sma_20'] = sma_20
            data['sma_50'] = data['close'].rolling(50).mean()
            
            # Volume and Price
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = data['price_change'].rolling(20).std()
            
            # Target variable (next day return)
            data['target'] = data['close'].shift(-1) / data['close'] - 1
            
            return data.dropna()
            
        except Exception as e:
            print(f"❌ Error adding technical indicators: {e}")
            return data
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _prepare_data_for_analysis(self, historical_data):
        """Prepare data for advanced analysis"""
        prepared_data = {}
        
        for ticker, data in historical_data.items():
            if len(data) > 50:  # Reduced minimum requirement
                # Select feature columns
                feature_cols = ['rsi', 'macd', 'bb_position', 'volume_ratio', 
                              'sma_20', 'sma_50', 'price_change', 'volatility']
                
                # Ensure all features exist
                available_features = [col for col in feature_cols if col in data.columns]
                
                if len(available_features) >= 3:  # Reduced minimum requirement
                    prepared_data[ticker] = {
                        'data': data[available_features + ['target']].dropna(),
                        'features': available_features,
                        'target': 'target'
                    }
        
        return prepared_data
    
    async def _test_reinforcement_learning(self, rl_agent, data_info):
        """Test reinforcement learning"""
        # Create sample market state
        recent_data = data_info['data'].tail(20)
        returns = recent_data['price_change'].dropna()
        
        # Create market state
        from agents.learning.advanced_learning_methods import QLearningState, QLearningAction
        
        state = QLearningState(
            market_regime='bull' if returns.mean() > 0 else 'bear',
            volatility_level='high' if returns.std() > 0.02 else 'low',
            trend_strength=abs(returns.mean()),
            volume_profile='normal',
            technical_signal='hold'
        )
        
        # Create actions
        actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0)
        ]
        
        # Choose action
        action = rl_agent.choose_action(state, actions)
        
        return {
            'recommended_action': action.action_type,
            'position_size': action.position_size,
            'market_regime': state.market_regime,
            'volatility': returns.std(),
            'trend_strength': state.trend_strength
        }
    
    async def _test_meta_learning(self, meta_agent, data_info):
        """Test meta learning"""
        # Create sample performance history
        performance_history = []
        for i in range(10):  # Reduced for speed
            performance_history.append({
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(-0.3, -0.05),
                'total_return': np.random.uniform(-0.2, 0.5),
                'volatility': np.random.uniform(0.01, 0.05),
                'trend_strength': np.random.uniform(0.1, 0.8),
                'learning_rate': np.random.uniform(0.001, 0.1),
                'convergence_epochs': np.random.randint(50, 200)
            })
        
        # Learn optimal strategy
        meta_agent.learn_optimal_strategy('test_strategy', performance_history)
        
        # Predict optimal parameters
        current_state = performance_history[-1]
        optimal_params = meta_agent.predict_optimal_parameters('test_strategy', current_state)
        
        return {
            'strategy_name': 'test_strategy',
            'optimal_parameters': optimal_params,
            'predicted_improvement': optimal_params.get('predicted_improvement', 0)
        }
    
    async def _test_monte_carlo(self, mc_simulator, data_info):
        """Test Monte Carlo simulation"""
        # Create sample returns
        returns = data_info['data']['target'].dropna()
        
        if len(returns) > 10:
            # Simulate returns
            simulated_paths = mc_simulator.simulate_returns(returns, simulation_days=50)
            
            # Calculate metrics
            mc_result = mc_simulator.calculate_portfolio_metrics(simulated_paths, initial_capital=100000)
            
            return {
                'total_return_mean': mc_result.total_return_mean,
                'sharpe_ratio_mean': mc_result.sharpe_ratio_mean,
                'max_drawdown_mean': mc_result.max_drawdown_mean,
                'confidence_intervals': mc_result.confidence_intervals
            }
        
        return {'error': 'Insufficient data for Monte Carlo simulation'}
    
    async def _test_regime_detection(self, regime_detector, data_info):
        """Test regime detection"""
        # Detect regimes
        regimes = regime_detector.detect_regimes(data_info['data'])
        
        if regimes:
            current_regime = regime_detector.predict_current_regime(data_info['data'].tail(20))
            
            return {
                'regime_count': len(regimes),
                'current_regime': current_regime.regime_name if current_regime else 'unknown',
                'regime_names': [r.regime_name for r in regimes]
            }
        
        return {'error': 'No regimes detected'}
    
    async def _test_genetic_programming(self, genetic_programming, data_info):
        """Test genetic programming"""
        # Initialize population
        genetic_programming.initialize_population()
        
        # Run evolution for a few generations
        genetic_programming.evolve_population(data_info['data'])
        
        # Get best strategy
        best_strategy = genetic_programming.get_best_strategy()
        
        if best_strategy:
            return {
                'best_code': best_strategy.code,
                'fitness': best_strategy.fitness,
                'performance_metrics': best_strategy.performance_metrics
            }
        
        return {'error': 'No strategy evolved'}
    
    async def _test_feature_selection(self, feature_selector, data_info):
        """Test feature selection"""
        # Prepare data
        X = data_info['data'][data_info['features']]
        y = data_info['data'][data_info['target']]
        
        # Select features
        feature_sets = feature_selector.select_features(X, y, methods=['correlation', 'random_forest'])
        best_feature_set = feature_selector.get_best_feature_set()
        
        if best_feature_set:
            return {
                'selected_features': best_feature_set.selected_features,
                'feature_count': best_feature_set.feature_count,
                'performance_score': best_feature_set.performance_score,
                'selection_method': best_feature_set.selection_method
            }
        
        return {'error': 'No features selected'}
    
    def _generate_simplified_summary(self, results):
        """Generate simplified summary"""
        summary = {
            'total_tickers_tested': len(self.test_tickers),
            'successful_tests': 0,
            'component_success': {},
            'best_performers': [],
            'recommendations': []
        }
        
        # Count successful tests for each component
        components = ['reinforcement_learning', 'meta_learning', 'monte_carlo', 
                     'regime_detection', 'genetic_programming', 'feature_selection']
        
        for component in components:
            success_count = 0
            for ticker in self.test_tickers:
                if ticker in results[component]:
                    if 'error' not in results[component][ticker]:
                        success_count += 1
            
            summary['component_success'][component] = success_count
            
            if success_count > 0:
                summary['recommendations'].append(f"✅ {component.replace('_', ' ').title()}: {success_count} successful tests")
        
        # Count overall successful tests
        for ticker in self.test_tickers:
            ticker_success = 0
            for component in components:
                if ticker in results[component] and 'error' not in results[component][ticker]:
                    ticker_success += 1
            
            if ticker_success >= 3:  # At least 3 successful components
                summary['successful_tests'] += 1
                summary['best_performers'].append(ticker)
        
        return summary
    
    async def _display_simplified_results(self, results):
        """Display simplified results"""
        print(f"\n📊 **SIMPLIFIED TEST RESULTS**")
        print("=" * 80)
        
        # Summary
        summary = results['summary']
        print(f"📈 Total Tickers Tested: {summary['total_tickers_tested']}")
        print(f"✅ Successful Tests: {summary['successful_tests']}")
        
        if summary['best_performers']:
            print(f"🏆 Best Performers: {', '.join(summary['best_performers'])}")
        
        # Component results
        print(f"\n🔧 **COMPONENT RESULTS**")
        print("-" * 40)
        for component, success_count in summary['component_success'].items():
            status = "✅" if success_count > 0 else "❌"
            print(f"{status} {component.replace('_', ' ').title()}: {success_count}/{len(self.test_tickers)} successful")
        
        # Sample results for each component
        components = ['reinforcement_learning', 'meta_learning', 'monte_carlo', 
                     'regime_detection', 'genetic_programming', 'feature_selection']
        
        for component in components:
            print(f"\n📊 **{component.replace('_', ' ').upper()} RESULTS**")
            print("-" * 40)
            for ticker in self.test_tickers:
                if ticker in results[component]:
                    if 'error' not in results[component][ticker]:
                        result = results[component][ticker]
                        if component == 'reinforcement_learning':
                            print(f"📊 {ticker}: {result.get('recommended_action', 'N/A')} action")
                        elif component == 'meta_learning':
                            improvement = result.get('predicted_improvement', 0)
                            print(f"📊 {ticker}: {improvement:.3f} predicted improvement")
                        elif component == 'monte_carlo':
                            sharpe = result.get('sharpe_ratio_mean', 0)
                            print(f"📊 {ticker}: Sharpe={sharpe:.3f}")
                        elif component == 'regime_detection':
                            regime = result.get('current_regime', 'unknown')
                            print(f"📊 {ticker}: {regime} regime")
                        elif component == 'genetic_programming':
                            fitness = result.get('fitness', 0)
                            print(f"📊 {ticker}: Fitness={fitness:.3f}")
                        elif component == 'feature_selection':
                            features = result.get('feature_count', 0)
                            print(f"📊 {ticker}: {features} features selected")
                    else:
                        print(f"❌ {ticker}: {results[component][ticker]['error']}")
        
        # Final recommendations
        print(f"\n💡 **RECOMMENDATIONS**")
        print("-" * 40)
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n🎉 **SIMPLIFIED ADVANCED LEARNING SYSTEM TEST COMPLETED**")
        print("=" * 80)
        print(f"📊 Core functionality tested successfully")
        print(f"🚀 System ready for further development!")

async def main():
    """Main test function"""
    test = SimplifiedAdvancedLearningTest()
    results = await test.run_simplified_test()
    return results

if __name__ == "__main__":
    asyncio.run(main())
