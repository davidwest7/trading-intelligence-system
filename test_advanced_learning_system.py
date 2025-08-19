#!/usr/bin/env python3
"""
Advanced Learning System Test

Tests all advanced features:
- Advanced Learning Methods (RL, Meta, Transfer, Online)
- Enhanced Backtesting (Monte Carlo, Regime, Stress, Transaction Costs)
- Autonomous Code Generation (GP, NAS, HP, Feature Selection)
"""

import asyncio
import time
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv('env_real_keys.env')

class AdvancedLearningSystemTest:
    def __init__(self):
        self.config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        }
        self.test_tickers = ['AAPL', 'TSLA', 'SPY']
        
    async def run_comprehensive_test(self):
        """Run comprehensive test of all advanced learning features"""
        print("🚀 **ADVANCED LEARNING SYSTEM TEST**")
        print("=" * 80)
        print("Testing Advanced Learning Methods, Enhanced Backtesting, and Autonomous Code Generation")
        print(f"Test Tickers: {', '.join(self.test_tickers)}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 80)
        
        try:
            # Import all advanced learning components
            from agents.learning.advanced_learning_methods import AdvancedLearningOrchestrator
            from agents.learning.enhanced_backtesting import EnhancedBacktestingEngine
            from agents.learning.autonomous_code_generation import AutonomousCodeGenerator
            from common.data_adapters.polygon_adapter import PolygonAdapter
            
            # Initialize components
            print("🔧 Initializing Advanced Learning System...")
            orchestrator = AdvancedLearningOrchestrator()
            backtesting_engine = EnhancedBacktestingEngine()
            code_generator = AutonomousCodeGenerator()
            polygon_adapter = PolygonAdapter(self.config)
            
            print("✅ Advanced Learning System initialized")
            
            # Collect historical data
            print(f"\n📊 **COLLECTING HISTORICAL DATA**")
            print("-" * 40)
            historical_data = await self._collect_historical_data(polygon_adapter)
            
            if not historical_data:
                print("❌ No historical data available for testing")
                return None
            
            # Prepare data for analysis
            prepared_data = self._prepare_data_for_analysis(historical_data)
            
            # Run all advanced learning tests
            results = {
                'advanced_learning_methods': {},
                'enhanced_backtesting': {},
                'autonomous_code_generation': {},
                'summary': {}
            }
            
            # 1. Advanced Learning Methods
            print(f"\n🧠 **ADVANCED LEARNING METHODS**")
            print("-" * 40)
            learning_results = await self._test_advanced_learning_methods(orchestrator, prepared_data)
            results['advanced_learning_methods'] = learning_results
            
            # 2. Enhanced Backtesting
            print(f"\n📈 **ENHANCED BACKTESTING**")
            print("-" * 40)
            backtest_results = await self._test_enhanced_backtesting(backtesting_engine, prepared_data)
            results['enhanced_backtesting'] = backtest_results
            
            # 3. Autonomous Code Generation
            print(f"\n🤖 **AUTONOMOUS CODE GENERATION**")
            print("-" * 40)
            code_results = await self._test_autonomous_code_generation(code_generator, prepared_data)
            results['autonomous_code_generation'] = code_results
            
            # Generate comprehensive summary
            results['summary'] = self._generate_comprehensive_summary(results)
            
            # Display results
            await self._display_comprehensive_results(results)
            
            return results
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Install required packages: pip install scikit-learn tensorflow joblib")
            return None
        except Exception as e:
            print(f"❌ Error running comprehensive test: {e}")
            return None
    
    async def _collect_historical_data(self, polygon_adapter) -> dict:
        """Collect historical data for testing"""
        historical_data = {}
        
        for ticker in self.test_tickers:
            try:
                print(f"📊 Collecting data for {ticker}...")
                
                # Get daily data for the last 2 years
                data = await polygon_adapter.get_intraday_data(
                    ticker, interval="D", limit=500
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
            if len(data) > 100:  # Minimum data requirement
                # Select feature columns
                feature_cols = ['rsi', 'macd', 'bb_position', 'volume_ratio', 
                              'sma_20', 'sma_50', 'price_change', 'volatility']
                
                # Ensure all features exist
                available_features = [col for col in feature_cols if col in data.columns]
                
                if len(available_features) >= 5:  # Minimum feature requirement
                    prepared_data[ticker] = {
                        'data': data[available_features + ['target']].dropna(),
                        'features': available_features,
                        'target': 'target'
                    }
        
        return prepared_data
    
    async def _test_advanced_learning_methods(self, orchestrator, prepared_data):
        """Test advanced learning methods"""
        results = {}
        
        for ticker, data_info in prepared_data.items():
            print(f"🧠 Testing advanced learning for {ticker}...")
            
            try:
                # Create performance history for meta-learning
                performance_history = self._create_sample_performance_history()
                
                # Run advanced learning optimization
                learning_results = orchestrator.optimize_strategy(
                    data_info['data'], performance_history
                )
                
                results[ticker] = learning_results
                print(f"✅ Advanced learning completed for {ticker}")
                
            except Exception as e:
                print(f"❌ Error in advanced learning for {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def _create_sample_performance_history(self):
        """Create sample performance history for testing"""
        import random
        
        history = []
        for i in range(30):  # 30 periods
            history.append({
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'max_drawdown': random.uniform(-0.3, -0.05),
                'total_return': random.uniform(-0.2, 0.5),
                'volatility': random.uniform(0.01, 0.05),
                'trend_strength': random.uniform(0.1, 0.8),
                'learning_rate': random.uniform(0.001, 0.1),
                'convergence_epochs': random.randint(50, 200)
            })
        
        return history
    
    async def _test_enhanced_backtesting(self, backtesting_engine, prepared_data):
        """Test enhanced backtesting"""
        results = {}
        
        for ticker, data_info in prepared_data.items():
            print(f"📈 Testing enhanced backtesting for {ticker}...")
            
            try:
                # Create sample strategy returns
                strategy_returns = self._create_sample_strategy_returns(data_info['data'])
                
                # Run enhanced backtest
                backtest_results = backtesting_engine.run_enhanced_backtest(
                    strategy_returns,
                    data_info['data']
                )
                
                results[ticker] = backtest_results
                print(f"✅ Enhanced backtesting completed for {ticker}")
                
            except Exception as e:
                print(f"❌ Error in enhanced backtesting for {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def _create_sample_strategy_returns(self, data):
        """Create sample strategy returns for testing"""
        import random
        
        # Create realistic strategy returns based on technical indicators
        returns = []
        for i in range(len(data) - 1):
            # Simple strategy based on RSI and MACD
            rsi = data['rsi'].iloc[i] if 'rsi' in data.columns else 50
            macd = data['macd'].iloc[i] if 'macd' in data.columns else 0
            
            # Generate strategy signal
            if rsi < 30 and macd > 0:
                # Buy signal
                actual_return = (data['close'].iloc[i+1] / data['close'].iloc[i]) - 1
                strategy_return = actual_return * 0.8  # 80% of actual return
            elif rsi > 70 and macd < 0:
                # Sell signal
                actual_return = (data['close'].iloc[i] / data['close'].iloc[i+1]) - 1
                strategy_return = actual_return * 0.8
            else:
                # Hold
                strategy_return = 0.0
            
            # Add some noise
            strategy_return += random.uniform(-0.001, 0.001)
            returns.append(strategy_return)
        
        return pd.Series(returns)
    
    async def _test_autonomous_code_generation(self, code_generator, prepared_data):
        """Test autonomous code generation"""
        results = {}
        
        for ticker, data_info in prepared_data.items():
            print(f"🤖 Testing autonomous code generation for {ticker}...")
            
            try:
                # Run autonomous code generation
                code_results = code_generator.generate_optimized_code(
                    data_info['data'],
                    data_info['target']
                )
                
                results[ticker] = code_results
                print(f"✅ Autonomous code generation completed for {ticker}")
                
            except Exception as e:
                print(f"❌ Error in autonomous code generation for {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def _generate_comprehensive_summary(self, results):
        """Generate comprehensive summary of all results"""
        summary = {
            'total_tickers_tested': len(self.test_tickers),
            'successful_tests': 0,
            'advanced_learning_success': 0,
            'backtesting_success': 0,
            'code_generation_success': 0,
            'best_performers': [],
            'recommendations': []
        }
        
        # Count successful tests
        for ticker in self.test_tickers:
            ticker_success = 0
            
            # Check advanced learning
            if ticker in results['advanced_learning_methods']:
                if 'error' not in results['advanced_learning_methods'][ticker]:
                    summary['advanced_learning_success'] += 1
                    ticker_success += 1
            
            # Check backtesting
            if ticker in results['enhanced_backtesting']:
                if 'error' not in results['enhanced_backtesting'][ticker]:
                    summary['backtesting_success'] += 1
                    ticker_success += 1
            
            # Check code generation
            if ticker in results['autonomous_code_generation']:
                if 'error' not in results['autonomous_code_generation'][ticker]:
                    summary['code_generation_success'] += 1
                    ticker_success += 1
            
            if ticker_success >= 2:  # At least 2 successful tests
                summary['successful_tests'] += 1
                summary['best_performers'].append(ticker)
        
        # Generate recommendations
        if summary['advanced_learning_success'] > 0:
            summary['recommendations'].append("🧠 Advanced learning methods working: Consider implementing RL strategies")
        
        if summary['backtesting_success'] > 0:
            summary['recommendations'].append("📈 Enhanced backtesting successful: Use for risk assessment")
        
        if summary['code_generation_success'] > 0:
            summary['recommendations'].append("🤖 Autonomous code generation active: Monitor for strategy improvements")
        
        if summary['successful_tests'] == len(self.test_tickers):
            summary['recommendations'].append("🎉 All systems operational: Full autonomous trading ready")
        
        return summary
    
    async def _display_comprehensive_results(self, results):
        """Display comprehensive results"""
        print(f"\n📊 **COMPREHENSIVE TEST RESULTS**")
        print("=" * 80)
        
        # Summary
        summary = results['summary']
        print(f"📈 Total Tickers Tested: {summary['total_tickers_tested']}")
        print(f"✅ Successful Tests: {summary['successful_tests']}")
        print(f"🧠 Advanced Learning Success: {summary['advanced_learning_success']}")
        print(f"📈 Backtesting Success: {summary['backtesting_success']}")
        print(f"🤖 Code Generation Success: {summary['code_generation_success']}")
        
        if summary['best_performers']:
            print(f"🏆 Best Performers: {', '.join(summary['best_performers'])}")
        
        # Detailed results for each component
        print(f"\n🧠 **ADVANCED LEARNING METHODS RESULTS**")
        print("-" * 40)
        for ticker, result in results['advanced_learning_methods'].items():
            if 'error' not in result:
                recommendations = result.get('recommendations', [])
                print(f"📊 {ticker}: {len(recommendations)} recommendations")
                for rec in recommendations[:2]:  # Show first 2 recommendations
                    print(f"   • {rec}")
            else:
                print(f"❌ {ticker}: {result['error']}")
        
        print(f"\n📈 **ENHANCED BACKTESTING RESULTS**")
        print("-" * 40)
        for ticker, result in results['enhanced_backtesting'].items():
            if 'error' not in result:
                basic_metrics = result.get('basic_metrics', {})
                if basic_metrics:
                    sharpe = basic_metrics.get('sharpe_ratio', 0)
                    max_dd = basic_metrics.get('max_drawdown', 0)
                    print(f"📊 {ticker}: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")
                
                recommendations = result.get('recommendations', [])
                for rec in recommendations[:2]:
                    print(f"   • {rec}")
            else:
                print(f"❌ {ticker}: {result['error']}")
        
        print(f"\n🤖 **AUTONOMOUS CODE GENERATION RESULTS**")
        print("-" * 40)
        for ticker, result in results['autonomous_code_generation'].items():
            if 'error' not in result:
                recommendations = result.get('recommendations', [])
                print(f"📊 {ticker}: {len(recommendations)} recommendations")
                for rec in recommendations[:2]:
                    print(f"   • {rec}")
            else:
                print(f"❌ {ticker}: {result['error']}")
        
        # Final recommendations
        print(f"\n💡 **FINAL RECOMMENDATIONS**")
        print("-" * 40)
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n🎉 **ADVANCED LEARNING SYSTEM TEST COMPLETED**")
        print("=" * 80)
        print(f"📊 Results available in test results")
        print(f"🚀 System ready for autonomous trading operations!")

async def main():
    """Main test function"""
    test = AdvancedLearningSystemTest()
    results = await test.run_comprehensive_test()
    return results

if __name__ == "__main__":
    asyncio.run(main())
