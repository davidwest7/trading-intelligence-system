#!/usr/bin/env python3
"""
Test Enhanced Learning Agent with Advanced Backtesting and Autonomous Updates

This script demonstrates:
- 5-year historical backtesting
- Multi-model ensemble analysis
- Autonomous code updates
- Performance monitoring
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

class EnhancedLearningAgentTest:
    def __init__(self):
        self.config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        }
        self.test_tickers = ['AAPL', 'TSLA', 'SPY']
        
    async def run_enhanced_learning_test(self):
        """Run comprehensive test of Enhanced Learning Agent"""
        print("🧠 **ENHANCED LEARNING AGENT TEST**")
        print("=" * 60)
        print("Testing Enhanced Learning Agent with 5-year backtesting")
        print(f"Test Tickers: {', '.join(self.test_tickers)}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 60)
        
        try:
            # Import the enhanced learning agent
            from agents.learning.agent_enhanced_backtesting import EnhancedLearningAgent
            
            # Initialize agent
            print("🔧 Initializing Enhanced Learning Agent...")
            agent = EnhancedLearningAgent(self.config)
            print("✅ Enhanced Learning Agent initialized")
            
            # Run analysis
            print(f"\n🔍 **ANALYZING {len(self.test_tickers)} TICKERS**")
            print("-" * 40)
            
            start_time = time.time()
            results = await agent.analyze_learning_system(self.test_tickers)
            processing_time = time.time() - start_time
            
            # Display results
            await self._display_enhanced_results(results, processing_time)
            
            return results
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Install required packages: pip install scikit-learn tensorflow joblib")
            return None
        except Exception as e:
            print(f"❌ Error running enhanced learning test: {e}")
            return None
    
    async def _display_enhanced_results(self, results, processing_time):
        """Display comprehensive results from enhanced learning agent"""
        if not results:
            print("❌ No results to display")
            return
        
        print(f"\n📊 **ENHANCED LEARNING AGENT RESULTS**")
        print("=" * 60)
        
        # Basic info
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"📈 Tickers Analyzed: {results.get('tickers_analyzed', 0)}")
        print(f"🌐 Data Source: {results.get('data_source', 'Unknown')}")
        
        # Model performances
        performances = results.get('models_performance', {})
        if performances:
            print(f"\n🎯 **MODEL PERFORMANCES**")
            print("-" * 40)
            
            for model_id, performance in performances.items():
                print(f"\n📊 **{model_id.upper()} MODEL**")
                print(f"   Sharpe Ratio: {performance.sharpe_ratio:.3f}")
                print(f"   Max Drawdown: {performance.max_drawdown:.3f}")
                print(f"   Hit Rate: {performance.hit_rate:.3f}")
                print(f"   Profit Factor: {performance.profit_factor:.3f}")
                print(f"   Total Return: {performance.total_return:.3f}")
                print(f"   Volatility: {performance.volatility:.3f}")
                print(f"   R² Score: {performance.r2_score:.3f}")
                print(f"   Training Loss: {performance.training_loss:.4f}")
                print(f"   Validation Loss: {performance.validation_loss:.4f}")
        
        # Backtest results
        backtest_results = results.get('backtest_results', {})
        if backtest_results:
            print(f"\n📈 **BACKTEST RESULTS (5-YEAR)**")
            print("-" * 40)
            
            for model_id, backtest in backtest_results.items():
                print(f"\n📊 **{model_id.upper()} BACKTEST**")
                print(f"   Total Return: {backtest.total_return:.3f}")
                print(f"   Sharpe Ratio: {backtest.sharpe_ratio:.3f}")
                print(f"   Max Drawdown: {backtest.max_drawdown:.3f}")
                print(f"   Win Rate: {backtest.win_rate:.3f}")
                print(f"   Profit Factor: {backtest.profit_factor:.3f}")
                print(f"   Total Trades: {backtest.total_trades}")
                print(f"   Avg Trade Return: {backtest.avg_trade_return:.4f}")
                print(f"   Volatility: {backtest.volatility:.4f}")
                print(f"   Calmar Ratio: {backtest.calmar_ratio:.3f}")
                print(f"   Sortino Ratio: {backtest.sortino_ratio:.3f}")
        
        # Code updates
        code_updates = results.get('code_updates', [])
        if code_updates:
            print(f"\n🔄 **AUTONOMOUS CODE UPDATES**")
            print("-" * 40)
            
            for update in code_updates:
                print(f"\n📝 **Update: {update.update_id}**")
                print(f"   Model: {update.model_id}")
                print(f"   Type: {update.update_type}")
                print(f"   Performance Improvement: {update.performance_improvement:.3f}")
                print(f"   Validation Score: {update.validation_score:.3f}")
                print(f"   Rollback Available: {update.rollback_available}")
                
                if update.changes:
                    print(f"   Changes:")
                    for change, value in update.changes.items():
                        print(f"     - {change}: {value}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 **RECOMMENDATIONS**")
            print("-" * 40)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Summary
        print(f"\n📊 **ENHANCED LEARNING AGENT TEST SUMMARY**")
        print("=" * 60)
        print(f"✅ Models Analyzed: {len(performances)}")
        print(f"📈 Backtests Completed: {len(backtest_results)}")
        print(f"🔄 Updates Applied: {len(code_updates)}")
        print(f"💡 Recommendations: {len(recommendations)}")
        print(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        
        # Find best performing model
        if performances:
            best_model = max(performances.items(), key=lambda x: x[1].sharpe_ratio)
            print(f"🎯 Best Model: {best_model[0]} (Sharpe: {best_model[1].sharpe_ratio:.3f})")
        
        print(f"\n🎉 Enhanced Learning Agent test completed successfully!")
        print(f"📊 Results available in test results")

async def main():
    """Main test function"""
    test = EnhancedLearningAgentTest()
    results = await test.run_enhanced_learning_test()
    return results

if __name__ == "__main__":
    asyncio.run(main())
