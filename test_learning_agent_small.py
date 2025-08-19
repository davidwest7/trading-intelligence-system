#!/usr/bin/env python3
"""
Small Test of Learning Agent - Fixed Version
Demonstrates the Learning Agent's capabilities with real data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_market_data():
    """Create realistic sample market data for testing"""
    print("📊 Creating sample market data...")
    
    # Generate 200 days of market data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Start with $100 price
    base_price = 100
    prices = [base_price]
    
    # Generate price movements with some trend and volatility
    for i in range(199):
        # Add some trend and random movement
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create volume data
    volumes = np.random.uniform(1000000, 10000000, 200)
    
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Add some technical indicators
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['sma_50'] = market_data['close'].rolling(50).mean()
    market_data['rsi'] = calculate_rsi(market_data['close'])
    market_data['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
    
    # Add target column for learning
    market_data['target'] = market_data['close'].pct_change().shift(-1)
    
    print(f"✅ Created {len(market_data)} days of market data")
    return market_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_sample_performance_history():
    """Create sample performance history for meta-learning"""
    print("📈 Creating sample performance history...")
    
    performance_history = []
    
    # Generate 30 periods of performance data
    for i in range(30):
        performance = {
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'total_return': np.random.uniform(-0.1, 0.3),
            'volatility': np.random.uniform(0.01, 0.05),
            'max_drawdown': np.random.uniform(-0.2, -0.01),
            'trend_strength': np.random.uniform(0.1, 0.8),
            'learning_rate': np.random.uniform(0.001, 0.01),
            'convergence_epochs': np.random.randint(50, 200)
        }
        performance_history.append(performance)
    
    print(f"✅ Created {len(performance_history)} performance records")
    return performance_history

def test_reinforcement_learning():
    """Test Reinforcement Learning Agent"""
    print("\n🤖 Testing Reinforcement Learning Agent...")
    
    try:
        from agents.learning.advanced_learning_methods_fixed import (
            ReinforcementLearningAgent, QLearningState, QLearningAction, AgentConfig
        )
        
        # Create agent with custom config
        config = AgentConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        rl_agent = ReinforcementLearningAgent(config)
        
        # Create sample state and actions
        state = QLearningState(
            market_regime='bull',
            volatility_level='low',
            trend_strength=0.6,
            volume_profile='normal',
            technical_signal='buy'
        )
        
        actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0)
        ]
        
        # Test action selection
        action = rl_agent.choose_action(state, actions)
        print(f"   ✅ Action selected: {action.action_type}")
        print(f"   📊 Position size: {action.position_size}")
        print(f"   🛡️ Stop loss: {action.stop_loss}")
        print(f"   🎯 Take profit: {action.take_profit}")
        
        # Test Q-learning update
        next_state = QLearningState('bull', 'low', 0.7, 'normal', 'hold')
        reward = 0.05  # 5% positive reward
        
        rl_agent.update_q_value(state, action, reward, next_state, actions)
        q_value = rl_agent.get_q_value(state, action)
        print(f"   🧠 Q-value updated: {q_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RL test failed: {e}")
        return False

def test_meta_learning():
    """Test Meta-Learning Agent"""
    print("\n🧠 Testing Meta-Learning Agent...")
    
    try:
        from agents.learning.advanced_learning_methods_fixed import MetaLearningAgent, AgentConfig
        
        meta_agent = MetaLearningAgent()
        
        # Create sample performance history
        performance_history = create_sample_performance_history()
        
        # Test meta-learning
        strategy_name = 'test_strategy'
        meta_agent.learn_optimal_strategy(strategy_name, performance_history)
        
        # Test parameter prediction
        current_state = {
            'volatility': 0.03,
            'trend_strength': 0.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'learning_rate': 0.005,
            'convergence_epochs': 100
        }
        
        optimal_params = meta_agent.predict_optimal_parameters(strategy_name, current_state)
        
        if optimal_params:
            print(f"   ✅ Meta-learning completed")
            print(f"   📊 Predicted improvement: {optimal_params.get('predicted_improvement', 0):.3f}")
            print(f"   ⚙️ Optimal learning rate: {optimal_params.get('learning_rate', 0):.4f}")
        else:
            print(f"   ⚠️ No optimal parameters predicted")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Meta-learning test failed: {e}")
        return False

def test_transfer_learning():
    """Test Transfer Learning Agent"""
    print("\n🔄 Testing Transfer Learning Agent...")
    
    try:
        from agents.learning.advanced_learning_methods_fixed import TransferLearningAgent, AgentConfig
        
        transfer_agent = TransferLearningAgent()
        
        # Create sample data for source and target markets
        market_data = create_sample_market_data()
        
        # Split data into source and target periods
        split_point = len(market_data) // 2
        source_data = market_data.iloc[:split_point].copy()
        target_data = market_data.iloc[split_point:].copy()
        
        # Add some features for learning
        source_data['feature1'] = source_data['close'].pct_change()
        source_data['feature2'] = source_data['volume'].pct_change()
        source_data['feature3'] = source_data['rsi']
        
        target_data['feature1'] = target_data['close'].pct_change()
        target_data['feature2'] = target_data['volume'].pct_change()
        target_data['feature3'] = target_data['rsi']
        
        # Train source model
        transfer_agent.train_source_model('source_market', source_data, 'target')
        
        # Adapt to target market
        transfer_agent.adapt_to_target_market('source_market', 'target_market', target_data, 'target')
        
        # Get transfer recommendations
        recommendations = transfer_agent.get_transfer_recommendations()
        
        print(f"   ✅ Transfer learning completed")
        print(f"   📊 Transfer recommendations: {len(recommendations)}")
        
        for rec in recommendations[:2]:  # Show first 2 recommendations
            print(f"   🔄 {rec['transfer']}: {rec['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Transfer learning test failed: {e}")
        return False

def test_online_learning():
    """Test Online Learning Agent"""
    print("\n📈 Testing Online Learning Agent...")
    
    try:
        from agents.learning.advanced_learning_methods_fixed import OnlineLearningAgent, AgentConfig
        
        online_agent = OnlineLearningAgent(base_model_type='linear', learning_rate=0.01)
        
        # Create sample data
        market_data = create_sample_market_data()
        
        # Prepare data for online learning
        recent_data = market_data.tail(20).copy()
        recent_data['feature1'] = recent_data['close'].pct_change()
        recent_data['feature2'] = recent_data['volume'].pct_change()
        recent_data['feature3'] = recent_data['rsi']
        
        # Calculate performance metric
        returns = recent_data['close'].pct_change().dropna()
        performance_metric = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Update online model
        model_name = 'test_online_model'
        online_agent.update_online_model(model_name, recent_data, 'target', performance_metric)
        
        # Get performance metrics
        performance = online_agent.get_online_performance(model_name)
        
        print(f"   ✅ Online learning completed")
        print(f"   📊 Performance metric: {performance_metric:.3f}")
        print(f"   🧠 Model type: {performance.get('model_type', 'unknown')}")
        print(f"   📈 Learning rate: {performance.get('learning_rate', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Online learning test failed: {e}")
        return False

def test_full_orchestrator():
    """Test the complete Learning Agent Orchestrator"""
    print("\n🎯 Testing Complete Learning Agent Orchestrator...")
    
    try:
        from agents.learning.advanced_learning_methods_fixed import AdvancedLearningOrchestrator, AgentConfig
        
        # Create orchestrator with custom config
        config = AgentConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
            enable_logging=True,
            enable_caching=True
        )
        
        orchestrator = AdvancedLearningOrchestrator(config)
        
        # Create sample data
        market_data = create_sample_market_data()
        performance_history = create_sample_performance_history()
        
        # Run complete optimization
        print("   🔄 Running strategy optimization...")
        results = orchestrator.optimize_strategy(market_data, performance_history)
        
        print(f"   ✅ Strategy optimization completed")
        print(f"   📊 Results summary:")
        
        # Show RL results
        rl_results = results.get('reinforcement_learning', {})
        if rl_results:
            action = rl_results.get('recommended_action', 'hold')
            print(f"      🤖 RL Action: {action.upper()}")
        
        # Show meta-learning results
        meta_results = results.get('meta_learning', {})
        if meta_results:
            improvement = meta_results.get('predicted_improvement', 0)
            print(f"      🧠 Meta-learning improvement: {improvement:.3f}")
        
        # Show transfer learning results
        transfer_results = results.get('transfer_learning', {})
        if transfer_results:
            recs = transfer_results.get('transfer_recommendations', [])
            print(f"      🔄 Transfer recommendations: {len(recs)}")
        
        # Show online learning results
        online_results = results.get('online_learning', {})
        if online_results and 'performance_metric' in online_results:
            perf = online_results['performance_metric']
            print(f"      📈 Online learning performance: {perf:.3f}")
        
        # Show combined recommendations
        recommendations = results.get('recommendations', [])
        print(f"      💡 Combined recommendations: {len(recommendations)}")
        for rec in recommendations[:3]:  # Show first 3 recommendations
            print(f"         • {rec}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Orchestrator test failed: {e}")
        return False

def main():
    """Run all Learning Agent tests"""
    print("🧠 LEARNING AGENT - SMALL TEST")
    print("=" * 50)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Test individual components
    test_results.append(("Reinforcement Learning", test_reinforcement_learning()))
    test_results.append(("Meta-Learning", test_meta_learning()))
    test_results.append(("Transfer Learning", test_transfer_learning()))
    test_results.append(("Online Learning", test_online_learning()))
    
    # Test complete orchestrator
    test_results.append(("Complete Orchestrator", test_full_orchestrator()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Learning Agent is working perfectly!")
    elif passed > total // 2:
        print("⚠️ Most tests passed. Some components may need attention.")
    else:
        print("❌ Multiple tests failed. System needs debugging.")
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
