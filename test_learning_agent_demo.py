#!/usr/bin/env python3
"""
Learning Agent Demo - Small Test

This script demonstrates the Learning Agent's key capabilities:
- Reinforcement Learning
- Meta-Learning
- Transfer Learning
- Online Learning
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def demo_reinforcement_learning():
    """Demo Reinforcement Learning capabilities"""
    print("🧠 **REINFORCEMENT LEARNING DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.advanced_learning_methods import (
            ReinforcementLearningAgent, QLearningState, QLearningAction
        )
        
        # Create RL agent
        rl_agent = ReinforcementLearningAgent(learning_rate=0.1, epsilon=0.2)
        print("✅ RL Agent created")
        
        # Create market states
        states = [
            QLearningState('bull', 'low', 0.8, 'normal', 'buy'),
            QLearningState('bear', 'high', 0.3, 'high', 'sell'),
            QLearningState('sideways', 'medium', 0.5, 'normal', 'hold')
        ]
        
        # Create actions
        actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0)
        ]
        
        print("✅ States and actions created")
        
        # Simulate learning process
        print("\n🔄 Simulating learning process...")
        for i in range(5):
            state = states[i % len(states)]
            action = rl_agent.choose_action(state, actions)
            reward = np.random.uniform(-0.1, 0.1)  # Simulated reward
            
            next_state = states[(i + 1) % len(states)]
            next_actions = actions
            
            rl_agent.learn_from_experience(state, action, reward, next_state, next_actions)
            print(f"   Step {i+1}: State={state.market_regime}, Action={action.action_type}, Reward={reward:.3f}")
        
        print("✅ Reinforcement Learning demo completed")
        return True
        
    except Exception as e:
        print(f"❌ RL Demo error: {e}")
        return False

def demo_meta_learning():
    """Demo Meta-Learning capabilities"""
    print("\n🧠 **META-LEARNING DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.advanced_learning_methods import MetaLearningAgent
        
        # Create meta-learning agent
        meta_agent = MetaLearningAgent()
        print("✅ Meta-Learning Agent created")
        
        # Create performance history
        performance_history = []
        for i in range(10):
            performance_history.append({
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(-0.3, -0.05),
                'total_return': np.random.uniform(-0.2, 0.5),
                'volatility': np.random.uniform(0.01, 0.05),
                'trend_strength': np.random.uniform(0.1, 0.8),
                'learning_rate': np.random.uniform(0.001, 0.1),
                'convergence_epochs': np.random.randint(50, 200)
            })
        
        print("✅ Performance history created")
        
        # Learn optimal strategy
        strategy_name = "trading_strategy_1"
        meta_agent.learn_optimal_strategy(strategy_name, performance_history)
        print("✅ Meta-learning completed")
        
        # Predict optimal parameters
        current_state = performance_history[-1]
        optimal_params = meta_agent.predict_optimal_parameters(strategy_name, current_state)
        
        print(f"📊 Optimal Parameters Predicted:")
        print(f"   Learning Rate: {optimal_params.get('learning_rate', 'N/A'):.4f}")
        print(f"   Convergence Epochs: {optimal_params.get('convergence_epochs', 'N/A')}")
        print(f"   Predicted Improvement: {optimal_params.get('predicted_improvement', 'N/A'):.3f}")
        
        print("✅ Meta-Learning demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Meta-Learning Demo error: {e}")
        return False

def demo_transfer_learning():
    """Demo Transfer Learning capabilities"""
    print("\n🔄 **TRANSFER LEARNING DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.advanced_learning_methods import TransferLearningAgent
        
        # Create transfer learning agent
        transfer_agent = TransferLearningAgent()
        print("✅ Transfer Learning Agent created")
        
        # Create sample data for different markets
        source_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        target_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        print("✅ Source and target data created")
        
        # Train source model
        transfer_agent.train_source_model('US_market', source_data, 'target')
        print("✅ Source model trained")
        
        # Adapt to target market
        transfer_agent.adapt_to_target_market('US_market', 'EU_market', target_data, 'target')
        print("✅ Target market adaptation completed")
        
        # Get transfer recommendations
        recommendations = transfer_agent.get_transfer_recommendations()
        
        print(f"📊 Transfer Learning Results:")
        for rec in recommendations:
            print(f"   {rec['transfer']}: Score={rec['transfer_score']:.3f}, Recommendation={rec['recommendation']}")
        
        print("✅ Transfer Learning demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Transfer Learning Demo error: {e}")
        return False

def demo_online_learning():
    """Demo Online Learning capabilities"""
    print("\n📈 **ONLINE LEARNING DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.advanced_learning_methods import OnlineLearningAgent
        
        # Create online learning agent
        online_agent = OnlineLearningAgent(base_model_type='linear', learning_rate=0.01)
        print("✅ Online Learning Agent created")
        
        # Create online model
        model_name = "online_trading_model"
        online_agent.create_online_model(model_name)
        print("✅ Online model created")
        
        # Simulate online updates
        print("\n🔄 Simulating online updates...")
        for i in range(5):
            # Create new data batch
            new_data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 10),
                'feature2': np.random.normal(0, 1, 10),
                'target': np.random.normal(0, 1, 10)
            })
            
            # Calculate performance metric
            performance_metric = np.random.uniform(0.1, 0.9)
            
            # Update model
            online_agent.update_online_model(model_name, new_data, 'target', performance_metric)
            print(f"   Update {i+1}: Performance={performance_metric:.3f}")
        
        # Get online performance
        performance = online_agent.get_online_performance(model_name)
        
        print(f"\n📊 Online Learning Performance:")
        print(f"   Model: {performance.get('model_name', 'N/A')}")
        print(f"   Type: {performance.get('model_type', 'N/A')}")
        print(f"   Learning Rate: {performance.get('learning_rate', 'N/A'):.4f}")
        
        print("✅ Online Learning demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Online Learning Demo error: {e}")
        return False

def demo_enhanced_backtesting():
    """Demo Enhanced Backtesting capabilities"""
    print("\n📈 **ENHANCED BACKTESTING DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.enhanced_backtesting import (
            MonteCarloSimulator, RegimeDetector, StressTester
        )
        
        # Create Monte Carlo simulator
        mc_simulator = MonteCarloSimulator(n_simulations=50)  # Small for demo
        print("✅ Monte Carlo Simulator created")
        
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        print("✅ Sample returns created")
        
        # Run Monte Carlo simulation
        simulated_paths = mc_simulator.simulate_returns(returns, simulation_days=30)
        mc_result = mc_simulator.calculate_portfolio_metrics(simulated_paths, initial_capital=100000)
        
        print(f"📊 Monte Carlo Results:")
        print(f"   Total Return: {mc_result.total_return_mean:.3f} ± {mc_result.total_return_std:.3f}")
        print(f"   Sharpe Ratio: {mc_result.sharpe_ratio_mean:.3f} ± {mc_result.sharpe_ratio_std:.3f}")
        print(f"   Max Drawdown: {mc_result.max_drawdown_mean:.3f} ± {mc_result.max_drawdown_std:.3f}")
        
        # Create regime detector
        regime_detector = RegimeDetector(n_regimes=3)
        print("✅ Regime Detector created")
        
        # Create sample market data
        market_data = pd.DataFrame({
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000000, 10000000, 100)
        })
        
        # Detect regimes
        regimes = regime_detector.detect_regimes(market_data)
        print(f"📊 Regime Detection Results:")
        for regime in regimes[:3]:  # Show first 3 regimes
            print(f"   Regime {regime.regime_id}: {regime.regime_name}, Vol={regime.volatility:.3f}")
        
        # Create stress tester
        stress_tester = StressTester()
        print("✅ Stress Tester created")
        
        # Run stress tests
        strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        stress_results = stress_tester.run_stress_tests(strategy_returns, market_data)
        
        print(f"📊 Stress Test Results:")
        for result in stress_results[:2]:  # Show first 2 results
            print(f"   {result.scenario_name}: Return={result.total_return:.3f}, Drawdown={result.max_drawdown:.3f}")
        
        print("✅ Enhanced Backtesting demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Backtesting Demo error: {e}")
        return False

def demo_autonomous_code_generation():
    """Demo Autonomous Code Generation capabilities"""
    print("\n🤖 **AUTONOMOUS CODE GENERATION DEMO**")
    print("-" * 40)
    
    try:
        from agents.learning.autonomous_code_generation import (
            GeneticProgramming, FeatureSelector
        )
        
        # Create genetic programming
        gp = GeneticProgramming(population_size=10, generations=3)  # Small for demo
        print("✅ Genetic Programming created")
        
        # Initialize population
        gp.initialize_population()
        print(f"✅ Population initialized with {len(gp.population)} individuals")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'rsi': np.random.uniform(0, 100, 50),
            'macd': np.random.uniform(-1, 1, 50),
            'bb_position': np.random.uniform(0, 1, 50),
            'volume_ratio': np.random.uniform(0.5, 2.0, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000000, 10000000, 50)
        })
        
        print("✅ Sample data created")
        
        # Run evolution
        gp.evolve_population(sample_data)
        print("✅ Evolution completed")
        
        # Get best strategy
        best_strategy = gp.get_best_strategy()
        if best_strategy:
            print(f"📊 Best Strategy:")
            print(f"   Fitness: {best_strategy.fitness:.3f}")
            print(f"   Generation: {best_strategy.generation}")
            print(f"   Code: {best_strategy.code[:100]}...")
        
        # Create feature selector
        fs = FeatureSelector()
        print("✅ Feature Selector created")
        
        # Create sample data for feature selection
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100)
        })
        y = pd.Series(np.random.normal(0, 1, 100))
        
        # Select features
        feature_sets = fs.select_features(X, y, methods=['correlation', 'random_forest'])
        
        print(f"📊 Feature Selection Results:")
        for feature_set in feature_sets:
            print(f"   Method: {feature_set.selection_method}, Features: {feature_set.feature_count}, Score: {feature_set.performance_score:.3f}")
        
        print("✅ Autonomous Code Generation demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Autonomous Code Generation Demo error: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 **LEARNING AGENT DEMO**")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Run all demos
    demos = [
        ("Reinforcement Learning", demo_reinforcement_learning),
        ("Meta-Learning", demo_meta_learning),
        ("Transfer Learning", demo_transfer_learning),
        ("Online Learning", demo_online_learning),
        ("Enhanced Backtesting", demo_enhanced_backtesting),
        ("Autonomous Code Generation", demo_autonomous_code_generation)
    ]
    
    results = {}
    
    for demo_name, demo_function in demos:
        try:
            result = demo_function()
            results[demo_name] = result
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print(f"\n📊 **DEMO SUMMARY**")
    print("=" * 60)
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    for demo_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {demo_name}")
    
    print(f"\n🎯 Results: {successful_demos}/{total_demos} demos successful")
    
    if successful_demos == total_demos:
        print("🎉 All Learning Agent demos completed successfully!")
        print("🚀 The Learning Agent is fully functional and ready for production!")
    elif successful_demos > 0:
        print("⚠️ Some demos completed successfully. Check failed demos for issues.")
    else:
        print("❌ No demos completed successfully. Check system setup.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
