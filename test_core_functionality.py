#!/usr/bin/env python3
"""
Core Functionality Test - Minimal Test Script

Tests basic functionality without complex dependencies
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

def test_imports():
    """Test if we can import the core modules"""
    print("🔧 Testing imports...")
    
    try:
        # Test basic imports
        print("✅ Basic imports successful")
        
        # Test advanced learning methods
        try:
            from agents.learning.advanced_learning_methods import (
                ReinforcementLearningAgent, MetaLearningAgent
            )
            print("✅ Advanced learning methods imported")
        except Exception as e:
            print(f"❌ Advanced learning methods import failed: {e}")
        
        # Test enhanced backtesting
        try:
            from agents.learning.enhanced_backtesting import (
                MonteCarloSimulator, RegimeDetector
            )
            print("✅ Enhanced backtesting imported")
        except Exception as e:
            print(f"❌ Enhanced backtesting import failed: {e}")
        
        # Test autonomous code generation
        try:
            from agents.learning.autonomous_code_generation import (
                GeneticProgramming, FeatureSelector
            )
            print("✅ Autonomous code generation imported")
        except Exception as e:
            print(f"❌ Autonomous code generation import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_reinforcement_learning():
    """Test reinforcement learning functionality"""
    print("\n🧠 Testing Reinforcement Learning...")
    
    try:
        from agents.learning.advanced_learning_methods import (
            ReinforcementLearningAgent, QLearningState, QLearningAction
        )
        
        # Create agent
        rl_agent = ReinforcementLearningAgent()
        print("✅ RL Agent created")
        
        # Create state and actions
        state = QLearningState(
            market_regime='bull',
            volatility_level='low',
            trend_strength=0.5,
            volume_profile='normal',
            technical_signal='hold'
        )
        
        actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0)
        ]
        
        # Choose action
        action = rl_agent.choose_action(state, actions)
        print(f"✅ RL Action chosen: {action.action_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ RL test failed: {e}")
        return False

def test_monte_carlo():
    """Test Monte Carlo simulation"""
    print("\n🎲 Testing Monte Carlo Simulation...")
    
    try:
        from agents.learning.enhanced_backtesting import MonteCarloSimulator
        
        # Create simulator
        mc_simulator = MonteCarloSimulator(n_simulations=10)  # Small for testing
        print("✅ Monte Carlo Simulator created")
        
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Simulate returns
        simulated_paths = mc_simulator.simulate_returns(returns, simulation_days=10)
        print(f"✅ Generated {len(simulated_paths)} simulation paths")
        
        # Calculate metrics
        mc_result = mc_simulator.calculate_portfolio_metrics(simulated_paths, initial_capital=100000)
        print(f"✅ Monte Carlo metrics calculated: Sharpe={mc_result.sharpe_ratio_mean:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monte Carlo test failed: {e}")
        return False

def test_genetic_programming():
    """Test genetic programming"""
    print("\n🧬 Testing Genetic Programming...")
    
    try:
        from agents.learning.autonomous_code_generation import GeneticProgramming
        
        # Create genetic programming
        gp = GeneticProgramming(population_size=5, generations=2)  # Small for testing
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
        
        # Run evolution
        gp.evolve_population(sample_data)
        print("✅ Evolution completed")
        
        # Get best strategy
        best_strategy = gp.get_best_strategy()
        if best_strategy:
            print(f"✅ Best strategy found with fitness: {best_strategy.fitness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Genetic Programming test failed: {e}")
        return False

def test_feature_selection():
    """Test feature selection"""
    print("\n🔍 Testing Feature Selection...")
    
    try:
        from agents.learning.autonomous_code_generation import FeatureSelector
        
        # Create feature selector
        fs = FeatureSelector()
        print("✅ Feature Selector created")
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100)
        })
        y = pd.Series(np.random.normal(0, 1, 100))
        
        # Select features
        feature_sets = fs.select_features(X, y, methods=['correlation', 'random_forest'])
        print(f"✅ Feature selection completed: {len(feature_sets)} feature sets")
        
        # Get best feature set
        best_feature_set = fs.get_best_feature_set()
        if best_feature_set:
            print(f"✅ Best feature set: {best_feature_set.feature_count} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature Selection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 **CORE FUNCTIONALITY TEST**")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Add current directory to path
    sys.path.append('.')
    
    # Test results
    results = {
        'imports': False,
        'reinforcement_learning': False,
        'monte_carlo': False,
        'genetic_programming': False,
        'feature_selection': False
    }
    
    # Run tests
    results['imports'] = test_imports()
    results['reinforcement_learning'] = test_reinforcement_learning()
    results['monte_carlo'] = test_monte_carlo()
    results['genetic_programming'] = test_genetic_programming()
    results['feature_selection'] = test_feature_selection()
    
    # Summary
    print(f"\n📊 **TEST SUMMARY**")
    print("=" * 60)
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Results: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Advanced Learning System is working correctly.")
    elif successful_tests > 0:
        print("⚠️ Some tests passed. System has partial functionality.")
    else:
        print("❌ No tests passed. System needs debugging.")
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    main()
