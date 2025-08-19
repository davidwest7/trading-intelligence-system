#!/usr/bin/env python3
"""
Comprehensive Learning Agent Test with Enhanced Logging
Tests all components with detailed logging to ensure all bugs are fixed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
import traceback
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

def create_realistic_market_data():
    """Create realistic market data for comprehensive testing"""
    print("📊 Creating realistic market data...")
    
    # Generate 500 days of market data (2 years)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    # Create realistic price movements with trends and volatility
    np.random.seed(42)  # For reproducible results
    
    # Start with $100 price
    base_price = 100
    prices = [base_price]
    
    # Generate price movements with realistic market behavior
    for i in range(499):
        # Add trend component (slight upward bias)
        trend_component = 0.0005  # 0.05% daily trend
        
        # Add volatility component
        volatility = 0.02  # 2% daily volatility
        
        # Add some market cycles
        cycle_component = 0.001 * np.sin(i * 2 * np.pi / 252)  # Annual cycle
        
        # Combine components
        daily_return = trend_component + cycle_component + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create volume data with some correlation to price movements
    volumes = []
    for i, price in enumerate(prices):
        base_volume = 1000000
        price_change = abs((price / prices[max(0, i-1)]) - 1) if i > 0 else 0
        volume_multiplier = 1 + price_change * 10  # Higher volume on price changes
        volume = base_volume * volume_multiplier * np.random.uniform(0.8, 1.2)
        volumes.append(volume)
    
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.015)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.015)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Add technical indicators
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['sma_50'] = market_data['close'].rolling(50).mean()
    market_data['ema_12'] = market_data['close'].ewm(span=12).mean()
    market_data['ema_26'] = market_data['close'].ewm(span=26).mean()
    market_data['rsi'] = calculate_rsi(market_data['close'])
    market_data['macd'] = market_data['ema_12'] - market_data['ema_26']
    market_data['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
    market_data['price_momentum'] = market_data['close'].pct_change(5)
    market_data['volatility'] = market_data['close'].pct_change().rolling(20).std()
    
    # Add target column for learning (next day's return)
    market_data['target'] = market_data['close'].pct_change().shift(-1)
    
    # Add some features for learning
    market_data['feature1'] = market_data['close'].pct_change()
    market_data['feature2'] = market_data['volume'].pct_change()
    market_data['feature3'] = market_data['rsi']
    market_data['feature4'] = market_data['macd']
    market_data['feature5'] = market_data['price_momentum']
    
    print(f"✅ Created {len(market_data)} days of realistic market data")
    return market_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_comprehensive_performance_history():
    """Create comprehensive performance history for testing"""
    print("📈 Creating comprehensive performance history...")
    
    performance_history = []
    
    # Generate 50 periods of performance data with realistic patterns
    for i in range(50):
        # Create realistic performance patterns
        base_sharpe = 1.0
        sharpe_variation = np.random.normal(0, 0.3)
        sharpe_ratio = max(0.1, base_sharpe + sharpe_variation)
        
        # Total return with some correlation to Sharpe ratio
        total_return = np.random.normal(sharpe_ratio * 0.1, 0.05)
        
        # Volatility inversely related to Sharpe ratio
        volatility = max(0.01, 0.05 - (sharpe_ratio - 1.0) * 0.01)
        
        # Drawdown related to volatility
        max_drawdown = -np.random.uniform(0.01, volatility * 2)
        
        # Trend strength based on performance
        trend_strength = max(0.1, min(0.9, (sharpe_ratio - 0.5) / 2))
        
        performance = {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'trend_strength': trend_strength,
            'learning_rate': np.random.uniform(0.001, 0.01),
            'convergence_epochs': np.random.randint(50, 200),
            'win_rate': np.random.uniform(0.4, 0.7),
            'profit_factor': np.random.uniform(0.8, 2.0),
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
        performance_history.append(performance)
    
    print(f"✅ Created {len(performance_history)} comprehensive performance records")
    return performance_history

def test_reinforcement_learning_with_logging():
    """Test Reinforcement Learning Agent with enhanced logging"""
    print("\n🤖 Testing Reinforcement Learning Agent with Enhanced Logging...")
    
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_debug, log_success
        from agents.learning.advanced_learning_methods_fixed import (
            ReinforcementLearningAgent, QLearningState, QLearningAction, AgentConfig
        )
        
        # Create enhanced logger
        logger = get_enhanced_logger("RL_Agent_Test")
        logger.start_timer("RL_Agent_Test")
        
        log_info(logger, "Starting Reinforcement Learning Agent test")
        
        # Create agent with custom config
        config = AgentConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            enable_logging=True,
            enable_caching=True
        )
        
        logger.log_configuration(config.__dict__)
        
        rl_agent = ReinforcementLearningAgent(config)
        log_success(logger, "ReinforcementLearningAgent initialized successfully")
        
        # Test multiple states and actions
        test_states = [
            QLearningState('bull', 'low', 0.6, 'normal', 'buy'),
            QLearningState('bear', 'high', 0.8, 'high', 'sell'),
            QLearningState('sideways', 'medium', 0.3, 'normal', 'hold'),
            QLearningState('bull', 'medium', 0.7, 'low', 'buy')
        ]
        
        test_actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0),
            QLearningAction('buy', 0.8, 0.03, 0.08),
            QLearningAction('sell', 0.3, 0.01, 0.03)
        ]
        
        log_info(logger, f"Testing {len(test_states)} states and {len(test_actions)} actions")
        
        # Test action selection for each state
        for i, state in enumerate(test_states):
            log_debug(logger, f"Testing state {i+1}: {state.market_regime} market, {state.volatility_level} volatility")
            
            action = rl_agent.choose_action(state, test_actions)
            log_debug(logger, f"Selected action: {action.action_type} with position size {action.position_size}")
            
            # Test Q-learning update
            next_state = QLearningState('bull', 'low', 0.7, 'normal', 'hold')
            reward = np.random.uniform(-0.1, 0.1)  # Random reward
            
            rl_agent.update_q_value(state, action, reward, next_state, test_actions)
            q_value = rl_agent.get_q_value(state, action)
            
            log_debug(logger, f"Q-value for state {i+1}: {q_value:.4f}")
        
        # Test epsilon decay
        initial_epsilon = rl_agent.epsilon
        rl_agent.decay_epsilon()
        log_debug(logger, f"Epsilon decayed from {initial_epsilon:.4f} to {rl_agent.epsilon:.4f}")
        
        # Test reward calculation
        test_action = QLearningAction('buy', 0.5, 0.02, 0.05)
        reward = rl_agent.calculate_reward(test_action, 0.05, -0.02, 0.015)
        log_debug(logger, f"Calculated reward: {reward:.4f}")
        
        # Test experience learning
        state = test_states[0]
        action = test_actions[0]
        next_state = test_states[1]
        reward = 0.03
        
        rl_agent.learn_from_experience(state, action, reward, next_state, test_actions)
        log_success(logger, "Experience learning completed successfully")
        
        # Log Q-table statistics
        q_table_size = len(rl_agent.q_table)
        log_info(logger, f"Q-table contains {q_table_size} state-action pairs")
        
        duration = logger.end_timer("RL_Agent_Test")
        log_success(logger, f"Reinforcement Learning test completed successfully in {duration:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RL test failed: {e}")
        if 'logger' in locals():
            logger.log_error_with_context(e, "Reinforcement Learning test")
        return False

def test_meta_learning_with_logging():
    """Test Meta-Learning Agent with enhanced logging"""
    print("\n🧠 Testing Meta-Learning Agent with Enhanced Logging...")
    
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_debug, log_success
        from agents.learning.advanced_learning_methods_fixed import MetaLearningAgent, AgentConfig
        
        # Create enhanced logger
        logger = get_enhanced_logger("Meta_Learning_Test")
        logger.start_timer("Meta_Learning_Test")
        
        log_info(logger, "Starting Meta-Learning Agent test")
        
        meta_agent = MetaLearningAgent()
        log_success(logger, "MetaLearningAgent initialized successfully")
        
        # Create comprehensive performance history
        performance_history = create_comprehensive_performance_history()
        logger.log_data_validation(performance_history, "performance_history")
        
        # Test meta-learning with multiple strategies
        strategies = ['momentum_strategy', 'mean_reversion_strategy', 'trend_following_strategy']
        
        for strategy_name in strategies:
            log_info(logger, f"Training meta-model for strategy: {strategy_name}")
            
            # Learn optimal strategy
            meta_agent.learn_optimal_strategy(strategy_name, performance_history)
            
            # Test parameter prediction
            current_state = {
                'volatility': np.random.uniform(0.01, 0.05),
                'trend_strength': np.random.uniform(0.1, 0.9),
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(-0.2, -0.01),
                'learning_rate': np.random.uniform(0.001, 0.01),
                'convergence_epochs': np.random.randint(50, 200)
            }
            
            optimal_params = meta_agent.predict_optimal_parameters(strategy_name, current_state)
            
            if optimal_params:
                logger.log_model_performance(f"{strategy_name}_meta_model", optimal_params)
                log_success(logger, f"Meta-learning completed for {strategy_name}")
            else:
                log_debug(logger, f"No optimal parameters predicted for {strategy_name}")
        
        # Test meta-feature extraction
        meta_features = meta_agent.extract_meta_features(performance_history)
        log_debug(logger, f"Extracted {len(meta_features)} meta-features")
        
        duration = logger.end_timer("Meta_Learning_Test")
        log_success(logger, f"Meta-Learning test completed successfully in {duration:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Meta-learning test failed: {e}")
        if 'logger' in locals():
            logger.log_error_with_context(e, "Meta-Learning test")
        return False

def test_transfer_learning_with_logging():
    """Test Transfer Learning Agent with enhanced logging"""
    print("\n🔄 Testing Transfer Learning Agent with Enhanced Logging...")
    
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_debug, log_success
        from agents.learning.advanced_learning_methods_fixed import TransferLearningAgent, AgentConfig
        
        # Create enhanced logger
        logger = get_enhanced_logger("Transfer_Learning_Test")
        logger.start_timer("Transfer_Learning_Test")
        
        log_info(logger, "Starting Transfer Learning Agent test")
        
        transfer_agent = TransferLearningAgent()
        log_success(logger, "TransferLearningAgent initialized successfully")
        
        # Create market data
        market_data = create_realistic_market_data()
        logger.log_data_validation(market_data, "market_data")
        
        # Split data into multiple periods for transfer learning
        periods = [
            ('period_1', market_data.iloc[:100]),
            ('period_2', market_data.iloc[100:200]),
            ('period_3', market_data.iloc[200:300]),
            ('period_4', market_data.iloc[300:400])
        ]
        
        # Train source models on different periods
        for period_name, period_data in periods[:-1]:
            log_info(logger, f"Training source model on {period_name}")
            
            # Prepare features
            period_data = period_data.copy()
            period_data['feature1'] = period_data['close'].pct_change()
            period_data['feature2'] = period_data['volume'].pct_change()
            period_data['feature3'] = period_data['rsi']
            period_data['feature4'] = period_data['macd']
            period_data['feature5'] = period_data['price_momentum']
            
            transfer_agent.train_source_model(period_name, period_data, 'target')
            log_success(logger, f"Source model trained for {period_name}")
        
        # Test transfer to target period
        target_period_name, target_data = periods[-1]
        log_info(logger, f"Testing transfer to {target_period_name}")
        
        # Prepare target data
        target_data = target_data.copy()
        target_data['feature1'] = target_data['close'].pct_change()
        target_data['feature2'] = target_data['volume'].pct_change()
        target_data['feature3'] = target_data['rsi']
        target_data['feature4'] = target_data['macd']
        target_data['feature5'] = target_data['price_momentum']
        
        # Test transfer from each source period
        for period_name, _ in periods[:-1]:
            log_info(logger, f"Transferring from {period_name} to {target_period_name}")
            
            transfer_agent.adapt_to_target_market(period_name, target_period_name, target_data, 'target')
            
            # Get transfer recommendations
            recommendations = transfer_agent.get_transfer_recommendations()
            
            if recommendations:
                logger.log_recommendations(recommendations)
                log_success(logger, f"Transfer completed from {period_name}")
            else:
                log_debug(logger, f"No transfer recommendations for {period_name}")
        
        # Test memory management
        log_info(logger, "Testing memory management")
        transfer_agent.cleanup_old_models()
        log_success(logger, "Memory cleanup completed")
        
        duration = logger.end_timer("Transfer_Learning_Test")
        log_success(logger, f"Transfer Learning test completed successfully in {duration:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Transfer learning test failed: {e}")
        if 'logger' in locals():
            logger.log_error_with_context(e, "Transfer Learning test")
        return False

def test_online_learning_with_logging():
    """Test Online Learning Agent with enhanced logging"""
    print("\n📈 Testing Online Learning Agent with Enhanced Logging...")
    
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_debug, log_success
        from agents.learning.advanced_learning_methods_fixed import OnlineLearningAgent, AgentConfig
        
        # Create enhanced logger
        logger = get_enhanced_logger("Online_Learning_Test")
        logger.start_timer("Online_Learning_Test")
        
        log_info(logger, "Starting Online Learning Agent test")
        
        online_agent = OnlineLearningAgent(base_model_type='linear', learning_rate=0.01)
        log_success(logger, "OnlineLearningAgent initialized successfully")
        
        # Create market data
        market_data = create_realistic_market_data()
        logger.log_data_validation(market_data, "market_data")
        
        # Test online learning with multiple models
        model_names = ['momentum_model', 'mean_reversion_model', 'trend_model']
        
        for model_name in model_names:
            log_info(logger, f"Testing online learning for {model_name}")
            
            # Create online model
            online_agent.create_online_model(model_name)
            log_success(logger, f"Online model created: {model_name}")
            
            # Simulate online updates with different data windows
            for window_size in [20, 30, 40]:
                log_debug(logger, f"Updating {model_name} with {window_size}-day window")
                
                # Prepare data for online update
                recent_data = market_data.tail(window_size).copy()
                recent_data['feature1'] = recent_data['close'].pct_change()
                recent_data['feature2'] = recent_data['volume'].pct_change()
                recent_data['feature3'] = recent_data['rsi']
                recent_data['feature4'] = recent_data['macd']
                recent_data['feature5'] = recent_data['price_momentum']
                
                # Calculate performance metric
                returns = recent_data['close'].pct_change().dropna()
                performance_metric = returns.mean() / returns.std() if returns.std() > 0 else 0
                
                # Update online model
                online_agent.update_online_model(model_name, recent_data, 'target', performance_metric)
                
                # Get performance metrics
                performance = online_agent.get_online_performance(model_name)
                
                if performance:
                    logger.log_model_performance(f"{model_name}_window_{window_size}", performance)
                
                log_debug(logger, f"Updated {model_name} with performance metric: {performance_metric:.4f}")
            
            # Test convergence
            convergence_status = online_agent.check_convergence(model_name, performance_metric)
            log_info(logger, f"Convergence status for {model_name}: {convergence_status}")
        
        # Test predictions
        test_features = np.random.randn(10, 5)  # 10 samples, 5 features
        for model_name in model_names:
            predictions = online_agent.predict_online(model_name, test_features)
            log_debug(logger, f"Predictions from {model_name}: {predictions[:3]}...")  # Show first 3
        
        duration = logger.end_timer("Online_Learning_Test")
        log_success(logger, f"Online Learning test completed successfully in {duration:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Online learning test failed: {e}")
        if 'logger' in locals():
            logger.log_error_with_context(e, "Online Learning test")
        return False

def test_full_orchestrator_with_logging():
    """Test the complete Learning Agent Orchestrator with enhanced logging"""
    print("\n🎯 Testing Complete Learning Agent Orchestrator with Enhanced Logging...")
    
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_debug, log_success
        from agents.learning.advanced_learning_methods_fixed import AdvancedLearningOrchestrator, AgentConfig
        
        # Create enhanced logger
        logger = get_enhanced_logger("Orchestrator_Test")
        logger.start_timer("Orchestrator_Test")
        
        log_info(logger, "Starting Complete Learning Agent Orchestrator test")
        
        # Create orchestrator with custom config
        config = AgentConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
            enable_logging=True,
            enable_caching=True,
            max_models=15
        )
        
        logger.log_configuration(config.__dict__)
        
        orchestrator = AdvancedLearningOrchestrator(config)
        log_success(logger, "AdvancedLearningOrchestrator initialized successfully")
        
        # Create comprehensive test data
        market_data = create_realistic_market_data()
        performance_history = create_comprehensive_performance_history()
        
        logger.log_data_validation(market_data, "market_data")
        logger.log_data_validation(performance_history, "performance_history")
        
        # Run complete optimization
        log_info(logger, "Running comprehensive strategy optimization")
        results = orchestrator.optimize_strategy(market_data, performance_history)
        
        log_success(logger, "Strategy optimization completed")
        
        # Log detailed results
        log_info(logger, "Analyzing optimization results:")
        
        # RL results
        rl_results = results.get('reinforcement_learning', {})
        if rl_results:
            logger.log_model_performance("Reinforcement_Learning", rl_results)
        
        # Meta-learning results
        meta_results = results.get('meta_learning', {})
        if meta_results:
            logger.log_model_performance("Meta_Learning", meta_results)
        
        # Transfer learning results
        transfer_results = results.get('transfer_learning', {})
        if transfer_results:
            transfer_recs = transfer_results.get('transfer_recommendations', [])
            log_info(logger, f"Transfer learning generated {len(transfer_recs)} recommendations")
        
        # Online learning results
        online_results = results.get('online_learning', {})
        if online_results:
            logger.log_model_performance("Online_Learning", online_results)
        
        # Combined recommendations
        recommendations = results.get('recommendations', [])
        logger.log_recommendations(recommendations)
        
        # Log performance summary
        logger.log_performance_summary()
        
        duration = logger.end_timer("Orchestrator_Test")
        log_success(logger, f"Complete Orchestrator test completed successfully in {duration:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Orchestrator test failed: {e}")
        if 'logger' in locals():
            logger.log_error_with_context(e, "Complete Orchestrator test")
        return False

def main():
    """Run comprehensive Learning Agent tests with enhanced logging"""
    print("🧠 COMPREHENSIVE LEARNING AGENT TEST WITH ENHANCED LOGGING")
    print("=" * 70)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create main logger
    try:
        from agents.learning.enhanced_logging import get_enhanced_logger, log_info, log_success
        main_logger = get_enhanced_logger("Comprehensive_Test")
        main_logger.start_timer("Comprehensive_Test")
        
        log_info(main_logger, "Starting comprehensive Learning Agent test suite")
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced logging: {e}")
        main_logger = None
    
    test_results = []
    
    # Test individual components with enhanced logging
    test_results.append(("Reinforcement Learning", test_reinforcement_learning_with_logging()))
    test_results.append(("Meta-Learning", test_meta_learning_with_logging()))
    test_results.append(("Transfer Learning", test_transfer_learning_with_logging()))
    test_results.append(("Online Learning", test_online_learning_with_logging()))
    
    # Test complete orchestrator
    test_results.append(("Complete Orchestrator", test_full_orchestrator_with_logging()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Learning Agent is working perfectly!")
        if main_logger:
            log_success(main_logger, "All comprehensive tests passed successfully")
    elif passed > total // 2:
        print("⚠️ Most tests passed. Some components may need attention.")
        if main_logger:
            log_info(main_logger, f"Most tests passed ({passed}/{total})")
    else:
        print("❌ Multiple tests failed. System needs debugging.")
        if main_logger:
            log_info(main_logger, f"Multiple tests failed ({passed}/{total})")
    
    # Log final summary
    if main_logger:
        duration = main_logger.end_timer("Comprehensive_Test")
        log_success(main_logger, f"Comprehensive test suite completed in {duration:.4f} seconds")
        main_logger.log_performance_summary()
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Check the 'logs' directory for detailed log files:")
    print("   - Comprehensive_Test_YYYYMMDD.log (detailed logs)")
    print("   - Comprehensive_Test_errors_YYYYMMDD.log (error logs)")
    print("   - Comprehensive_Test_performance_YYYYMMDD.log (performance logs)")

if __name__ == "__main__":
    main()
