#!/usr/bin/env python3
"""
Test Backtesting System
=======================

Test script to verify the backtesting system components work correctly.
This script tests each component individually and then runs a simple backtest.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_polygon_client():
    """Test Polygon client functionality"""
    logger.info("🧪 Testing Polygon Client...")
    
    try:
        from backtesting.polygon_client import PolygonClient, PolygonConfig
        
        # Check if API key is available
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            logger.warning("⚠️ POLYGON_API_KEY not set, skipping Polygon client test")
            return False
        
        # Initialize client
        config = PolygonConfig(api_key=api_key)
        client = PolygonClient(config)
        
        # Test health check
        health = client.health_check()
        logger.info(f"Health check result: {health}")
        
        if health.get('status') == 'healthy':
            logger.info("✅ Polygon client test passed")
            return True
        else:
            logger.error("❌ Polygon client health check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Polygon client test failed: {e}")
        return False

def test_data_ingestion():
    """Test data ingestion components"""
    logger.info("🧪 Testing Data Ingestion...")
    
    try:
        from backtesting.data_ingestion import DataValidator, S3Storage
        
        # Test data validator
        validator = DataValidator()
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Test validation
        is_valid = validator.validate_bars_df(test_data)
        logger.info(f"Data validation result: {is_valid}")
        
        if is_valid:
            logger.info("✅ Data ingestion test passed")
            return True
        else:
            logger.error("❌ Data validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data ingestion test failed: {e}")
        return False

def test_execution_engine():
    """Test execution engine"""
    logger.info("🧪 Testing Execution Engine...")
    
    try:
        from backtesting.execution import ExecutionEngine, ExecutionConfig
        
        # Create execution config
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        # Create test data
        weights = pd.DataFrame({
            'AAPL': [0.3, 0.4, 0.5],
            'MSFT': [0.3, 0.3, 0.3],
            'GOOGL': [0.4, 0.3, 0.2]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        prices = pd.DataFrame({
            'AAPL': [150, 155, 160],
            'MSFT': [300, 305, 310],
            'GOOGL': [2500, 2550, 2600]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        volumes = pd.DataFrame({
            'AAPL': [1000000, 1100000, 1200000],
            'MSFT': [800000, 850000, 900000],
            'GOOGL': [500000, 550000, 600000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Test execution
        results = engine.execute_trades(
            target_weights=weights,
            current_weights=weights * 0.9,  # Slightly different
            prices=prices,
            volumes=volumes,
            portfolio_value=1000000
        )
        
        logger.info(f"Execution results: {results.keys()}")
        logger.info("✅ Execution engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Execution engine test failed: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    logger.info("🧪 Testing Metrics Calculation...")
    
    try:
        from backtesting.metrics import BacktestMetrics
        
        # Create test portfolio data
        portfolio_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=252, freq='D'),
            'total_value': np.random.randn(252).cumsum() + 1000000
        })
        
        # Calculate metrics
        metrics_calc = BacktestMetrics()
        metrics = metrics_calc.calculate_metrics(portfolio_data)
        
        logger.info(f"Calculated {len(metrics)} metrics")
        logger.info(f"Sample metrics: {list(metrics.keys())[:5]}")
        
        logger.info("✅ Metrics calculation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Metrics calculation test failed: {e}")
        return False

def test_backtest_engine():
    """Test backtest engine with mock data"""
    logger.info("🧪 Testing Backtest Engine...")
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        from backtesting.execution import ExecutionConfig
        
        # Create mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 150,
            'MSFT': np.random.randn(100).cumsum() + 300,
            'GOOGL': np.random.randn(100).cumsum() + 2500,
            'SPY': np.random.randn(100).cumsum() + 400
        }, index=dates)
        
        # Create execution config
        execution_config = ExecutionConfig()
        
        # Create backtest config
        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2024-01-01",
            end_date="2024-04-10",
            timeframe="1d",
            initial_capital=1000000.0,
            rebalance_frequency="1w",
            execution_config=execution_config,
            polygon_api_key=None,  # Use mock data
            strategy_function=lambda data, date, prices, **kwargs: pd.Series(1.0/3, index=["AAPL", "MSFT", "GOOGL"])
        )
        
        # Initialize engine
        engine = BacktestEngine(config)
        
        # Mock the data loading
        engine._load_data = lambda: mock_data
        
        # Run backtest
        results = engine.run_backtest(max_retries=1)
        
        logger.info(f"Backtest completed with {len(results.get('portfolio_history', []))} portfolio records")
        logger.info("✅ Backtest engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtest engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and retry mechanisms"""
    logger.info("🧪 Testing Error Handling...")
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        from backtesting.execution import ExecutionConfig
        
        # Create config that will cause errors
        execution_config = ExecutionConfig()
        config = BacktestConfig(
            symbols=["INVALID_SYMBOL"],
            start_date="2024-01-01",
            end_date="2024-04-10",
            timeframe="1d",
            initial_capital=1000000.0,
            rebalance_frequency="1w",
            execution_config=execution_config,
            polygon_api_key="invalid_key"
        )
        
        # Initialize engine
        engine = BacktestEngine(config)
        
        # Test error handling
        try:
            results = engine.run_backtest(max_retries=2)
            logger.warning("⚠️ Expected error but backtest succeeded")
            return False
        except Exception as e:
            logger.info(f"✅ Expected error caught: {type(e).__name__}")
            
            # Check error log
            error_summary = engine.get_error_summary()
            logger.info(f"Error summary: {error_summary}")
            
            if error_summary.get('total_errors', 0) > 0:
                logger.info("✅ Error handling test passed")
                return True
            else:
                logger.error("❌ No errors logged")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    logger.info("🚀 Starting Comprehensive Backtesting System Tests")
    
    tests = [
        ("Polygon Client", test_polygon_client),
        ("Data Ingestion", test_data_ingestion),
        ("Execution Engine", test_execution_engine),
        ("Metrics Calculation", test_metrics_calculation),
        ("Backtest Engine", test_backtest_engine),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Backtesting system is ready.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the logs.")
    
    return passed == total

def test_with_real_data():
    """Test with real data if API key is available"""
    logger.info("🧪 Testing with Real Data...")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.warning("⚠️ POLYGON_API_KEY not set, skipping real data test")
        return False
    
    try:
        from backtesting.engine import BacktestEngine, BacktestConfig
        from backtesting.execution import ExecutionConfig
        
        # Create config for real data test
        execution_config = ExecutionConfig()
        config = BacktestConfig(
            symbols=["AAPL", "MSFT", "SPY"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            timeframe="1d",
            initial_capital=1000000.0,
            rebalance_frequency="1w",
            execution_config=execution_config,
            polygon_api_key=api_key,
            strategy_function=lambda data, date, prices, **kwargs: pd.Series(1.0/3, index=["AAPL", "MSFT", "SPY"])
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest(max_retries=2)
        
        logger.info(f"✅ Real data test completed successfully")
        logger.info(f"Portfolio records: {len(results.get('portfolio_history', []))}")
        logger.info(f"Performance metrics: {list(results.get('performance_metrics', {}).keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real data test failed: {e}")
        return False

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_test()
    
    # If basic tests pass, try real data test
    if success:
        logger.info("\n" + "="*50)
        logger.info("Testing with Real Data")
        logger.info("="*50)
        test_with_real_data()
    
    logger.info("\n🎉 Testing completed!")
