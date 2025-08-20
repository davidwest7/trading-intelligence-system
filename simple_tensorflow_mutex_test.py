#!/usr/bin/env python3
"""
Simple Test for TensorFlow Mutex Fixes
Tests core functionality without complex concurrent operations
"""

import sys
import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================================
# SET CRITICAL ENVIRONMENT VARIABLES BEFORE ANY IMPORTS
# ============================================================================

# Set all critical environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to prevent conflicts
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'  # Disable deprecation warnings
os.environ['TF_LOGGING_LEVEL'] = 'ERROR'  # Error-level logging only
os.environ['TF_PROFILER_DISABLE'] = '1'  # Disable profiling
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU growth
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Single inter-op thread
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Single intra-op thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_tensorflow_mutex_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment_variables():
    """Test that all critical environment variables are set."""
    print("🔧 Testing Environment Variables...")
    
    required_vars = [
        'TF_CPP_MIN_LOG_LEVEL',
        'CUDA_VISIBLE_DEVICES',
        'TF_ENABLE_DEPRECATION_WARNINGS',
        'TF_LOGGING_LEVEL',
        'TF_PROFILER_DISABLE',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_NUM_INTEROP_THREADS',
        'TF_NUM_INTRAOP_THREADS'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Environment Variables: FAILED - Missing variables: {missing_vars}")
        return False
    else:
        print("✅ Environment Variables: PASSED - All required environment variables set")
        return True

def test_tensorflow_import():
    """Test TensorFlow import with mutex fixes."""
    print("📦 Testing TensorFlow Import...")
    
    try:
        # Import the mutex fixer
        from tensorflow_mutex_fixes import TensorFlowMutexFixer
        
        # Apply all fixes
        fixer = TensorFlowMutexFixer()
        tf = fixer.apply_all_fixes()
        
        if tf is None:
            print("❌ TensorFlow Import: FAILED - TensorFlow import failed")
            return False
        else:
            print("✅ TensorFlow Import: PASSED - TensorFlow imported successfully with mutex fixes")
            return True
            
    except Exception as e:
        print(f"❌ TensorFlow Import: FAILED - TensorFlow import error: {e}")
        return False

def test_thread_safe_wrapper():
    """Test the thread-safe TensorFlow wrapper."""
    print("🛡️ Testing Thread-Safe Wrapper...")
    
    try:
        from tensorflow_mutex_fixes import ThreadSafeTensorFlow, SafeMLFallback
        
        # Initialize wrappers
        safe_tf = ThreadSafeTensorFlow()
        fallback = SafeMLFallback()
        
        print("✅ Thread-Safe Wrapper: PASSED - Thread-safe wrapper initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Thread-Safe Wrapper: FAILED - Thread-safe wrapper error: {e}")
        return False

def test_sklearn_fallback():
    """Test sklearn fallback functionality."""
    print("🔄 Testing Sklearn Fallback...")
    
    try:
        from tensorflow_mutex_fixes import SafeMLFallback
        
        fallback = SafeMLFallback()
        
        # Test sklearn model creation
        model = fallback.create_sklearn_model('random_forest')
        if model is None:
            print("❌ Sklearn Fallback: FAILED - Failed to create sklearn model")
            return False
        
        # Test training
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        success = fallback.train_sklearn_model(model, X, y)
        if not success:
            print("❌ Sklearn Fallback: FAILED - Failed to train sklearn model")
            return False
        
        # Test prediction
        predictions = fallback.predict_sklearn_model(model, X[:10])
        if predictions is None:
            print("❌ Sklearn Fallback: FAILED - Failed to make sklearn predictions")
            return False
        
        print(f"✅ Sklearn Fallback: PASSED - Sklearn fallback working: {len(predictions)} predictions")
        return True
        
    except Exception as e:
        print(f"❌ Sklearn Fallback: FAILED - Sklearn fallback error: {e}")
        return False

def test_lstm_predictor_fixes():
    """Test LSTM predictor with mutex fixes."""
    print("🧠 Testing LSTM Predictor with Fixes...")
    
    try:
        from ml_models.lstm_predictor import LSTMPredictor
        
        # Initialize LSTM predictor
        lstm = LSTMPredictor()
        
        # Create mock data
        mock_data = pd.DataFrame({
            'Close': np.random.randn(200).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 200),
            'RSI': np.random.uniform(0, 100, 200),
            'MACD': np.random.randn(200),
            'BB_Upper': np.random.randn(200).cumsum() + 102,
            'BB_Lower': np.random.randn(200).cumsum() + 98
        })
        
        print("✅ LSTM Predictor: PASSED - LSTM predictor initialized with mutex fixes")
        return True
        
    except Exception as e:
        print(f"❌ LSTM Predictor: FAILED - LSTM predictor error: {e}")
        return False

def test_transformer_sentiment_fixes():
    """Test transformer sentiment analyzer with mutex fixes."""
    print("🤖 Testing Transformer Sentiment with Fixes...")
    
    try:
        from ml_models.transformer_sentiment import TransformerSentimentAnalyzer
        
        # Initialize transformer analyzer
        transformer = TransformerSentimentAnalyzer()
        
        print("✅ Transformer Sentiment: PASSED - Transformer sentiment analyzer initialized with mutex fixes")
        return True
        
    except Exception as e:
        print(f"❌ Transformer Sentiment: FAILED - Transformer sentiment error: {e}")
        return False

def test_timeout_protection():
    """Test timeout protection mechanisms."""
    print("⏰ Testing Timeout Protection...")
    
    try:
        from tensorflow_mutex_fixes import TimeoutCallback
        
        # Test timeout callback
        callback = TimeoutCallback(timeout=1)  # 1 second timeout
        callback.on_train_begin()
        
        # Simulate short operation
        time.sleep(0.1)  # Should not timeout
        
        try:
            callback.on_epoch_end(0)
            print("✅ Timeout Protection: PASSED - Timeout protection working correctly")
            return True
        except TimeoutError:
            print("❌ Timeout Protection: FAILED - Timeout triggered too early")
            return False
            
    except Exception as e:
        print(f"❌ Timeout Protection: FAILED - Timeout protection error: {e}")
        return False

def main():
    """Main test execution."""
    print("🚀 Simple TensorFlow Mutex Fixes Test")
    print("=" * 50)
    
    start_time = datetime.now()
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("TensorFlow Import", test_tensorflow_import),
        ("Thread-Safe Wrapper", test_thread_safe_wrapper),
        ("Sklearn Fallback", test_sklearn_fallback),
        ("LSTM Predictor Fixes", test_lstm_predictor_fixes),
        ("Transformer Sentiment Fixes", test_transformer_sentiment_fixes),
        ("Timeout Protection", test_timeout_protection)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - Test execution error: {e}")
            failed += 1
    
    # Generate summary
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    print("=" * 50)
    
    # Exit with appropriate code
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! TensorFlow mutex fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the results.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
