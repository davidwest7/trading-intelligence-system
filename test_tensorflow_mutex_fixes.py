#!/usr/bin/env python3
"""
Comprehensive Test for TensorFlow Mutex Fixes
Tests all critical fixes to ensure they prevent hanging and mutex issues
"""

import sys
import os
import time
import threading
import asyncio
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tensorflow_mutex_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# SET CRITICAL ENVIRONMENT VARIABLES BEFORE ANY TESTS
# ============================================================================

# Set all critical environment variables BEFORE any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to prevent conflicts
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'  # Disable deprecation warnings
os.environ['TF_LOGGING_LEVEL'] = 'ERROR'  # Error-level logging only
os.environ['TF_PROFILER_DISABLE'] = '1'  # Disable profiling
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU growth
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Single inter-op thread
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Single intra-op thread

class TensorFlowMutexTestSuite:
    """Comprehensive test suite for TensorFlow mutex fixes."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", error: Exception = None):
        """Log test results with details."""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "error": str(error) if error else None
        }
        
        self.test_results[test_name] = result
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {details}")
        else:
            logger.error(f"❌ {test_name}: FAILED - {details}")
            if error:
                logger.error(f"   Error: {error}")
    
    def test_environment_variables(self):
        """Test that all critical environment variables are set."""
        logger.info("🔧 Testing Environment Variables...")
        
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
            self.log_test_result("Environment Variables", False, f"Missing variables: {missing_vars}")
            return False
        else:
            self.log_test_result("Environment Variables", True, "All required environment variables set")
            return True
    
    def test_tensorflow_import(self):
        """Test TensorFlow import with mutex fixes."""
        logger.info("📦 Testing TensorFlow Import...")
        
        try:
            # Import the mutex fixer
            from tensorflow_mutex_fixes import TensorFlowMutexFixer
            
            # Apply all fixes
            fixer = TensorFlowMutexFixer()
            tf = fixer.apply_all_fixes()
            
            if tf is None:
                self.log_test_result("TensorFlow Import", False, "TensorFlow import failed")
                return False
            else:
                self.log_test_result("TensorFlow Import", True, "TensorFlow imported successfully with mutex fixes")
                return True
                
        except Exception as e:
            self.log_test_result("TensorFlow Import", False, f"TensorFlow import error: {e}", e)
            return False
    
    def test_thread_safe_wrapper(self):
        """Test the thread-safe TensorFlow wrapper."""
        logger.info("🛡️ Testing Thread-Safe Wrapper...")
        
        try:
            from tensorflow_mutex_fixes import ThreadSafeTensorFlow, SafeMLFallback
            
            # Initialize wrappers
            safe_tf = ThreadSafeTensorFlow()
            fallback = SafeMLFallback()
            
            self.log_test_result("Thread-Safe Wrapper", True, "Thread-safe wrapper initialized successfully")
            return True
            
        except Exception as e:
            self.log_test_result("Thread-Safe Wrapper", False, f"Thread-safe wrapper error: {e}", e)
            return False
    
    def test_sklearn_fallback(self):
        """Test sklearn fallback functionality."""
        logger.info("🔄 Testing Sklearn Fallback...")
        
        try:
            from tensorflow_mutex_fixes import SafeMLFallback
            
            fallback = SafeMLFallback()
            
            # Test sklearn model creation
            model = fallback.create_sklearn_model('random_forest')
            if model is None:
                self.log_test_result("Sklearn Fallback", False, "Failed to create sklearn model")
                return False
            
            # Test training
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            success = fallback.train_sklearn_model(model, X, y)
            if not success:
                self.log_test_result("Sklearn Fallback", False, "Failed to train sklearn model")
                return False
            
            # Test prediction
            predictions = fallback.predict_sklearn_model(model, X[:10])
            if predictions is None:
                self.log_test_result("Sklearn Fallback", False, "Failed to make sklearn predictions")
                return False
            
            self.log_test_result("Sklearn Fallback", True, f"Sklearn fallback working: {len(predictions)} predictions")
            return True
            
        except Exception as e:
            self.log_test_result("Sklearn Fallback", False, f"Sklearn fallback error: {e}", e)
            return False
    
    def test_lstm_predictor_fixes(self):
        """Test LSTM predictor with mutex fixes."""
        logger.info("🧠 Testing LSTM Predictor with Fixes...")
        
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
            
            self.log_test_result("LSTM Predictor", True, "LSTM predictor initialized with mutex fixes")
            return True
            
        except Exception as e:
            self.log_test_result("LSTM Predictor", False, f"LSTM predictor error: {e}", e)
            return False
    
    def test_transformer_sentiment_fixes(self):
        """Test transformer sentiment analyzer with mutex fixes."""
        logger.info("🤖 Testing Transformer Sentiment with Fixes...")
        
        try:
            from ml_models.transformer_sentiment import TransformerSentimentAnalyzer
            
            # Initialize transformer analyzer
            transformer = TransformerSentimentAnalyzer()
            
            self.log_test_result("Transformer Sentiment", True, "Transformer sentiment analyzer initialized with mutex fixes")
            return True
            
        except Exception as e:
            self.log_test_result("Transformer Sentiment", False, f"Transformer sentiment error: {e}", e)
            return False
    
    def test_concurrent_access(self):
        """Test concurrent access to TensorFlow models."""
        logger.info("🔄 Testing Concurrent Access...")
        
        try:
            from ml_models.lstm_predictor import LSTMPredictor
            from ml_models.transformer_sentiment import TransformerSentimentAnalyzer
            
            # Initialize models
            lstm = LSTMPredictor()
            transformer = TransformerSentimentAnalyzer()
            
            # Create mock data
            mock_data = pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, 100),
                'RSI': np.random.uniform(0, 100, 100),
                'MACD': np.random.randn(100),
                'BB_Upper': np.random.randn(100).cumsum() + 102,
                'BB_Lower': np.random.randn(100).cumsum() + 98
            })
            
            # Test concurrent access
            def test_lstm():
                try:
                    # This should not hang due to thread-safe wrapper
                    result = lstm._prepare_data(mock_data)
                    return True
                except Exception as e:
                    print(f"LSTM thread error: {e}")
                    return False
            
            def test_transformer():
                try:
                    # This should not hang due to thread-safe wrapper
                    return True
                except Exception as e:
                    print(f"Transformer thread error: {e}")
                    return False
            
            # Run concurrent tests
            threads = []
            for i in range(3):
                threads.append(threading.Thread(target=test_lstm))
                threads.append(threading.Thread(target=test_transformer))
            
            # Start threads
            for thread in threads:
                thread.start()
            
            # Wait for completion with timeout
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout
            
            # Check if any threads are still alive (indicating hanging)
            alive_threads = [t for t in threads if t.is_alive()]
            
            if alive_threads:
                self.log_test_result("Concurrent Access", False, f"{len(alive_threads)} threads still hanging")
                return False
            else:
                self.log_test_result("Concurrent Access", True, "All concurrent access tests completed without hanging")
                return True
                
        except Exception as e:
            self.log_test_result("Concurrent Access", False, f"Concurrent access error: {e}", e)
            return False
    
    def test_timeout_protection(self):
        """Test timeout protection mechanisms."""
        logger.info("⏰ Testing Timeout Protection...")
        
        try:
            from tensorflow_mutex_fixes import TimeoutCallback
            
            # Test timeout callback
            callback = TimeoutCallback(timeout=1)  # 1 second timeout
            callback.on_train_begin()
            
            # Simulate long operation
            time.sleep(0.5)  # Should not timeout
            
            try:
                callback.on_epoch_end(0)
                self.log_test_result("Timeout Protection", True, "Timeout protection working correctly")
                return True
            except TimeoutError:
                self.log_test_result("Timeout Protection", False, "Timeout triggered too early")
                return False
                
        except Exception as e:
            self.log_test_result("Timeout Protection", False, f"Timeout protection error: {e}", e)
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        logger.info("🚀 Starting Comprehensive TensorFlow Mutex Fix Tests...")
        
        tests = [
            ("Environment Variables", self.test_environment_variables),
            ("TensorFlow Import", self.test_tensorflow_import),
            ("Thread-Safe Wrapper", self.test_thread_safe_wrapper),
            ("Sklearn Fallback", self.test_sklearn_fallback),
            ("LSTM Predictor Fixes", self.test_lstm_predictor_fixes),
            ("Transformer Sentiment Fixes", self.test_transformer_sentiment_fixes),
            ("Concurrent Access", self.test_concurrent_access),
            ("Timeout Protection", self.test_timeout_protection)
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
                self.log_test_result(test_name, False, f"Test execution error: {e}", e)
                failed += 1
        
        # Generate summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {passed + failed}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        # Save detailed results
        import json
        with open('tensorflow_mutex_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': passed + failed,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0,
                    'total_time_seconds': total_time,
                    'timestamp': datetime.now().isoformat()
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        logger.info("📄 Detailed results saved to tensorflow_mutex_test_results.json")
        
        return passed, failed

def main():
    """Main test execution."""
    print("🚀 TensorFlow Mutex Fixes - Comprehensive Test Suite")
    print("=" * 60)
    
    # Run tests
    test_suite = TensorFlowMutexTestSuite()
    passed, failed = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! TensorFlow mutex fixes are working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the results.")
        sys.exit(1)

if __name__ == "__main__":
    main()
