#!/usr/bin/env python3
"""
Test Sklearn-Only Solution (No TensorFlow)
Tests the fallback solution that avoids TensorFlow mutex issues entirely
"""

import sys
import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sklearn_only_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SklearnOnlyTestSuite:
    """Test suite for sklearn-only solutions (no TensorFlow)."""
    
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
    
    def test_sklearn_import(self):
        """Test sklearn import and basic functionality."""
        logger.info("📦 Testing Sklearn Import...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.svm import SVR, SVC
            from sklearn.metrics import mean_squared_error, accuracy_score
            from sklearn.preprocessing import StandardScaler
            
            self.log_test_result("Sklearn Import", True, "All sklearn components imported successfully")
            return True
            
        except Exception as e:
            self.log_test_result("Sklearn Import", False, f"Sklearn import error: {e}", e)
            return False
    
    def test_sklearn_regression(self):
        """Test sklearn regression models."""
        logger.info("📈 Testing Sklearn Regression...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            # Create mock data
            X = np.random.randn(1000, 10)
            y = np.random.randn(1000)
            
            # Test Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X[:10])
            
            # Test Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(X[:10])
            
            self.log_test_result("Sklearn Regression", True, f"RF and LR models trained, {len(rf_pred)} predictions each")
            return True
            
        except Exception as e:
            self.log_test_result("Sklearn Regression", False, f"Sklearn regression error: {e}", e)
            return False
    
    def test_sklearn_classification(self):
        """Test sklearn classification models."""
        logger.info("🎯 Testing Sklearn Classification...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            
            # Create mock data
            X = np.random.randn(1000, 10)
            y = np.random.randint(0, 3, 1000)  # 3 classes
            
            # Test Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X[:10])
            
            # Test Logistic Regression
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(X[:10])
            
            self.log_test_result("Sklearn Classification", True, f"RF and LR classifiers trained, {len(rf_pred)} predictions each")
            return True
            
        except Exception as e:
            self.log_test_result("Sklearn Classification", False, f"Sklearn classification error: {e}", e)
            return False
    
    def test_ml_models_advanced(self):
        """Test advanced ML models without TensorFlow."""
        logger.info("🧠 Testing Advanced ML Models (No TensorFlow)...")
        
        try:
            from ml_models.advanced_ml_models import AdvancedMLPredictor
            from ml_models.ensemble_predictor import EnsemblePredictor
            
            # Initialize models
            advanced_models = AdvancedMLPredictor()
            ensemble = EnsemblePredictor()
            
            # Create mock data
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            # Test ensemble training
            if hasattr(ensemble, 'train'):
                ensemble.train(X, y)
                predictions = ensemble.predict(X[:10])
                self.log_test_result("Advanced ML Models", True, f"Ensemble model trained, {len(predictions)} predictions")
            else:
                self.log_test_result("Advanced ML Models", True, "Advanced ML models initialized successfully")
            
            return True
            
        except Exception as e:
            self.log_test_result("Advanced ML Models", False, f"Advanced ML models error: {e}", e)
            return False
    
    def test_sklearn_fallback_implementation(self):
        """Test the sklearn fallback implementation."""
        logger.info("🔄 Testing Sklearn Fallback Implementation...")
        
        try:
            # Import the fallback class
            from tensorflow_mutex_fixes import SafeMLFallback
            
            fallback = SafeMLFallback()
            
            # Test different model types
            model_types = ['random_forest', 'linear', 'svm']
            
            for model_type in model_types:
                model = fallback.create_sklearn_model(model_type)
                if model is None:
                    self.log_test_result("Sklearn Fallback", False, f"Failed to create {model_type} model")
                    return False
            
            # Test training and prediction
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            model = fallback.create_sklearn_model('random_forest')
            success = fallback.train_sklearn_model(model, X, y)
            
            if not success:
                self.log_test_result("Sklearn Fallback", False, "Failed to train sklearn model")
                return False
            
            predictions = fallback.predict_sklearn_model(model, X[:10])
            if predictions is None:
                self.log_test_result("Sklearn Fallback", False, "Failed to make predictions")
                return False
            
            self.log_test_result("Sklearn Fallback", True, f"Fallback working: {len(predictions)} predictions")
            return True
            
        except Exception as e:
            self.log_test_result("Sklearn Fallback", False, f"Sklearn fallback error: {e}", e)
            return False
    
    def test_data_preprocessing(self):
        """Test data preprocessing without TensorFlow."""
        logger.info("🔧 Testing Data Preprocessing...")
        
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            from sklearn.model_selection import train_test_split
            
            # Create mock financial data
            data = pd.DataFrame({
                'Close': np.random.randn(1000).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, 1000),
                'RSI': np.random.uniform(0, 100, 1000),
                'MACD': np.random.randn(1000),
                'BB_Upper': np.random.randn(1000).cumsum() + 102,
                'BB_Lower': np.random.randn(1000).cumsum() + 98
            })
            
            # Test scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Test train/test split
            X = scaled_data[:, :-1]  # All features except last
            y = scaled_data[:, -1]   # Last feature as target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.log_test_result("Data Preprocessing", True, f"Data scaled and split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            return True
            
        except Exception as e:
            self.log_test_result("Data Preprocessing", False, f"Data preprocessing error: {e}", e)
            return False
    
    def test_model_pipeline(self):
        """Test complete model pipeline without TensorFlow."""
        logger.info("🔗 Testing Complete Model Pipeline...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Create mock financial data
            np.random.seed(42)
            data = pd.DataFrame({
                'Close': np.random.randn(1000).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, 1000),
                'RSI': np.random.uniform(0, 100, 1000),
                'MACD': np.random.randn(1000),
                'BB_Upper': np.random.randn(1000).cumsum() + 102,
                'BB_Lower': np.random.randn(1000).cumsum() + 98
            })
            
            # Prepare features and target
            features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper']
            X = data[features].values
            y = data['BB_Lower'].values  # Predict BB_Lower
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.log_test_result("Model Pipeline", True, f"Pipeline complete: MSE={mse:.4f}, R²={r2:.4f}")
            return True
            
        except Exception as e:
            self.log_test_result("Model Pipeline", False, f"Model pipeline error: {e}", e)
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        logger.info("🚀 Starting Sklearn-Only Test Suite (No TensorFlow)...")
        
        tests = [
            ("Sklearn Import", self.test_sklearn_import),
            ("Sklearn Regression", self.test_sklearn_regression),
            ("Sklearn Classification", self.test_sklearn_classification),
            ("Advanced ML Models", self.test_ml_models_advanced),
            ("Sklearn Fallback", self.test_sklearn_fallback_implementation),
            ("Data Preprocessing", self.test_data_preprocessing),
            ("Model Pipeline", self.test_model_pipeline)
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
        logger.info("📊 SKLEARN-ONLY TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {passed + failed}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        # Save detailed results
        import json
        with open('sklearn_only_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': passed + failed,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0,
                    'total_time_seconds': total_time,
                    'timestamp': datetime.now().isoformat(),
                    'approach': 'sklearn_only_no_tensorflow'
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        logger.info("📄 Detailed results saved to sklearn_only_test_results.json")
        
        return passed, failed

def main():
    """Main test execution."""
    print("🚀 Sklearn-Only Test Suite (No TensorFlow)")
    print("=" * 60)
    print("This test avoids TensorFlow entirely to prevent mutex hanging issues")
    print("=" * 60)
    
    # Run tests
    test_suite = SklearnOnlyTestSuite()
    passed, failed = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Sklearn-only solution is working correctly.")
        print("✅ This confirms we have a working fallback when TensorFlow causes issues.")
        sys.exit(0)
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the results.")
        sys.exit(1)

if __name__ == "__main__":
    main()
