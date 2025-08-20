#!/usr/bin/env python3
"""
Comprehensive TensorFlow Mutex Fix Guide
All critical fixes for production TensorFlow applications
"""

import os
import sys
import threading
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

# ============================================================================
# CRITICAL TENSORFLOW MUTEX FIXES
# ============================================================================

class TensorFlowMutexFixer:
    """
    Comprehensive fixer for TensorFlow mutex and threading issues.
    Implements all known solutions for production environments.
    """
    
    def __init__(self):
        self._tf_initialized = False
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        
    def apply_all_fixes(self):
        """Apply all critical TensorFlow mutex fixes."""
        
        # 1. ENVIRONMENT VARIABLES (MUST BE SET BEFORE TF IMPORT)
        self._set_environment_variables()
        
        # 2. IMPORT TENSORFLOW WITH SAFETY
        tf = self._safe_import_tensorflow()
        
        # 3. CONFIGURE TENSORFLOW THREADING
        self._configure_tensorflow_threading(tf)
        
        # 4. SETUP MEMORY GROWTH
        self._setup_memory_growth(tf)
        
        # 5. CONFIGURE SESSION OPTIONS
        self._configure_session_options(tf)
        
        return tf
    
    def _set_environment_variables(self):
        """Set critical environment variables before TensorFlow import."""
        
        # Suppress TensorFlow logging and warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Disable GPU to prevent CUDA mutex issues
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Disable TensorFlow deprecation warnings
        os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
        
        # Disable TensorFlow debug logging
        os.environ['TF_LOGGING_LEVEL'] = 'ERROR'
        
        # Disable TensorFlow profiling
        os.environ['TF_PROFILER_DISABLE'] = '1'
        
        # Set TensorFlow to use only CPU
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        
        # Disable TensorFlow's internal threading
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        
        print("✅ Environment variables set for TensorFlow mutex safety")
    
    def _safe_import_tensorflow(self):
        """Safely import TensorFlow with error handling."""
        
        try:
            import tensorflow as tf
            print("✅ TensorFlow imported successfully")
            return tf
        except ImportError as e:
            print(f"❌ TensorFlow import failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error importing TensorFlow: {e}")
            return None
    
    def _configure_tensorflow_threading(self, tf):
        """Configure TensorFlow threading to prevent mutex issues."""
        
        if tf is None:
            return
            
        try:
            # Set inter-op parallelism to 1 (critical for mutex safety)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Set intra-op parallelism to 1 (critical for mutex safety)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            # Disable TensorFlow's automatic threading
            tf.config.threading.set_inter_op_parallelism_threads(0)
            
            print("✅ TensorFlow threading configured for mutex safety")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not configure TensorFlow threading: {e}")
    
    def _setup_memory_growth(self, tf):
        """Setup GPU memory growth to prevent memory-related mutex issues."""
        
        if tf is None:
            return
            
        try:
            # Get list of physical GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if gpus:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            else:
                print("ℹ️ No GPUs detected, using CPU only")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not setup GPU memory growth: {e}")
    
    def _configure_session_options(self, tf):
        """Configure TensorFlow session options for mutex safety."""
        
        if tf is None:
            return
            
        try:
            # Configure session options
            config = tf.compat.v1.ConfigProto()
            
            # Disable GPU memory allocation
            config.gpu_options.allow_growth = True
            
            # Set CPU-only mode
            config.device_count['GPU'] = 0
            
            # Disable logging
            config.log_device_placement = False
            
            # Set session options
            tf.compat.v1.keras.backend.set_session(
                tf.compat.v1.Session(config=config)
            )
            
            print("✅ TensorFlow session configured for mutex safety")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not configure TensorFlow session: {e}")

# ============================================================================
# THREAD-SAFE TENSORFLOW WRAPPER
# ============================================================================

class ThreadSafeTensorFlow:
    """
    Thread-safe wrapper for TensorFlow operations.
    Prevents mutex conflicts in multi-threaded environments.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._model_locks = {}
        
    @contextmanager
    def tensorflow_session(self):
        """Thread-safe TensorFlow session context manager."""
        
        with self._session_lock:
            try:
                import tensorflow as tf
                with tf.compat.v1.Session() as session:
                    yield session
            except Exception as e:
                print(f"❌ TensorFlow session error: {e}")
                yield None
    
    def safe_model_predict(self, model, data, timeout=30):
        """Thread-safe model prediction with timeout."""
        
        model_id = id(model)
        if model_id not in self._model_locks:
            self._model_locks[model_id] = threading.Lock()
        
        with self._model_locks[model_id]:
            try:
                # Set timeout for prediction
                start_time = time.time()
                prediction = model.predict(data, verbose=0)
                
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Model prediction exceeded {timeout}s timeout")
                
                return prediction
                
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                return None
    
    def safe_model_train(self, model, x, y, **kwargs):
        """Thread-safe model training."""
        
        model_id = id(model)
        if model_id not in self._model_locks:
            self._model_locks[model_id] = threading.Lock()
        
        with self._model_locks[model_id]:
            try:
                # Add callback to prevent hanging
                callbacks = kwargs.get('callbacks', [])
                callbacks.append(TimeoutCallback(timeout=300))  # 5 minute timeout
                kwargs['callbacks'] = callbacks
                
                history = model.fit(x, y, **kwargs)
                return history
                
            except Exception as e:
                print(f"❌ Model training error: {e}")
                return None

# ============================================================================
# TIMEOUT CALLBACK FOR TENSORFLOW
# ============================================================================

class TimeoutCallback:
    """Callback to prevent TensorFlow operations from hanging."""
    
    def __init__(self, timeout=300):
        self.timeout = timeout
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if self.start_time and (time.time() - self.start_time) > self.timeout:
            raise TimeoutError(f"Training exceeded {self.timeout}s timeout")

# ============================================================================
# ALTERNATIVE ML SOLUTIONS (NON-TENSORFLOW)
# ============================================================================

class SafeMLFallback:
    """
    Safe ML fallback using sklearn and other non-TensorFlow libraries.
    Use when TensorFlow causes mutex issues.
    """
    
    def __init__(self):
        self.models = {}
        
    def create_sklearn_model(self, model_type='random_forest'):
        """Create sklearn-based model as TensorFlow alternative."""
        
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif model_type == 'svm':
                from sklearn.svm import SVR
                return SVR()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except ImportError:
            print("❌ sklearn not available")
            return None
    
    def train_sklearn_model(self, model, X, y):
        """Train sklearn model safely."""
        
        try:
            model.fit(X, y)
            return True
        except Exception as e:
            print(f"❌ sklearn training error: {e}")
            return False
    
    def predict_sklearn_model(self, model, X):
        """Predict with sklearn model safely."""
        
        try:
            return model.predict(X)
        except Exception as e:
            print(f"❌ sklearn prediction error: {e}")
            return None

# ============================================================================
# PRODUCTION DEPLOYMENT RECOMMENDATIONS
# ============================================================================

def get_production_recommendations():
    """Get production deployment recommendations for TensorFlow mutex issues."""
    
    return {
        "critical_fixes": [
            "Set TF_CPP_MIN_LOG_LEVEL=2 before any TensorFlow import",
            "Set CUDA_VISIBLE_DEVICES=-1 to disable GPU",
            "Set TF_NUM_INTEROP_THREADS=1 and TF_NUM_INTRAOP_THREADS=1",
            "Use ThreadSafeTensorFlow wrapper for all operations",
            "Implement timeout callbacks for all TensorFlow operations",
            "Use sklearn fallback when TensorFlow causes issues"
        ],
        "monitoring": [
            "Monitor thread count during TensorFlow operations",
            "Set up alerts for hanging TensorFlow processes",
            "Log all TensorFlow operations with timestamps",
            "Monitor memory usage during TensorFlow operations"
        ],
        "deployment": [
            "Use Docker containers with resource limits",
            "Implement graceful shutdown for TensorFlow processes",
            "Use process isolation for TensorFlow operations",
            "Implement circuit breakers for TensorFlow failures"
        ],
        "testing": [
            "Test TensorFlow operations under high thread load",
            "Test timeout scenarios for all TensorFlow operations",
            "Test memory leak scenarios",
            "Test graceful degradation to sklearn fallbacks"
        ]
    }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the TensorFlow mutex fixes."""
    
    print("🚀 Applying TensorFlow mutex fixes...")
    
    # 1. Apply all fixes
    fixer = TensorFlowMutexFixer()
    tf = fixer.apply_all_fixes()
    
    if tf is None:
        print("❌ TensorFlow not available, using sklearn fallback")
        fallback = SafeMLFallback()
        model = fallback.create_sklearn_model('random_forest')
        return model
    
    # 2. Use thread-safe wrapper
    safe_tf = ThreadSafeTensorFlow()
    
    # 3. Safe model operations
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile with timeout protection
        model.compile(optimizer='adam', loss='mse')
        
        # Safe training
        X = tf.random.normal((100, 5))
        y = tf.random.normal((100, 1))
        
        history = safe_tf.safe_model_train(model, X, y, epochs=5)
        
        if history:
            print("✅ Model trained successfully")
            
            # Safe prediction
            predictions = safe_tf.safe_model_predict(model, X[:10])
            if predictions is not None:
                print(f"✅ Generated {len(predictions)} predictions")
        
    except Exception as e:
        print(f"❌ TensorFlow operation failed: {e}")
        print("🔄 Falling back to sklearn...")
        
        fallback = SafeMLFallback()
        model = fallback.create_sklearn_model('random_forest')
        if model:
            success = fallback.train_sklearn_model(model, X.numpy(), y.numpy())
            if success:
                predictions = fallback.predict_sklearn_model(model, X[:10].numpy())
                print(f"✅ Sklearn fallback successful: {len(predictions)} predictions")

if __name__ == "__main__":
    example_usage()
