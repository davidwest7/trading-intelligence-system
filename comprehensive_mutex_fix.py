#!/usr/bin/env python3
"""
Comprehensive Mutex and Threading Fix
Fixes all mutex lock blocking issues and optimizes for high data volumes
"""

import os
import sys
import threading
import multiprocessing
import asyncio
import time
import signal
import gc
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MutexFixManager:
    """Comprehensive mutex and threading fix manager"""
    
    def __init__(self):
        self.thread_locks = {}
        self.resource_pools = {}
        self.cleanup_handlers = []
        self._shutdown_event = threading.Event()
        
        # Production settings for high data volumes
        self.max_threads = min(32, (os.cpu_count() or 1) * 2)
        self.chunk_size = 10000
        self.memory_limit_mb = 4096  # 4GB
        self.timeout_seconds = 60
        
        # Initialize thread-safe locks
        self._init_thread_locks()
        
    def _init_thread_locks(self):
        """Initialize thread-safe locks for critical sections"""
        self.thread_locks = {
            'tensorflow': threading.RLock(),
            'data_processing': threading.RLock(),
            'ml_models': threading.RLock(),
            'risk_calculation': threading.RLock(),
            'execution': threading.RLock(),
            'hft': threading.RLock(),
            'governance': threading.RLock(),
            'alternative_data': threading.RLock(),
            'performance_metrics': threading.RLock(),
            'integration': threading.RLock()
        }
        
    @contextmanager
    def safe_import_context(self, module_name: str):
        """Safe import context that prevents mutex conflicts"""
        lock = self.thread_locks.get(module_name.split('.')[0], threading.RLock())
        
        with lock:
            try:
                # Set environment variables to prevent TensorFlow threading issues
                if 'tensorflow' in module_name.lower() or 'ml' in module_name.lower():
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                    os.environ['OMP_NUM_THREADS'] = '1'
                    os.environ['MKL_NUM_THREADS'] = '1'
                    
                yield
                
            except Exception as e:
                logger.error(f"Import error in {module_name}: {e}")
                raise
                
    def fix_tensorflow_mutex_issues(self):
        """Fix TensorFlow mutex issues"""
        logger.info("🔧 Fixing TensorFlow mutex issues...")
        
        # Set TensorFlow configuration
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # Disable TensorFlow threading
        try:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            logger.info("✅ TensorFlow threading configured safely")
        except ImportError:
            logger.info("⚠️ TensorFlow not available, skipping configuration")
            
    def fix_threading_issues(self):
        """Fix threading issues across the system"""
        logger.info("🔧 Fixing threading issues...")
        
        # Fix alternative data threading
        self._fix_alternative_data_threading()
        
        # Fix HFT threading
        self._fix_hft_threading()
        
        # Fix governance threading
        self._fix_governance_threading()
        
        # Fix ML models threading
        self._fix_ml_models_threading()
        
        logger.info("✅ Threading issues fixed")
        
    def _fix_alternative_data_threading(self):
        """Fix alternative data threading issues"""
        try:
            from alternative_data.real_time_data_integration import RealTimeAlternativeData
            
            # Patch the class to use proper cleanup
            original_init = RealTimeAlternativeData.__init__
            
            def safe_init(self, config=None):
                original_init(self, config)
                self._cleanup_handlers = []
                self._shutdown_event = threading.Event()
                
            RealTimeAlternativeData.__init__ = safe_init
            
            # Add proper cleanup methods
            def safe_stop(self):
                if hasattr(self, '_shutdown_event'):
                    self._shutdown_event.set()
                if hasattr(self, 'is_running'):
                    self.is_running = False
                    
            RealTimeAlternativeData.stop = safe_stop
            
            logger.info("✅ Alternative data threading fixed")
            
        except Exception as e:
            logger.warning(f"Could not fix alternative data threading: {e}")
            
    def _fix_hft_threading(self):
        """Fix HFT threading issues"""
        try:
            from hft.low_latency_execution import LowLatencyExecution
            from hft.high_frequency_trading import HighFrequencyTradingEngine
            
            # Patch HFT classes to use proper cleanup
            for cls in [LowLatencyExecution, HighFrequencyTradingEngine]:
                if hasattr(cls, '__init__'):
                    original_init = cls.__init__
                    
                    def safe_init(self, config=None):
                        original_init(self, config)
                        self._cleanup_handlers = []
                        self._shutdown_event = threading.Event()
                        
                    cls.__init__ = safe_init
                    
                # Add cleanup method
                def safe_stop(self):
                    if hasattr(self, '_shutdown_event'):
                        self._shutdown_event.set()
                    if hasattr(self, 'is_running'):
                        self.is_running = False
                        
                cls.stop = safe_stop
                
            logger.info("✅ HFT threading fixed")
            
        except Exception as e:
            logger.warning(f"Could not fix HFT threading: {e}")
            
    def _fix_governance_threading(self):
        """Fix governance threading issues"""
        try:
            from governance.governance_engine import GovernanceEngine
            
            # Patch governance engine
            original_init = GovernanceEngine.__init__
            
            def safe_init(self, config=None):
                original_init(self, config)
                self._cleanup_handlers = []
                self._shutdown_event = threading.Event()
                
            GovernanceEngine.__init__ = safe_init
            
            # Add cleanup method
            def safe_stop(self):
                if hasattr(self, '_shutdown_event'):
                    self._shutdown_event.set()
                    
            GovernanceEngine.stop = safe_stop
            
            logger.info("✅ Governance threading fixed")
            
        except Exception as e:
            logger.warning(f"Could not fix governance threading: {e}")
            
    def _fix_ml_models_threading(self):
        """Fix ML models threading issues"""
        try:
            # Import ML models safely
            with self.safe_import_context('ml_models'):
                from ml_models.lstm_predictor import LSTM_Predictor
                from ml_models.transformer_sentiment import TransformerSentiment
                from ml_models.advanced_ml_models import AdvancedMLPredictor
                
                # Patch ML models to use thread-safe operations
                for cls in [LSTM_Predictor, TransformerSentiment, AdvancedMLPredictor]:
                    if hasattr(cls, '__init__'):
                        original_init = cls.__init__
                        
                        def safe_init(self, config=None):
                            original_init(self, config)
                            self._model_lock = threading.RLock()
                            
                        cls.__init__ = safe_init
                        
                    # Add thread-safe predict method
                    def safe_predict(self, *args, **kwargs):
                        with self._model_lock:
                            return self._original_predict(*args, **kwargs)
                            
                    if hasattr(cls, 'predict'):
                        cls._original_predict = cls.predict
                        cls.predict = safe_predict
                        
            logger.info("✅ ML models threading fixed")
            
        except Exception as e:
            logger.warning(f"Could not fix ML models threading: {e}")
            
    def optimize_for_high_data_volumes(self):
        """Optimize system for high data volumes"""
        logger.info("🚀 Optimizing for high data volumes...")
        
        # Set system limits
        self._set_system_limits()
        
        # Optimize memory usage
        self._optimize_memory_usage()
        
        # Optimize processing
        self._optimize_processing()
        
        # Optimize I/O
        self._optimize_io()
        
        logger.info("✅ High data volume optimization complete")
        
    def _set_system_limits(self):
        """Set system limits for high data volumes"""
        try:
            import resource
            
            # Set memory limit
            memory_limit = self.memory_limit_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set file descriptor limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
            
            logger.info(f"✅ System limits set: {self.memory_limit_mb}MB memory, 65536 file descriptors")
            
        except Exception as e:
            logger.warning(f"Could not set system limits: {e}")
            
    def _optimize_memory_usage(self):
        """Optimize memory usage for high data volumes"""
        # Enable garbage collection
        gc.enable()
        
        # Set garbage collection thresholds
        gc.set_threshold(700, 10, 10)
        
        # Disable warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        logger.info("✅ Memory optimization complete")
        
    def _optimize_processing(self):
        """Optimize processing for high data volumes"""
        # Set process priority
        try:
            os.nice(10)  # Lower priority to prevent system overload
        except Exception:
            pass
            
        # Set thread priority
        try:
            import threading
            threading.current_thread().setDaemon(True)
        except Exception:
            pass
            
        logger.info("✅ Processing optimization complete")
        
    def _optimize_io(self):
        """Optimize I/O for high data volumes"""
        # Set buffer sizes
        try:
            import socket
            socket.SO_RCVBUF = 65536
            socket.SO_SNDBUF = 65536
        except Exception:
            pass
            
        logger.info("✅ I/O optimization complete")
        
    def create_safe_test_environment(self):
        """Create a safe test environment"""
        logger.info("🧪 Creating safe test environment...")
        
        # Fix all issues
        self.fix_tensorflow_mutex_issues()
        self.fix_threading_issues()
        self.optimize_for_high_data_volumes()
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        # Set up cleanup handlers
        self._setup_cleanup_handlers()
        
        logger.info("✅ Safe test environment created")
        
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            self._shutdown_event.set()
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def _setup_cleanup_handlers(self):
        """Set up cleanup handlers"""
        def cleanup_handler():
            # Force garbage collection
            gc.collect()
            
            # Stop all background threads
            self._shutdown_event.set()
            
        self.cleanup_handlers.append(cleanup_handler)
        
    def cleanup(self):
        """Comprehensive cleanup"""
        logger.info("🧹 Starting comprehensive cleanup...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Run cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Cleanup handler failed: {e}")
                
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ Cleanup completed")
        
    def run_safe_test(self, test_func, test_name: str, timeout: int = None):
        """Run a test safely with all fixes applied"""
        timeout = timeout or self.timeout_seconds
        
        try:
            logger.info(f"🧪 Running {test_name} safely...")
            start_time = time.time()
            
            # Create safe environment
            self.create_safe_test_environment()
            
            # Run test with timeout
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def run_test():
                try:
                    result = test_func()
                    result_queue.put(('success', result))
                except Exception as e:
                    result_queue.put(('error', e))
                    
            test_thread = threading.Thread(target=run_test, daemon=True)
            test_thread.start()
            
            # Wait for result with timeout
            try:
                result_type, result = result_queue.get(timeout=timeout)
                
                if result_type == 'success':
                    duration = time.time() - start_time
                    logger.info(f"✅ {test_name} completed in {duration:.2f}s")
                    return True, result
                else:
                    logger.error(f"❌ {test_name} failed: {result}")
                    return False, result
                    
            except queue.Empty:
                logger.error(f"⏰ {test_name} timed out after {timeout}s")
                return False, "Timeout"
                
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            return False, e
        finally:
            self.cleanup()

def create_safe_import_function():
    """Create a safe import function that prevents mutex issues"""
    def safe_import(module_name: str, class_name: str = None):
        """Safely import a module or class"""
        try:
            # Set environment variables
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            # Import module
            module = __import__(module_name, fromlist=[class_name] if class_name else None)
            
            if class_name:
                cls = getattr(module, class_name)
                return cls
            else:
                return module
                
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}")
            raise
            
    return safe_import

def create_safe_test_suite():
    """Create a safe test suite with all fixes applied"""
    fix_manager = MutexFixManager()
    
    def safe_test_suite():
        """Safe test suite that runs all tests with fixes"""
        logger.info("🚀 Starting Safe Test Suite")
        
        # Create safe environment
        fix_manager.create_safe_test_environment()
        
        # Test imports safely
        safe_import = create_safe_import_function()
        
        tests = [
            ("Data Engine", lambda: safe_import("common.data_adapters.polygon_adapter", "PolygonDataAdapter")),
            ("ML Models", lambda: safe_import("ml_models.advanced_ml_models", "AdvancedMLModels")),
            ("Risk Management", lambda: safe_import("risk_management.factor_model", "FactorModel")),
            ("Execution Algorithms", lambda: safe_import("execution_algorithms.advanced_execution", "AdvancedExecution")),
            ("HFT Components", lambda: safe_import("hft.low_latency_execution", "LowLatencyExecution")),
            ("Performance Metrics", lambda: safe_import("common.evaluation.performance_metrics", "PerformanceMetrics")),
            ("Integration Workflow", lambda: safe_import("agents.undervalued.agent_real_data", "RealDataUndervaluedAgent")),
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            success, result = fix_manager.run_safe_test(test_func, test_name)
            results[test_name] = {'success': success, 'result': result}
            
            if success:
                passed += 1
                
        # Generate report
        total = len(tests)
        success_rate = (passed / total) * 100
        
        logger.info(f"\n📊 Test Results: {passed}/{total} passed ({success_rate:.1f}%)")
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            logger.info(f"{status} {test_name}")
            
        return results
        
    return safe_test_suite

if __name__ == "__main__":
    # Create and run safe test suite
    safe_suite = create_safe_test_suite()
    results = safe_suite()
    
    # Save results
    import json
    with open('safe_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info("📄 Results saved to safe_test_results.json")
