"""
Enhanced Logging System for Learning Agent
Provides detailed logging with multiple levels and comprehensive tracking
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
import json
import time

class EnhancedLogger:
    """Enhanced logging system with detailed tracking"""
    
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.DEBUG):
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # Create handlers
        self._setup_handlers()
        
        # Performance tracking
        self.performance_metrics = {}
        self.start_times = {}
        
        self.logger.info(f"Enhanced logger initialized for: {name}")
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with detailed format
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log")
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance file handler
        perf_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.name}_performance_{datetime.now().strftime('%Y%m%d')}.log")
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(perf_handler)
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        self.logger.debug(f"⏱️ Timer started for: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.performance_metrics[operation] = duration
            self.logger.info(f"⏱️ {operation} completed in {duration:.4f} seconds")
            del self.start_times[operation]
            return duration
        return 0.0
    
    def log_data_validation(self, data: Any, data_name: str):
        """Log detailed data validation"""
        self.logger.debug(f"🔍 Validating {data_name}:")
        
        if hasattr(data, 'shape'):
            self.logger.debug(f"   📊 Shape: {data.shape}")
        if hasattr(data, 'columns'):
            self.logger.debug(f"   📋 Columns: {list(data.columns)}")
        if hasattr(data, 'dtypes'):
            self.logger.debug(f"   🔧 Data types: {dict(data.dtypes)}")
        if hasattr(data, 'isnull'):
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                self.logger.warning(f"   ⚠️ Null values: {dict(null_counts)}")
            else:
                self.logger.debug(f"   ✅ No null values found")
        
        if isinstance(data, (list, tuple)):
            self.logger.debug(f"   📊 Length: {len(data)}")
            if len(data) > 0:
                self.logger.debug(f"   📋 First item type: {type(data[0])}")
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, Any]):
        """Log detailed model performance metrics"""
        self.logger.info(f"📊 Model Performance - {model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {metric}: {value:.4f}")
            else:
                self.logger.info(f"   {metric}: {value}")
    
    def log_learning_progress(self, agent_name: str, iteration: int, total: int, 
                            current_metric: float, best_metric: float):
        """Log learning progress with detailed metrics"""
        progress = (iteration / total) * 100
        self.logger.info(
            f"🧠 {agent_name} Progress: {iteration}/{total} ({progress:.1f}%) | "
            f"Current: {current_metric:.4f} | Best: {best_metric:.4f}"
        )
    
    def log_error_with_context(self, error: Exception, context: str = "", 
                             additional_info: Dict[str, Any] = None):
        """Log error with detailed context"""
        self.logger.error(f"❌ Error in {context}: {str(error)}")
        self.logger.error(f"📋 Error type: {type(error).__name__}")
        self.logger.error(f"📍 Traceback: {traceback.format_exc()}")
        
        if additional_info:
            self.logger.error(f"🔍 Additional context: {json.dumps(additional_info, indent=2)}")
    
    def log_memory_usage(self, operation: str, memory_before: float, memory_after: float):
        """Log memory usage for operations"""
        memory_change = memory_after - memory_before
        self.logger.debug(
            f"💾 Memory usage for {operation}: "
            f"Before: {memory_before:.2f}MB | "
            f"After: {memory_after:.2f}MB | "
            f"Change: {memory_change:+.2f}MB"
        )
    
    def log_configuration(self, config: Dict[str, Any]):
        """Log configuration details"""
        self.logger.info("⚙️ Configuration:")
        for key, value in config.items():
            self.logger.info(f"   {key}: {value}")
    
    def log_recommendations(self, recommendations: list):
        """Log recommendations with details"""
        self.logger.info(f"💡 Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            self.logger.info(f"   {i}. {rec}")
    
    def log_performance_summary(self):
        """Log performance summary"""
        if self.performance_metrics:
            self.logger.info("📊 Performance Summary:")
            for operation, duration in self.performance_metrics.items():
                self.logger.info(f"   {operation}: {duration:.4f}s")
        else:
            self.logger.info("📊 No performance metrics recorded")

# Create global logger instance
def get_enhanced_logger(name: str) -> EnhancedLogger:
    """Get enhanced logger instance"""
    return EnhancedLogger(name)

# Convenience functions
def log_info(logger: EnhancedLogger, message: str):
    """Log info message"""
    logger.logger.info(f"ℹ️ {message}")

def log_debug(logger: EnhancedLogger, message: str):
    """Log debug message"""
    logger.logger.debug(f"🔍 {message}")

def log_warning(logger: EnhancedLogger, message: str):
    """Log warning message"""
    logger.logger.warning(f"⚠️ {message}")

def log_error(logger: EnhancedLogger, message: str, error: Exception = None):
    """Log error message"""
    if error:
        logger.log_error_with_context(error, message)
    else:
        logger.logger.error(f"❌ {message}")

def log_success(logger: EnhancedLogger, message: str):
    """Log success message"""
    logger.logger.info(f"✅ {message}")
