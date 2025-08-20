#!/usr/bin/env python3
"""
Production-Ready TensorFlow Architecture
Best-in-class implementation to avoid mutex issues in multi-agent trading systems
COMPREHENSIVE VERSION - Includes ALL developed agents
"""

import os
import sys
import multiprocessing as mp
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from queue import Queue, Empty
import signal
import atexit

# ============================================================================
# CRITICAL: NEVER IMPORT TENSORFLOW IN MAIN PROCESS
# ============================================================================

# Set multiprocessing start method to avoid fork-with-TF deadlocks
mp.set_start_method("spawn", force=True)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model serving."""
    model_path: str
    batch_size: int = 32
    timeout_ms: int = 100
    max_queue_size: int = 1000
    gpu_id: Optional[int] = None
    cpu_cores: Optional[List[int]] = None

@dataclass
class ProcessConfig:
    """Configuration for worker processes."""
    num_intraop_threads: int = 4
    num_interop_threads: int = 2
    omp_num_threads: int = 4
    mkl_num_threads: int = 4
    kmp_blocktime: int = 0
    memory_growth: bool = True
    log_device_placement: bool = False

@dataclass
class AgentConfig:
    """Configuration for agents."""
    agent_type: str
    model_name: Optional[str] = None
    priority: int = 1
    enabled: bool = True
    max_concurrent: int = 1

# ============================================================================
# COMPREHENSIVE AGENT REGISTRY
# ============================================================================

class AgentRegistry:
    """Registry for all developed agents."""
    
    @staticmethod
    def get_all_agents() -> Dict[str, Dict[str, Any]]:
        """Get all available agents organized by category."""
        return {
            "technical_analysis": {
                "agent_enhanced": "agents.technical.agent_enhanced.TechnicalAgent",
                "agent_enhanced_multi_timeframe": "agents.technical.agent_enhanced_multi_timeframe.TechnicalAgentMultiTimeframe",
                "agent_optimized": "agents.technical.agent_optimized.TechnicalAgentOptimized",
                "agent_ultra_aggressive": "agents.technical.agent_ultra_aggressive.TechnicalAgentUltraAggressive",
                "agent_world_class": "agents.technical.agent_world_class.TechnicalAgentWorldClass",
                "agent_real_data": "agents.technical.agent_real_data.TechnicalAgentRealData",
                "agent_fixed": "agents.technical.agent_fixed.TechnicalAgentFixed"
            },
            "sentiment_analysis": {
                "agent_optimized": "agents.sentiment.agent_optimized.OptimizedSentimentAgent",
                "agent_enhanced": "agents.sentiment.agent_enhanced.EnhancedSentimentAgent",
                "agent_real_data": "agents.sentiment.agent_real_data.SentimentAgentRealData",
                "agent": "agents.sentiment.agent.SentimentAgent"
            },
            "learning": {
                "agent_optimized": "agents.learning.agent_optimized.LearningAgentOptimized",
                "agent_enhanced_backtesting": "agents.learning.agent_enhanced_backtesting.LearningAgentEnhancedBacktesting",
                "bandit_allocator": "agents.learning.bandit_allocator.BanditAllocator",
                "autonomous_code_generation": "agents.learning.autonomous_code_generation.AutonomousCodeGeneration",
                "agent": "agents.learning.agent.LearningAgent"
            },
            "undervalued": {
                "agent_optimized": "agents.undervalued.agent_optimized.UndervaluedAgentOptimized",
                "agent_enhanced": "agents.undervalued.agent_enhanced.UndervaluedAgentEnhanced",
                "agent_real_data": "agents.undervalued.agent_real_data.UndervaluedAgentRealData",
                "agent": "agents.undervalued.agent.UndervaluedAgent"
            },
            "moneyflows": {
                "agent_optimized": "agents.moneyflows.agent_optimized.MoneyFlowsAgentOptimized",
                "agent": "agents.moneyflows.agent.MoneyFlowsAgent"
            },
            "insider": {
                "agent_optimized": "agents.insider.agent_optimized.InsiderAgentOptimized",
                "agent": "agents.insider.agent.InsiderAgent"
            },
            "macro": {
                "agent_optimized": "agents.macro.agent_optimized.MacroAgentOptimized",
                "agent_complete": "agents.macro.agent_complete.MacroAgentComplete",
                "agent_real_data": "agents.macro.agent_real_data.MacroAgentRealData",
                "agent": "agents.macro.agent.MacroAgent"
            },
            "causal": {
                "agent_optimized": "agents.causal.agent_optimized.CausalAgentOptimized",
                "agent": "agents.causal.agent.CausalAgent"
            },
            "flow": {
                "agent_optimized": "agents.flow.agent_optimized.FlowAgentOptimized",
                "agent_complete": "agents.flow.agent_complete.FlowAgentComplete",
                "agent_real_data": "agents.flow.agent_real_data.FlowAgentRealData",
                "agent": "agents.flow.agent.FlowAgent"
            },
            "hedging": {
                "agent": "agents.hedging.agent.HedgingAgent"
            },
            "top_performers": {
                "agent_optimized": "agents.top_performers.agent_optimized.TopPerformersAgentOptimized",
                "agent_real_data": "agents.top_performers.agent_real_data.TopPerformersAgentRealData",
                "agent": "agents.top_performers.agent.TopPerformersAgent"
            }
        }
    
    @staticmethod
    def get_agent_class(agent_path: str):
        """Get agent class from path."""
        try:
            module_path, class_name = agent_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            print(f"❌ Failed to import agent {agent_path}: {e}")
            return None

# ============================================================================
# MODEL SERVING WORKER PROCESS
# ============================================================================

class TensorFlowWorker:
    """
    Isolated TensorFlow worker process.
    Each worker has its own TF runtime to avoid mutex conflicts.
    """
    
    def __init__(self, model_config: ModelConfig, process_config: ProcessConfig):
        self.model_config = model_config
        self.process_config = process_config
        self.input_queue = mp.Queue(maxsize=model_config.max_queue_size)
        self.output_queue = mp.Queue(maxsize=model_config.max_queue_size)
        self.control_queue = mp.Queue()
        self.process = None
        self.is_running = False
        
    def start(self):
        """Start the worker process."""
        self.process = mp.Process(
            target=self._worker_main,
            args=(self.model_config, self.process_config, 
                  self.input_queue, self.output_queue, self.control_queue)
        )
        self.process.start()
        self.is_running = True
        print(f"🚀 Started TensorFlow worker process (PID: {self.process.pid})")
        
    def stop(self):
        """Stop the worker process gracefully."""
        if self.is_running:
            self.control_queue.put("STOP")
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
            self.is_running = False
            print(f"🛑 Stopped TensorFlow worker process (PID: {self.process.pid})")
    
    def predict(self, data: np.ndarray, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Make prediction with timeout."""
        if not self.is_running:
            raise RuntimeError("Worker process not running")
        
        try:
            # Send data to worker
            self.input_queue.put(data, timeout=timeout)
            
            # Get result with timeout
            result = self.output_queue.get(timeout=timeout)
            return result
            
        except Empty:
            raise TimeoutError(f"Prediction timeout after {timeout}s")
    
    @staticmethod
    def _worker_main(model_config: ModelConfig, process_config: ProcessConfig,
                    input_queue: mp.Queue, output_queue: mp.Queue, control_queue: mp.Queue):
        """
        Main function for TensorFlow worker process.
        This is where TensorFlow is imported and used.
        """
        
        # ========================================================================
        # CRITICAL: Set environment variables BEFORE importing TensorFlow
        # ========================================================================
        
        # Set thread configuration
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(process_config.num_intraop_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(process_config.num_interop_threads)
        os.environ["OMP_NUM_THREADS"] = str(process_config.omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(process_config.mkl_num_threads)
        os.environ["KMP_BLOCKTIME"] = str(process_config.kmp_blocktime)
        
        # Suppress TensorFlow logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_LOGGING_LEVEL"] = "ERROR"
        
        # GPU configuration
        if model_config.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_id)
        
        # CPU affinity (if specified)
        if process_config.cpu_cores:
            try:
                import psutil
                process = psutil.Process()
                process.cpu_affinity(process_config.cpu_cores)
            except ImportError:
                pass  # psutil not available
        
        # ========================================================================
        # NOW import TensorFlow safely
        # ========================================================================
        
        try:
            import tensorflow as tf
            
            # Configure TensorFlow session
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = process_config.memory_growth
            config.log_device_placement = process_config.log_device_placement
            
            # Set session
            tf.compat.v1.keras.backend.set_session(
                tf.compat.v1.Session(config=config)
            )
            
            # Load model
            print(f"📦 Loading model from {model_config.model_path}")
            model = tf.saved_model.load(model_config.model_path)
            
            # Get inference function
            if hasattr(model, 'signatures'):
                infer_func = model.signatures.get("serving_default")
            else:
                infer_func = model
            
            print(f"✅ Model loaded successfully in worker process")
            
        except Exception as e:
            print(f"❌ Failed to load TensorFlow model: {e}")
            return
        
        # ========================================================================
        # Main inference loop
        # ========================================================================
        
        print(f"🔄 Starting inference loop in worker process")
        
        while True:
            try:
                # Check for stop signal
                try:
                    control_signal = control_queue.get_nowait()
                    if control_signal == "STOP":
                        break
                except Empty:
                    pass
                
                # Get input data
                try:
                    data = input_queue.get(timeout=0.1)  # Non-blocking with timeout
                except Empty:
                    continue
                
                # Make prediction
                start_time = time.time()
                try:
                    # Convert to tensor if needed
                    if not isinstance(data, tf.Tensor):
                        data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
                    else:
                        data_tensor = data
                    
                    # Make prediction
                    if hasattr(infer_func, '__call__'):
                        prediction = infer_func(data_tensor)
                    else:
                        prediction = model(data_tensor)
                    
                    # Convert back to numpy
                    if isinstance(prediction, dict):
                        prediction = {k: v.numpy() for k, v in prediction.items()}
                    else:
                        prediction = prediction.numpy()
                    
                    inference_time = time.time() - start_time
                    
                    # Send result
                    output_queue.put(prediction, timeout=1.0)
                    
                except Exception as e:
                    print(f"❌ Inference error: {e}")
                    output_queue.put(None, timeout=1.0)
                
            except Exception as e:
                print(f"❌ Worker loop error: {e}")
                time.sleep(0.1)
        
        print(f"🛑 Worker process shutting down")

# ============================================================================
# MODEL SERVING MANAGER
# ============================================================================

class ModelServingManager:
    """
    Manages multiple TensorFlow worker processes.
    Provides a clean interface for model serving.
    """
    
    def __init__(self):
        self.workers: Dict[str, TensorFlowWorker] = {}
        self.process_config = ProcessConfig()
        
    def register_model(self, model_name: str, model_config: ModelConfig):
        """Register a model for serving."""
        worker = TensorFlowWorker(model_config, self.process_config)
        self.workers[model_name] = worker
        worker.start()
        
    def predict(self, model_name: str, data: np.ndarray, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Make prediction using specified model."""
        if model_name not in self.workers:
            raise ValueError(f"Model '{model_name}' not registered")
        
        return self.workers[model_name].predict(data, timeout)
    
    def stop_all(self):
        """Stop all worker processes."""
        for name, worker in self.workers.items():
            print(f"🛑 Stopping worker for model: {name}")
            worker.stop()
        self.workers.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        status = {}
        for name, worker in self.workers.items():
            status[name] = {
                "running": worker.is_running,
                "pid": worker.process.pid if worker.process else None,
                "alive": worker.process.is_alive() if worker.process else False
            }
        return status

# ============================================================================
# COMPREHENSIVE AGENT COORDINATOR
# ============================================================================

class ComprehensiveAgentCoordinator:
    """
    Comprehensive coordinator for ALL developed agents.
    Manages all agent types without importing TensorFlow.
    """
    
    def __init__(self):
        self.model_manager = ModelServingManager()
        self.agents: Dict[str, Any] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.is_running = False
        self.agent_registry = AgentRegistry()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def register_agent(self, agent_name: str, agent_category: str, agent_type: str, **kwargs):
        """Register an agent from the comprehensive registry."""
        all_agents = self.agent_registry.get_all_agents()
        
        if agent_category not in all_agents:
            raise ValueError(f"Agent category '{agent_category}' not found")
        
        if agent_type not in all_agents[agent_category]:
            raise ValueError(f"Agent type '{agent_type}' not found in category '{agent_category}'")
        
        agent_path = all_agents[agent_category][agent_type]
        agent_class = self.agent_registry.get_agent_class(agent_path)
        
        if agent_class is None:
            raise ValueError(f"Failed to load agent class: {agent_path}")
        
        # Create agent instance (no TF import here)
        agent = agent_class(**kwargs)
        self.agents[agent_name] = agent
        
        # Store configuration
        self.agent_configs[agent_name] = AgentConfig(
            agent_type=agent_type,
            model_name=kwargs.get('model_name'),
            priority=kwargs.get('priority', 1),
            enabled=kwargs.get('enabled', True),
            max_concurrent=kwargs.get('max_concurrent', 1)
        )
        
        print(f"✅ Registered agent: {agent_name} ({agent_category}.{agent_type})")
    
    def register_model(self, model_name: str, model_path: str, **kwargs):
        """Register a TensorFlow model for serving."""
        model_config = ModelConfig(model_path=model_path, **kwargs)
        self.model_manager.register_model(model_name, model_config)
        print(f"✅ Registered model: {model_name} -> {model_path}")
    
    def run_agent_cycle(self, agent_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single agent cycle."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")
        
        agent = self.agents[agent_name]
        config = self.agent_configs[agent_name]
        
        if not config.enabled:
            return {"error": "Agent disabled"}
        
        # Run agent logic (no TensorFlow here)
        try:
            if hasattr(agent, 'process'):
                result = agent.process(data)
            elif hasattr(agent, 'analyze'):
                result = agent.analyze(data)
            elif hasattr(agent, 'find_opportunities'):
                result = agent.find_opportunities(data)
            else:
                result = {"error": "No known processing method found"}
            
            # If agent needs TensorFlow predictions, use model manager
            if config.model_name and hasattr(agent, 'prepare_tf_data'):
                try:
                    tf_data = agent.prepare_tf_data(data)
                    prediction = self.model_manager.predict(config.model_name, tf_data)
                    result['tf_prediction'] = prediction
                except Exception as e:
                    result['tf_prediction_error'] = str(e)
            
            return result
            
        except Exception as e:
            return {"error": f"Agent execution failed: {str(e)}"}
    
    def run_all_agents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents and collect results."""
        results = {}
        
        # Sort agents by priority
        sorted_agents = sorted(
            self.agents.keys(),
            key=lambda x: self.agent_configs[x].priority,
            reverse=True
        )
        
        for agent_name in sorted_agents:
            try:
                result = self.run_agent_cycle(agent_name, data)
                results[agent_name] = result
            except Exception as e:
                print(f"❌ Error running agent {agent_name}: {e}")
                results[agent_name] = {"error": str(e)}
        
        return results
    
    def run_agent_category(self, category: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents in a specific category."""
        results = {}
        
        for agent_name, agent in self.agents.items():
            config = self.agent_configs[agent_name]
            if config.agent_type.startswith(category):
                try:
                    result = self.run_agent_cycle(agent_name, data)
                    results[agent_name] = result
                except Exception as e:
                    print(f"❌ Error running agent {agent_name}: {e}")
                    results[agent_name] = {"error": str(e)}
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {}
        for agent_name, config in self.agent_configs.items():
            status[agent_name] = {
                "type": config.agent_type,
                "enabled": config.enabled,
                "priority": config.priority,
                "model_name": config.model_name,
                "max_concurrent": config.max_concurrent
            }
        return status
    
    def start(self):
        """Start the coordinator."""
        self.is_running = True
        print("🚀 Comprehensive Agent Coordinator started")
        print(f"📊 Registered {len(self.agents)} agents")
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_running:
            print("🧹 Cleaning up resources...")
            self.model_manager.stop_all()
            self.is_running = False
            print("✅ Cleanup complete")

# ============================================================================
# PRODUCTION USAGE EXAMPLE WITH ALL AGENTS
# ============================================================================

def main():
    """Example of production usage with ALL developed agents."""
    
    # Create comprehensive coordinator
    coordinator = ComprehensiveAgentCoordinator()
    
    try:
        # Register TensorFlow models (these will run in separate processes)
        coordinator.register_model(
            "lstm_predictor",
            model_path="./models/lstm_model",
            gpu_id=0,
            batch_size=32
        )
        
        coordinator.register_model(
            "sentiment_transformer",
            model_path="./models/sentiment_model",
            gpu_id=1,
            batch_size=16
        )
        
        coordinator.register_model(
            "causal_model",
            model_path="./models/causal_model",
            gpu_id=2,
            batch_size=16
        )
        
        # Register ALL developed agents
        print("\n📋 Registering ALL developed agents...")
        
        # Technical Analysis Agents
        coordinator.register_agent("technical_enhanced", "technical_analysis", "agent_enhanced", model_name="lstm_predictor")
        coordinator.register_agent("technical_multi_timeframe", "technical_analysis", "agent_enhanced_multi_timeframe")
        coordinator.register_agent("technical_optimized", "technical_analysis", "agent_optimized")
        coordinator.register_agent("technical_ultra_aggressive", "technical_analysis", "agent_ultra_aggressive")
        coordinator.register_agent("technical_world_class", "technical_analysis", "agent_world_class")
        
        # Sentiment Analysis Agents
        coordinator.register_agent("sentiment_optimized", "sentiment_analysis", "agent_optimized", model_name="sentiment_transformer")
        coordinator.register_agent("sentiment_enhanced", "sentiment_analysis", "agent_enhanced")
        coordinator.register_agent("sentiment_real_data", "sentiment_analysis", "agent_real_data")
        
        # Learning Agents
        coordinator.register_agent("learning_optimized", "learning", "agent_optimized")
        coordinator.register_agent("learning_enhanced_backtesting", "learning", "agent_enhanced_backtesting")
        coordinator.register_agent("bandit_allocator", "learning", "bandit_allocator")
        coordinator.register_agent("autonomous_code_generation", "learning", "autonomous_code_generation")
        
        # Undervalued Agents
        coordinator.register_agent("undervalued_optimized", "undervalued", "agent_optimized")
        coordinator.register_agent("undervalued_enhanced", "undervalued", "agent_enhanced")
        coordinator.register_agent("undervalued_real_data", "undervalued", "agent_real_data")
        
        # Money Flows Agents
        coordinator.register_agent("moneyflows_optimized", "moneyflows", "agent_optimized")
        
        # Insider Agents
        coordinator.register_agent("insider_optimized", "insider", "agent_optimized")
        
        # Macro Agents
        coordinator.register_agent("macro_optimized", "macro", "agent_optimized")
        coordinator.register_agent("macro_complete", "macro", "agent_complete")
        coordinator.register_agent("macro_real_data", "macro", "agent_real_data")
        
        # Causal Agents
        coordinator.register_agent("causal_optimized", "causal", "agent_optimized", model_name="causal_model")
        
        # Flow Agents
        coordinator.register_agent("flow_optimized", "flow", "agent_optimized")
        coordinator.register_agent("flow_complete", "flow", "agent_complete")
        coordinator.register_agent("flow_real_data", "flow", "agent_real_data")
        
        # Hedging Agents
        coordinator.register_agent("hedging", "hedging", "agent")
        
        # Top Performers Agents
        coordinator.register_agent("top_performers_optimized", "top_performers", "agent_optimized")
        coordinator.register_agent("top_performers_real_data", "top_performers", "agent_real_data")
        
        # Start coordinator
        coordinator.start()
        
        # Simulate comprehensive market data
        market_data = {
            "close_prices": [100 + i * 0.1 + np.random.randn() for i in range(100)],
            "volumes": [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
            "news_texts": [
                "Company reports strong quarterly growth",
                "Market shows bullish momentum",
                "Analysts predict positive outlook"
            ],
            "economic_indicators": {
                "gdp_growth": 2.5,
                "inflation_rate": 2.1,
                "unemployment_rate": 3.8
            },
            "insider_transactions": [
                {"executive": "CEO", "action": "BUY", "shares": 10000},
                {"executive": "CFO", "action": "SELL", "shares": 5000}
            ]
        }
        
        # Run all agents
        print("\n🔄 Running ALL agents...")
        results = coordinator.run_all_agents(market_data)
        
        # Print results by category
        print("\n📊 Results by Agent Category:")
        categories = ["technical", "sentiment", "learning", "undervalued", "moneyflows", 
                     "insider", "macro", "causal", "flow", "hedging", "top_performers"]
        
        for category in categories:
            category_results = coordinator.run_agent_category(category, market_data)
            if category_results:
                print(f"\n  {category.upper()} AGENTS:")
                for agent_name, result in category_results.items():
                    print(f"    {agent_name}: {result.get('signal', 'N/A')} - {result.get('confidence', 'N/A')}")
        
        # Check model manager status
        print("\n🔍 Model Manager Status:")
        status = coordinator.model_manager.get_status()
        for model_name, model_status in status.items():
            print(f"  {model_name}: {model_status}")
        
        # Check agent status
        print("\n👥 Agent Status:")
        agent_status = coordinator.get_agent_status()
        for agent_name, status in list(agent_status.items())[:10]:  # Show first 10
            print(f"  {agent_name}: {status['type']} (Priority: {status['priority']})")
        
        # Keep running for a bit
        print("\n⏰ Running for 10 seconds...")
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        coordinator.cleanup()

if __name__ == "__main__":
    main()
