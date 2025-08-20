#!/usr/bin/env python3
"""
Agentic TensorFlow Architecture
Multi-Agent System with Autonomous Decision Making
LOCAL DEVELOPMENT VERSION - Runs on laptop for testing
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

# ============================================================================
# CRITICAL: NEVER IMPORT TENSORFLOW IN MAIN PROCESS
# ============================================================================

# Set multiprocessing start method to avoid fork-with-TF deadlocks
mp.set_start_method("spawn", force=True)

# ============================================================================
# AGENTIC ARCHITECTURE CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for autonomous agents."""
    agent_id: str
    agent_type: str
    model_name: Optional[str] = None
    priority: int = 1
    enabled: bool = True
    autonomous: bool = True
    communication_enabled: bool = True
    decision_threshold: float = 0.7
    max_concurrent: int = 1
    local_mode: bool = True  # For laptop testing

@dataclass
class ModelConfig:
    """Configuration for model serving."""
    model_path: str
    batch_size: int = 16  # Smaller for laptop
    timeout_ms: int = 100
    max_queue_size: int = 100
    gpu_id: Optional[int] = None
    cpu_cores: Optional[List[int]] = None
    local_mode: bool = True  # For laptop testing

@dataclass
class ProcessConfig:
    """Configuration for worker processes."""
    num_intraop_threads: int = 2  # Reduced for laptop
    num_interop_threads: int = 1  # Reduced for laptop
    omp_num_threads: int = 2
    mkl_num_threads: int = 2
    kmp_blocktime: int = 0
    memory_growth: bool = True
    log_device_placement: bool = False

# ============================================================================
# AGENTIC COMMUNICATION SYSTEM
# ============================================================================

class AgentMessage:
    """Message for inter-agent communication."""
    
    def __init__(self, sender_id: str, message_type: str, content: Any, priority: int = 1):
        self.message_id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now()
        self.ttl = 300  # 5 minutes TTL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl
        }

class AgenticCommunicationHub:
    """Central communication hub for agent-to-agent messaging."""
    
    def __init__(self):
        self.message_queues: Dict[str, Queue] = {}
        self.broadcast_queue = Queue()
        self.subscribers: Dict[str, List[str]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
        
    def register_agent(self, agent_id: str):
        """Register an agent for communication."""
        self.message_queues[agent_id] = Queue()
        self.subscribers[agent_id] = []
        print(f"📡 Registered agent {agent_id} for communication")
    
    def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe an agent to specific message types."""
        for msg_type in message_types:
            if msg_type not in self.subscribers:
                self.subscribers[msg_type] = []
            self.subscribers[msg_type].append(agent_id)
    
    def send_message(self, message: AgentMessage):
        """Send a message to specific agent or broadcast."""
        if message.sender_id in self.message_queues:
            self.message_queues[message.sender_id].put(message)
        
        # Add to broadcast queue
        self.broadcast_queue.put(message)
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
    
    def broadcast_message(self, message: AgentMessage):
        """Broadcast message to all subscribers of the message type."""
        self.broadcast_queue.put(message)
        
        # Send to specific subscribers
        if message.message_type in self.subscribers:
            for agent_id in self.subscribers[message.message_type]:
                if agent_id in self.message_queues:
                    self.message_queues[agent_id].put(message)
    
    def get_messages(self, agent_id: str, timeout: float = 0.1) -> List[AgentMessage]:
        """Get messages for a specific agent."""
        messages = []
        
        # Get from agent's queue
        try:
            while True:
                message = self.message_queues[agent_id].get_nowait()
                messages.append(message)
        except Empty:
            pass
        
        # Get from broadcast queue
        try:
            while True:
                message = self.broadcast_queue.get_nowait()
                messages.append(message)
        except Empty:
            pass
        
        return messages

# ============================================================================
# AUTONOMOUS AGENT BASE CLASS
# ============================================================================

class AutonomousAgent:
    """Base class for autonomous agents with decision-making capabilities."""
    
    def __init__(self, agent_id: str, agent_type: str, config: AgentConfig):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.communication_hub = None
        self.is_running = False
        self.decision_history = []
        self.confidence_threshold = config.decision_threshold
        self.local_mode = config.local_mode
        
    def set_communication_hub(self, hub: AgenticCommunicationHub):
        """Set the communication hub for this agent."""
        self.communication_hub = hub
        hub.register_agent(self.agent_id)
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and make autonomous decisions."""
        raise NotImplementedError("Subclasses must implement process_data")
    
    def make_autonomous_decision(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decision based on analysis."""
        confidence = analysis_result.get('confidence', 0.0)
        
        if confidence >= self.confidence_threshold:
            decision = {
                "action": analysis_result.get('signal', 'HOLD'),
                "confidence": confidence,
                "reasoning": analysis_result.get('reasoning', 'High confidence threshold met'),
                "timestamp": datetime.now(),
                "agent_id": self.agent_id
            }
            
            # Store decision history
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:
                self.decision_history.pop(0)
            
            # Communicate decision to other agents
            if self.communication_hub and self.config.communication_enabled:
                message = AgentMessage(
                    sender_id=self.agent_id,
                    message_type="DECISION",
                    content=decision,
                    priority=2
                )
                self.communication_hub.broadcast_message(message)
            
            return decision
        else:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}",
                "timestamp": datetime.now(),
                "agent_id": self.agent_id
            }
    
    def receive_message(self, message: AgentMessage):
        """Receive and process messages from other agents."""
        if message.sender_id == self.agent_id:
            return  # Ignore own messages
        
        # Process message based on type
        if message.message_type == "DECISION":
            self.process_peer_decision(message.content)
        elif message.message_type == "MARKET_UPDATE":
            self.process_market_update(message.content)
        elif message.message_type == "RISK_ALERT":
            self.process_risk_alert(message.content)
    
    def process_peer_decision(self, decision: Dict[str, Any]):
        """Process decisions from other agents."""
        # Override in subclasses for specific logic
        pass
    
    def process_market_update(self, update: Dict[str, Any]):
        """Process market updates from other agents."""
        # Override in subclasses for specific logic
        pass
    
    def process_risk_alert(self, alert: Dict[str, Any]):
        """Process risk alerts from other agents."""
        # Override in subclasses for specific logic
        pass
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_running": self.is_running,
            "decision_count": len(self.decision_history),
            "last_decision": self.decision_history[-1] if self.decision_history else None,
            "confidence_threshold": self.confidence_threshold,
            "autonomous": self.config.autonomous,
            "communication_enabled": self.config.communication_enabled
        }

# ============================================================================
# AGENTIC MODEL SERVING (LOCAL-FRIENDLY)
# ============================================================================

class LocalTensorFlowWorker:
    """
    Local-friendly TensorFlow worker for laptop testing.
    Falls back to CPU-only mode when GPU not available.
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
        print(f"🚀 Started local TensorFlow worker (PID: {self.process.pid})")
        
    def stop(self):
        """Stop the worker process gracefully."""
        if self.is_running:
            self.control_queue.put("STOP")
            self.process.join(timeout=5)  # Shorter timeout for local
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
            self.is_running = False
            print(f"🛑 Stopped local TensorFlow worker (PID: {self.process.pid})")
    
    def predict(self, data: np.ndarray, timeout: float = 3.0) -> Optional[np.ndarray]:
        """Make prediction with timeout (shorter for local)."""
        if not self.is_running:
            raise RuntimeError("Worker process not running")
        
        try:
            self.input_queue.put(data, timeout=timeout)
            result = self.output_queue.get(timeout=timeout)
            return result
        except Empty:
            raise TimeoutError(f"Prediction timeout after {timeout}s")
    
    @staticmethod
    def _worker_main(model_config: ModelConfig, process_config: ProcessConfig,
                    input_queue: mp.Queue, output_queue: mp.Queue, control_queue: mp.Queue):
        """Main function for local TensorFlow worker."""
        
        # ========================================================================
        # LOCAL-FRIENDLY ENVIRONMENT SETUP
        # ========================================================================
        
        # Set thread configuration for laptop
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(process_config.num_intraop_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(process_config.num_interop_threads)
        os.environ["OMP_NUM_THREADS"] = str(process_config.omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(process_config.mkl_num_threads)
        os.environ["KMP_BLOCKTIME"] = str(process_config.kmp_blocktime)
        
        # Suppress TensorFlow logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_LOGGING_LEVEL"] = "ERROR"
        
        # Local mode: Disable GPU if not available
        if model_config.local_mode:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if not gpus:
                    print("⚠️ No GPU detected, using CPU-only mode")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                else:
                    print(f"✅ GPU detected: {len(gpus)} device(s)")
                    if model_config.gpu_id is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_id)
            except ImportError:
                print("⚠️ TensorFlow not available, using fallback")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # ========================================================================
        # SAFE TENSORFLOW IMPORT
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
            
            # Load model (with fallback for local testing)
            if os.path.exists(model_config.model_path):
                print(f"📦 Loading model from {model_config.model_path}")
                model = tf.saved_model.load(model_config.model_path)
                
                if hasattr(model, 'signatures'):
                    infer_func = model.signatures.get("serving_default")
                else:
                    infer_func = model
                
                print(f"✅ Model loaded successfully")
            else:
                print(f"⚠️ Model not found at {model_config.model_path}, using mock model")
                # Create a simple mock model for testing
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(10, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                infer_func = model
            
        except Exception as e:
            print(f"❌ Failed to load TensorFlow: {e}")
            print("🔄 Using sklearn fallback for local testing")
            return
        
        # ========================================================================
        # LOCAL INFERENCE LOOP
        # ========================================================================
        
        print(f"🔄 Starting local inference loop")
        
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
                    data = input_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Make prediction
                start_time = time.time()
                try:
                    if not isinstance(data, tf.Tensor):
                        data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
                    else:
                        data_tensor = data
                    
                    if hasattr(infer_func, '__call__'):
                        prediction = infer_func(data_tensor)
                    else:
                        prediction = model(data_tensor)
                    
                    if isinstance(prediction, dict):
                        prediction = {k: v.numpy() for k, v in prediction.items()}
                    else:
                        prediction = prediction.numpy()
                    
                    inference_time = time.time() - start_time
                    output_queue.put(prediction, timeout=1.0)
                    
                except Exception as e:
                    print(f"❌ Local inference error: {e}")
                    output_queue.put(None, timeout=1.0)
                
            except Exception as e:
                print(f"❌ Local worker loop error: {e}")
                time.sleep(0.1)
        
        print(f"🛑 Local worker shutting down")

# ============================================================================
# AGENTIC COORDINATOR
# ============================================================================

class AgenticCoordinator:
    """
    Agentic coordinator for autonomous multi-agent system.
    Runs locally on laptop for testing.
    """
    
    def __init__(self, local_mode: bool = True):
        self.local_mode = local_mode
        self.agents: Dict[str, AutonomousAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.model_workers: Dict[str, LocalTensorFlowWorker] = {}
        self.communication_hub = AgenticCommunicationHub()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)  # Limited for laptop
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def register_agent(self, agent_id: str, agent_class, **kwargs):
        """Register an autonomous agent."""
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=kwargs.get('agent_type', 'unknown'),
            model_name=kwargs.get('model_name'),
            priority=kwargs.get('priority', 1),
            enabled=kwargs.get('enabled', True),
            autonomous=kwargs.get('autonomous', True),
            communication_enabled=kwargs.get('communication_enabled', True),
            decision_threshold=kwargs.get('decision_threshold', 0.7),
            local_mode=self.local_mode
        )
        
        # Create agent instance
        agent = agent_class(agent_id, config.agent_type, config)
        agent.set_communication_hub(self.communication_hub)
        
        self.agents[agent_id] = agent
        self.agent_configs[agent_id] = config
        
        print(f"🤖 Registered autonomous agent: {agent_id} ({config.agent_type})")
    
    def register_model(self, model_name: str, model_path: str, **kwargs):
        """Register a TensorFlow model for local serving."""
        model_config = ModelConfig(
            model_path=model_path,
            local_mode=self.local_mode,
            **kwargs
        )
        
        process_config = ProcessConfig(
            num_intraop_threads=2,  # Conservative for laptop
            num_interop_threads=1,
            omp_num_threads=2,
            mkl_num_threads=2
        )
        
        worker = LocalTensorFlowWorker(model_config, process_config)
        self.model_workers[model_name] = worker
        worker.start()
        
        print(f"📦 Registered local model: {model_name} -> {model_path}")
    
    def run_agent_cycle(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single autonomous agent cycle."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' not registered")
        
        agent = self.agents[agent_id]
        config = self.agent_configs[agent_id]
        
        if not config.enabled:
            return {"error": "Agent disabled"}
        
        try:
            # Process data
            analysis_result = agent.process_data(data)
            
            # Make autonomous decision
            if config.autonomous:
                decision = agent.make_autonomous_decision(analysis_result)
                analysis_result['autonomous_decision'] = decision
            
            # Process incoming messages
            messages = self.communication_hub.get_messages(agent_id)
            for message in messages:
                agent.receive_message(message)
            
            return analysis_result
            
        except Exception as e:
            return {"error": f"Agent execution failed: {str(e)}"}
    
    def run_all_agents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all autonomous agents."""
        results = {}
        
        # Sort by priority
        sorted_agents = sorted(
            self.agents.keys(),
            key=lambda x: self.agent_configs[x].priority,
            reverse=True
        )
        
        for agent_id in sorted_agents:
            try:
                result = self.run_agent_cycle(agent_id, data)
                results[agent_id] = result
            except Exception as e:
                print(f"❌ Error running agent {agent_id}: {e}")
                results[agent_id] = {"error": str(e)}
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_agent_status()
        
        model_statuses = {}
        for model_name, worker in self.model_workers.items():
            model_statuses[model_name] = {
                "running": worker.is_running,
                "pid": worker.process.pid if worker.process else None,
                "alive": worker.process.is_alive() if worker.process else False
            }
        
        return {
            "total_agents": len(self.agents),
            "total_models": len(self.model_workers),
            "system_running": self.is_running,
            "local_mode": self.local_mode,
            "agents": agent_statuses,
            "models": model_statuses,
            "communication_hub": {
                "registered_agents": len(self.communication_hub.message_queues),
                "message_history_size": len(self.communication_hub.message_history)
            }
        }
    
    def start(self):
        """Start the agentic coordinator."""
        self.is_running = True
        print("🚀 Agentic Coordinator started (Local Mode)")
        print(f"🤖 Registered {len(self.agents)} autonomous agents")
        print(f"📦 Registered {len(self.model_workers)} local models")
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_running:
            print("🧹 Cleaning up agentic system...")
            
            # Stop all model workers
            for name, worker in self.model_workers.items():
                print(f"🛑 Stopping model worker: {name}")
                worker.stop()
            
            self.is_running = False
            print("✅ Agentic system cleanup complete")

# ============================================================================
# LOCAL TESTING AGENTS
# ============================================================================

class LocalTechnicalAgent(AutonomousAgent):
    """Local technical analysis agent for testing."""
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and make technical analysis."""
        close_prices = data.get('close_prices', [])
        
        if len(close_prices) < 20:
            return {"error": "Insufficient data", "confidence": 0.0}
        
        # Calculate technical indicators
        sma_20 = np.mean(close_prices[-20:])
        current_price = close_prices[-1]
        
        # Simple momentum analysis
        momentum = (current_price - close_prices[-5]) / close_prices[-5]
        
        # Decision logic
        if current_price > sma_20 and momentum > 0.01:
            signal = "BUY"
            confidence = 0.8
            reasoning = "Price above SMA20 with positive momentum"
        elif current_price < sma_20 and momentum < -0.01:
            signal = "SELL"
            confidence = 0.7
            reasoning = "Price below SMA20 with negative momentum"
        else:
            signal = "HOLD"
            confidence = 0.5
            reasoning = "No clear signal"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "sma_20": sma_20,
            "current_price": current_price,
            "momentum": momentum
        }

class LocalSentimentAgent(AutonomousAgent):
    """Local sentiment analysis agent for testing."""
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment data."""
        news_texts = data.get('news_texts', [])
        
        if not news_texts:
            return {"error": "No news data", "confidence": 0.0}
        
        # Simple sentiment scoring
        positive_words = ['bullish', 'growth', 'profit', 'gain', 'positive', 'strong']
        negative_words = ['bearish', 'loss', 'decline', 'negative', 'risk', 'weak']
        
        total_sentiment = 0
        for text in news_texts:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            total_sentiment += positive_count - negative_count
        
        avg_sentiment = total_sentiment / len(news_texts)
        
        # Decision logic
        if avg_sentiment > 1:
            signal = "BUY"
            confidence = 0.75
            reasoning = f"Positive sentiment score: {avg_sentiment:.2f}"
        elif avg_sentiment < -1:
            signal = "SELL"
            confidence = 0.7
            reasoning = f"Negative sentiment score: {avg_sentiment:.2f}"
        else:
            signal = "HOLD"
            confidence = 0.6
            reasoning = f"Neutral sentiment score: {avg_sentiment:.2f}"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "sentiment_score": avg_sentiment,
            "news_count": len(news_texts)
        }

# ============================================================================
# LOCAL TESTING MAIN
# ============================================================================

def main():
    """Local testing of agentic architecture on laptop."""
    
    print("🤖 Agentic TensorFlow Architecture - Local Testing")
    print("=" * 60)
    
    # Create agentic coordinator
    coordinator = AgenticCoordinator(local_mode=True)
    
    try:
        # Register local models (will use CPU fallback if no GPU)
        coordinator.register_model(
            "lstm_predictor",
            model_path="./models/lstm_model",
            batch_size=16  # Smaller for laptop
        )
        
        coordinator.register_model(
            "sentiment_transformer",
            model_path="./models/sentiment_model",
            batch_size=8  # Smaller for laptop
        )
        
        # Register autonomous agents
        print("\n🤖 Registering autonomous agents...")
        
        coordinator.register_agent(
            "technical_agent",
            LocalTechnicalAgent,
            agent_type="technical_analysis",
            model_name="lstm_predictor",
            priority=1,
            autonomous=True,
            communication_enabled=True,
            decision_threshold=0.7
        )
        
        coordinator.register_agent(
            "sentiment_agent",
            LocalSentimentAgent,
            agent_type="sentiment_analysis",
            model_name="sentiment_transformer",
            priority=2,
            autonomous=True,
            communication_enabled=True,
            decision_threshold=0.6
        )
        
        # Start coordinator
        coordinator.start()
        
        # Simulate market data for local testing
        market_data = {
            "close_prices": [100 + i * 0.1 + np.random.randn() * 0.5 for i in range(100)],
            "volumes": [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
            "news_texts": [
                "Company reports strong quarterly growth and bullish outlook",
                "Market shows positive momentum with increasing volume",
                "Analysts predict continued growth in the sector"
            ]
        }
        
        # Run autonomous agents
        print("\n🔄 Running autonomous agents...")
        results = coordinator.run_all_agents(market_data)
        
        # Print results
        print("\n📊 Autonomous Agent Results:")
        for agent_id, result in results.items():
            if 'error' not in result:
                decision = result.get('autonomous_decision', {})
                print(f"  {agent_id}:")
                print(f"    Signal: {result.get('signal', 'N/A')}")
                print(f"    Confidence: {result.get('confidence', 'N/A'):.2f}")
                print(f"    Reasoning: {result.get('reasoning', 'N/A')}")
                if decision:
                    print(f"    Autonomous Decision: {decision.get('action', 'N/A')}")
            else:
                print(f"  {agent_id}: ERROR - {result['error']}")
        
        # Get system status
        print("\n🔍 System Status:")
        status = coordinator.get_system_status()
        print(f"  Total Agents: {status['total_agents']}")
        print(f"  Total Models: {status['total_models']}")
        print(f"  Local Mode: {status['local_mode']}")
        print(f"  Communication Hub: {status['communication_hub']['registered_agents']} agents registered")
        
        # Keep running for testing
        print("\n⏰ Running for 30 seconds for testing...")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        coordinator.cleanup()

if __name__ == "__main__":
    main()
