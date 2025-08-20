#!/usr/bin/env python3
"""
Test Agentic Architecture - Local Laptop Testing
Tests the agentic system without requiring actual TensorFlow models
"""

import os
import sys
import time
import numpy as np
from agentic_tensorflow_architecture import (
    AgenticCoordinator, 
    LocalTechnicalAgent, 
    LocalSentimentAgent,
    AgentConfig
)

def test_agentic_architecture():
    """Test the agentic architecture locally."""
    
    print("🧪 Testing Agentic Architecture - Local Mode")
    print("=" * 50)
    
    # Create coordinator
    coordinator = AgenticCoordinator(local_mode=True)
    
    try:
        # Register agents (no models needed for basic testing)
        print("\n🤖 Registering autonomous agents...")
        
        coordinator.register_agent(
            "technical_agent",
            LocalTechnicalAgent,
            agent_type="technical_analysis",
            priority=1,
            autonomous=True,
            communication_enabled=True,
            decision_threshold=0.7
        )
        
        coordinator.register_agent(
            "sentiment_agent", 
            LocalSentimentAgent,
            agent_type="sentiment_analysis",
            priority=2,
            autonomous=True,
            communication_enabled=True,
            decision_threshold=0.6
        )
        
        # Start coordinator
        coordinator.start()
        
        # Test data
        test_data = {
            "close_prices": [100 + i * 0.2 + np.random.randn() * 0.3 for i in range(50)],
            "volumes": [1000000 + np.random.randint(-50000, 50000) for _ in range(50)],
            "news_texts": [
                "Company reports strong quarterly growth and bullish outlook",
                "Market shows positive momentum with increasing volume", 
                "Analysts predict continued growth in the sector",
                "New product launch exceeds expectations",
                "Partnership announcement boosts investor confidence"
            ]
        }
        
        print(f"\n📊 Test Data:")
        print(f"  Close prices: {len(test_data['close_prices'])} points")
        print(f"  Volumes: {len(test_data['volumes'])} points")
        print(f"  News texts: {len(test_data['news_texts'])} articles")
        
        # Run agents
        print("\n🔄 Running autonomous agents...")
        start_time = time.time()
        results = coordinator.run_all_agents(test_data)
        end_time = time.time()
        
        print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")
        
        # Display results
        print("\n📈 Agent Results:")
        for agent_id, result in results.items():
            print(f"\n  🤖 {agent_id}:")
            if 'error' not in result:
                print(f"    Signal: {result.get('signal', 'N/A')}")
                print(f"    Confidence: {result.get('confidence', 'N/A'):.2f}")
                print(f"    Reasoning: {result.get('reasoning', 'N/A')}")
                
                # Show autonomous decision
                decision = result.get('autonomous_decision', {})
                if decision:
                    print(f"    🎯 Autonomous Decision: {decision.get('action', 'N/A')}")
                    print(f"    📅 Timestamp: {decision.get('timestamp', 'N/A')}")
                
                # Show specific metrics
                if 'sma_20' in result:
                    print(f"    📊 SMA20: {result['sma_20']:.2f}")
                if 'momentum' in result:
                    print(f"    📈 Momentum: {result['momentum']:.4f}")
                if 'sentiment_score' in result:
                    print(f"    💭 Sentiment Score: {result['sentiment_score']:.2f}")
            else:
                print(f"    ❌ Error: {result['error']}")
        
        # Test communication
        print("\n📡 Testing inter-agent communication...")
        status = coordinator.get_system_status()
        print(f"  Communication Hub: {status['communication_hub']['registered_agents']} agents")
        print(f"  Message History: {status['communication_hub']['message_history_size']} messages")
        
        # Test individual agent cycles
        print("\n🔄 Testing individual agent cycles...")
        for agent_id in coordinator.agents.keys():
            print(f"  Testing {agent_id}...")
            result = coordinator.run_agent_cycle(agent_id, test_data)
            if 'error' not in result:
                print(f"    ✅ {agent_id} completed successfully")
            else:
                print(f"    ❌ {agent_id} failed: {result['error']}")
        
        # System status
        print("\n🔍 Final System Status:")
        final_status = coordinator.get_system_status()
        print(f"  Total Agents: {final_status['total_agents']}")
        print(f"  System Running: {final_status['system_running']}")
        print(f"  Local Mode: {final_status['local_mode']}")
        
        # Agent details
        for agent_id, agent_status in final_status['agents'].items():
            print(f"  {agent_id}:")
            print(f"    Type: {agent_status['agent_type']}")
            print(f"    Autonomous: {agent_status['autonomous']}")
            print(f"    Decisions: {agent_status['decision_count']}")
            print(f"    Communication: {agent_status['communication_enabled']}")
        
        print("\n✅ Agentic Architecture Test Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        coordinator.cleanup()

def test_agent_communication():
    """Test agent-to-agent communication."""
    
    print("\n📡 Testing Agent Communication...")
    
    coordinator = AgenticCoordinator(local_mode=True)
    
    try:
        # Register agents
        coordinator.register_agent(
            "agent_a",
            LocalTechnicalAgent,
            agent_type="technical_analysis",
            autonomous=True,
            communication_enabled=True
        )
        
        coordinator.register_agent(
            "agent_b",
            LocalSentimentAgent,
            agent_type="sentiment_analysis", 
            autonomous=True,
            communication_enabled=True
        )
        
        coordinator.start()
        
        # Subscribe agents to message types
        coordinator.communication_hub.subscribe("agent_a", ["DECISION", "MARKET_UPDATE"])
        coordinator.communication_hub.subscribe("agent_b", ["DECISION", "RISK_ALERT"])
        
        # Run agents to generate messages
        test_data = {
            "close_prices": [100 + i * 0.1 for i in range(30)],
            "news_texts": ["Positive market sentiment", "Strong growth reported"]
        }
        
        results = coordinator.run_all_agents(test_data)
        
        # Check message history
        message_count = len(coordinator.communication_hub.message_history)
        print(f"  Messages exchanged: {message_count}")
        
        if message_count > 0:
            print("  ✅ Agent communication working")
        else:
            print("  ⚠️ No messages exchanged")
        
    finally:
        coordinator.cleanup()

if __name__ == "__main__":
    print("🧪 Agentic Architecture Local Testing Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    test_agentic_architecture()
    
    # Test 2: Communication
    test_agent_communication()
    
    print("\n🎉 All tests completed!")
