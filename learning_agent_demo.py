#!/usr/bin/env python3
"""
Small Learning Agent Demo
Shows the Learning Agent's capabilities in the terminal
"""

import asyncio
import sys
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from agents.learning.agent import LearningAgent


async def demo_learning_agent():
    """Demo the Learning Agent"""
    print("🧠 LEARNING AGENT DEMO")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize agent
    print("🔧 Initializing Learning Agent...")
    agent = LearningAgent()
    print("✅ Agent initialized")
    print()
    
    # Run analysis
    print("📊 Running learning system analysis...")
    result = await agent.process()
    print("✅ Analysis completed")
    print()
    
    # Extract analysis
    analysis = result['learning_analysis']
    
    # Display system info
    print("🏢 SYSTEM INFORMATION")
    print("-" * 30)
    print(f"System ID: {analysis['system_id']}")
    print(f"Timestamp: {analysis['timestamp']}")
    print()
    
    # Display active models
    print("🤖 ACTIVE MODELS")
    print("-" * 30)
    active_models = analysis['active_models']
    print(f"Total models: {len(active_models)}")
    print()
    
    for i, model in enumerate(active_models[:3], 1):  # Show first 3
        print(f"Model {i}: {model['model_id']}")
        print(f"  Type: {model['model_type']}")
        print(f"  Sharpe Ratio: {model['sharpe_ratio']:.3f}")
        print(f"  Accuracy: {model['accuracy']:.3f}")
        print(f"  F1 Score: {model['f1_score']:.3f}")
        print()
    
    # Display best model
    print("🏆 BEST PERFORMING MODEL")
    print("-" * 30)
    best_model_id = analysis['best_performing_model']
    best_model = next((m for m in active_models if m['model_id'] == best_model_id), None)
    if best_model:
        print(f"Model: {best_model['model_id']}")
        print(f"Sharpe Ratio: {best_model['sharpe_ratio']:.3f}")
        print(f"Accuracy: {best_model['accuracy']:.3f}")
        print(f"Max Drawdown: {best_model['max_drawdown']:.3f}")
        print()
    
    # Display ensemble performance
    print("🎯 ENSEMBLE PERFORMANCE")
    print("-" * 30)
    ensemble = analysis['ensemble_performance']
    print(f"Sharpe Ratio: {ensemble['sharpe_ratio']:.3f}")
    print(f"Accuracy: {ensemble['accuracy']:.3f}")
    print(f"F1 Score: {ensemble['f1_score']:.3f}")
    print(f"Hit Rate: {ensemble['hit_rate']:.3f}")
    print()
    
    # Display recent adaptations
    print("🔄 RECENT ADAPTATIONS")
    print("-" * 30)
    adaptations = analysis['recent_adaptations']
    print(f"Total adaptations: {len(adaptations)}")
    print()
    
    for i, adaptation in enumerate(adaptations[:2], 1):  # Show first 2
        print(f"Adaptation {i}: {adaptation['adaptation_id']}")
        print(f"  Method: {adaptation['learning_method']}")
        print(f"  Improvement: {adaptation['improvement']:.3f}")
        print(f"  Features Added: {len(adaptation['features_added'])}")
        print(f"  Features Removed: {len(adaptation['features_removed'])}")
        print()
    
    # Display feature importance
    print("📈 FEATURE IMPORTANCE")
    print("-" * 30)
    feature_importance = analysis['feature_importance']
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.3f}")
    print()
    
    # Display regime detection
    print("🌍 REGIME DETECTION")
    print("-" * 30)
    regime = analysis['regime_detection']
    print(f"Current Regime: {regime['current_regime']}")
    print(f"Confidence: {regime['regime_confidence']}")
    print(f"Expected Duration: {regime['expected_regime_duration']}")
    print()
    
    # Display recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 30)
    recommendations = analysis['recommended_adaptations']
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print()
    
    # Display learning trajectory
    print("📊 LEARNING TRAJECTORY")
    print("-" * 30)
    trajectory = analysis['learning_trajectory']
    print(f"Trajectory points: {len(trajectory)}")
    print(f"Recent performance: {trajectory[-1]:.3f}")
    print(f"Average performance: {sum(trajectory) / len(trajectory):.3f}")
    print()
    
    # Display future projections
    print("🔮 FUTURE PROJECTIONS")
    print("-" * 30)
    print(f"Expected Improvement: {analysis['expected_performance_improvement']:.3f}")
    print(f"Next Learning Cycle: {analysis['next_learning_cycle']}")
    print()
    
    print("=" * 50)
    print("🎉 LEARNING AGENT DEMO COMPLETED!")
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(demo_learning_agent())
