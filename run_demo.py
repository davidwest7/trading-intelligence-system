#!/usr/bin/env python3
"""
Demo Launcher - Trading Intelligence System

Simple launcher for the full demo run.
"""

import asyncio
import sys
import os

def main():
    """Launch the full demo"""
    print("🚀 Trading Intelligence System - Full Demo")
    print("=" * 60)
    print("This demo will showcase:")
    print("✅ All optimized agents (Sentiment, Flow, Causal, Insider)")
    print("✅ Real-time processing capabilities")
    print("✅ Advanced analytics and ML integration")
    print("✅ Opportunity generation system")
    print("✅ Performance metrics and optimization")
    print("✅ System integration and dashboard capabilities")
    print("=" * 60)
    
    try:
        # Import and run the demo
        from full_demo_run import main as run_demo
        asyncio.run(run_demo())
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
