#!/usr/bin/env python3
"""
Simple Dashboard Launcher
"""

import subprocess
import sys
import os

def launch_dashboard():
    """Launch the complete Streamlit dashboard"""
    try:
        print("🚀 Launching Complete Trading Intelligence Dashboard...")
        print("📊 Starting Streamlit server...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_complete_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()
