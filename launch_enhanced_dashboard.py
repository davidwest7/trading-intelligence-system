#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher
"""

import subprocess
import sys
import os

def launch_dashboard():
    """Launch the enhanced Streamlit dashboard"""
    try:
        print("🚀 Launching Enhanced Trading Intelligence Dashboard...")
        print("📊 Starting Streamlit server...")
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_enhanced_dashboard.py",
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
