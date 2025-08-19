#!/usr/bin/env python3
"""
Quick Setup Test - Basic Functionality Check

This script performs a quick check of the basic setup without complex imports
that might cause hanging issues.
"""

import sys
import os
from datetime import datetime

def check_basic_setup():
    """Check basic setup and file structure"""
    print("🔧 **QUICK SETUP TEST**")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    # Add current directory to path
    sys.path.append('.')
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required files exist
    required_files = [
        'agents/learning/advanced_learning_methods.py',
        'agents/learning/enhanced_backtesting.py',
        'agents/learning/autonomous_code_generation.py',
        'common/models.py',
        'common/data_adapters/polygon_adapter.py'
    ]
    
    print(f"\n📁 **FILE STRUCTURE CHECK**")
    print("-" * 30)
    
    files_exist = []
    files_missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            files_exist.append(file_path)
        else:
            print(f"❌ {file_path}")
            files_missing.append(file_path)
    
    # Check package structure
    print(f"\n📦 **PACKAGE STRUCTURE CHECK**")
    print("-" * 30)
    
    init_files = [
        'common/__init__.py',
        'agents/__init__.py',
        'agents/learning/__init__.py',
        'common/data_adapters/__init__.py'
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✅ {init_file}")
        else:
            print(f"❌ {init_file}")
    
    # Check basic imports
    print(f"\n📦 **BASIC IMPORT CHECK**")
    print("-" * 30)
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except ImportError:
        print("❌ pandas not available")
    
    try:
        import numpy as np
        print("✅ numpy imported")
    except ImportError:
        print("❌ numpy not available")
    
    try:
        import sklearn
        print("✅ scikit-learn imported")
    except ImportError:
        print("❌ scikit-learn not available")
    
    # Check common models
    print(f"\n🔧 **COMMON MODELS CHECK**")
    print("-" * 30)
    
    try:
        from common.models import BaseAgent, BaseDataAdapter
        print("✅ common.models imported")
    except Exception as e:
        print(f"❌ common.models import failed: {e}")
    
    # Summary
    print(f"\n📊 **SUMMARY**")
    print("-" * 30)
    print(f"Files present: {len(files_exist)}/{len(required_files)}")
    print(f"Files missing: {len(files_missing)}")
    
    if len(files_missing) == 0:
        print("🎉 All required files present!")
    else:
        print(f"⚠️ Missing files: {files_missing}")
    
    print("=" * 50)
    return len(files_missing) == 0

def main():
    """Main function"""
    success = check_basic_setup()
    if success:
        print("✅ Basic setup check completed successfully!")
    else:
        print("❌ Basic setup check found issues that need attention.")

if __name__ == "__main__":
    main()
