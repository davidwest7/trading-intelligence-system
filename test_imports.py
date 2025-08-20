#!/usr/bin/env python3
"""
Import Test Script
Tests all module imports to ensure they work correctly
"""
import sys
import importlib
from pathlib import Path

def test_import(module_name: str) -> bool:
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name}: {e}")
        return False

def main():
    """Test all module imports"""
    print("🧪 Testing Module Imports")
    print("=" * 40)
    
    # Test evaluation modules
    print("\n📊 Testing Evaluation Modules:")
    eval_modules = [
        'common.evaluation.risk_metrics',
        'common.evaluation.performance_metrics', 
        'common.evaluation.allocator_metrics',
        'common.evaluation.backtest_engine'
    ]
    
    eval_success = 0
    for module in eval_modules:
        if test_import(module):
            print(f"✅ {module}")
            eval_success += 1
        else:
            print(f"❌ {module}")
    
    # Test feature store modules
    print("\n🏪 Testing Feature Store Modules:")
    feature_modules = [
        'common.feature_store.store',
        'common.feature_store.embargo'
    ]
    
    feature_success = 0
    for module in feature_modules:
        if test_import(module):
            print(f"✅ {module}")
            feature_success += 1
        else:
            print(f"❌ {module}")
    
    # Test scoring modules
    print("\n📈 Testing Scoring Modules:")
    scoring_modules = [
        'common.scoring.unified_score'
    ]
    
    scoring_success = 0
    for module in scoring_modules:
        if test_import(module):
            print(f"✅ {module}")
            scoring_success += 1
        else:
            print(f"❌ {module}")
    
    # Summary
    print("\n📋 IMPORT TEST SUMMARY:")
    print(f"   Evaluation Modules: {eval_success}/{len(eval_modules)}")
    print(f"   Feature Store Modules: {feature_success}/{len(feature_modules)}")
    print(f"   Scoring Modules: {scoring_success}/{len(scoring_modules)}")
    
    total_success = eval_success + feature_success + scoring_success
    total_modules = len(eval_modules) + len(feature_modules) + len(scoring_modules)
    
    if total_success == total_modules:
        print("\n🎉 All imports working correctly!")
    else:
        print(f"\n⚠️ {total_modules - total_success} import issues remaining")

if __name__ == "__main__":
    main()
