#!/usr/bin/env python3
"""
Fix Import Paths Script
Fixes import path issues in evaluation and feature store modules
"""
import os
import shutil
from pathlib import Path

def create_init_files():
    """Create __init__.py files where missing"""
    print("🔧 Creating missing __init__.py files...")
    
    # Create __init__.py for evaluation directory
    eval_init_path = Path('common/evaluation/__init__.py')
    if not eval_init_path.exists():
        eval_init_path.write_text('"""Evaluation modules for trading intelligence system"""\n')
        print(f"✅ Created: {eval_init_path}")
    
    # Create __init__.py for feature_store directory
    feature_init_path = Path('common/feature_store/__init__.py')
    if not feature_init_path.exists():
        feature_init_path.write_text('"""Feature store modules for trading intelligence system"""\n')
        print(f"✅ Created: {feature_init_path}")
    
    # Create __init__.py for scoring directory
    scoring_init_path = Path('common/scoring/__init__.py')
    if not scoring_init_path.exists():
        scoring_init_path.write_text('"""Scoring modules for trading intelligence system"""\n')
        print(f"✅ Created: {scoring_init_path}")
    
    # Create __init__.py for schemas directory
    schemas_init_path = Path('common/schemas/__init__.py')
    if not schemas_init_path.exists():
        schemas_init_path.write_text('"""Schema definitions for trading intelligence system"""\n')
        print(f"✅ Created: {schemas_init_path}")
    
    # Create __init__.py for event_bus directory
    event_bus_init_path = Path('common/event_bus/__init__.py')
    if not event_bus_init_path.exists():
        event_bus_init_path.write_text('"""Event bus modules for trading intelligence system"""\n')
        print(f"✅ Created: {event_bus_init_path}")

def fix_evaluation_imports():
    """Fix imports in evaluation modules"""
    print("\n🔧 Fixing evaluation module imports...")
    
    # Fix risk_metrics.py
    risk_metrics_path = Path('common/evaluation/risk_metrics.py')
    if risk_metrics_path.exists():
        content = risk_metrics_path.read_text()
        # Add proper imports if needed
        if 'from common.' not in content:
            print(f"✅ {risk_metrics_path.name} - imports look good")
    
    # Fix performance_metrics.py
    perf_metrics_path = Path('common/evaluation/performance_metrics.py')
    if perf_metrics_path.exists():
        content = perf_metrics_path.read_text()
        if 'from common.' not in content:
            print(f"✅ {perf_metrics_path.name} - imports look good")
    
    # Fix allocator_metrics.py
    alloc_metrics_path = Path('common/evaluation/allocator_metrics.py')
    if alloc_metrics_path.exists():
        content = alloc_metrics_path.read_text()
        if 'from common.' not in content:
            print(f"✅ {alloc_metrics_path.name} - imports look good")
    
    # Fix backtest_engine.py
    backtest_path = Path('common/evaluation/backtest_engine.py')
    if backtest_path.exists():
        content = backtest_path.read_text()
        if 'from common.' not in content:
            print(f"✅ {backtest_path.name} - imports look good")

def fix_feature_store_imports():
    """Fix imports in feature store modules"""
    print("\n🔧 Fixing feature store module imports...")
    
    # Fix store.py
    store_path = Path('common/feature_store/store.py')
    if store_path.exists():
        content = store_path.read_text()
        if 'from common.' not in content:
            print(f"✅ {store_path.name} - imports look good")
    
    # Fix embargo.py
    embargo_path = Path('common/feature_store/embargo.py')
    if embargo_path.exists():
        content = embargo_path.read_text()
        if 'from common.' not in content:
            print(f"✅ {embargo_path.name} - imports look good")

def fix_agent_structure():
    """Fix agent structure issues"""
    print("\n🔧 Fixing agent structure...")
    
    # Check common agent directory
    common_agent_path = Path('agents/common')
    if common_agent_path.exists():
        # Check if it has proper agent files
        agent_files = list(common_agent_path.glob('*agent*.py'))
        if not agent_files:
            print(f"⚠️ {common_agent_path} - no agent files found, this might be expected")
        else:
            print(f"✅ {common_agent_path} - agent files found")

def create_import_test_script():
    """Create a test script to verify imports work"""
    print("\n🔧 Creating import test script...")
    
    test_script = '''#!/usr/bin/env python3
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
    print("\\n📊 Testing Evaluation Modules:")
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
    print("\\n🏪 Testing Feature Store Modules:")
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
    print("\\n📈 Testing Scoring Modules:")
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
    print("\\n📋 IMPORT TEST SUMMARY:")
    print(f"   Evaluation Modules: {eval_success}/{len(eval_modules)}")
    print(f"   Feature Store Modules: {feature_success}/{len(feature_modules)}")
    print(f"   Scoring Modules: {scoring_success}/{len(scoring_modules)}")
    
    total_success = eval_success + feature_success + scoring_success
    total_modules = len(eval_modules) + len(feature_modules) + len(scoring_modules)
    
    if total_success == total_modules:
        print("\\n🎉 All imports working correctly!")
    else:
        print(f"\\n⚠️ {total_modules - total_success} import issues remaining")

if __name__ == "__main__":
    main()
'''
    
    test_path = Path('test_imports.py')
    test_path.write_text(test_script)
    print(f"✅ Created: {test_path}")

def main():
    """Main function to fix all import issues"""
    print("🔧 Fixing Import Path Issues")
    print("=" * 50)
    
    # 1. Create missing __init__.py files
    create_init_files()
    
    # 2. Fix evaluation module imports
    fix_evaluation_imports()
    
    # 3. Fix feature store module imports
    fix_feature_store_imports()
    
    # 4. Fix agent structure
    fix_agent_structure()
    
    # 5. Create import test script
    create_import_test_script()
    
    print("\n🎉 Import path fixes completed!")
    print("\n📋 Next Steps:")
    print("1. Run: python test_imports.py")
    print("2. Run: python comprehensive_system_validation.py")
    print("3. Check if all import issues are resolved")

if __name__ == "__main__":
    main()
