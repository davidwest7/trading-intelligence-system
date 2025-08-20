#!/usr/bin/env python3
"""
Isolated Import Test - Identify the source of mutex lock blocking
"""

import sys
import time

def test_import(module_name, class_name=None):
    """Test importing a specific module"""
    try:
        print(f"🔍 Testing import: {module_name}")
        start_time = time.time()
        
        module = __import__(module_name, fromlist=[class_name] if class_name else None)
        
        if class_name:
            cls = getattr(module, class_name)
            print(f"✅ Successfully imported {module_name}.{class_name}")
        else:
            print(f"✅ Successfully imported {module_name}")
        
        end_time = time.time()
        print(f"⏱️ Import time: {end_time - start_time:.3f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Test imports one by one to identify the problematic one"""
    print("🧪 Isolated Import Test")
    print("=" * 50)
    
    # Test basic imports first
    basic_imports = [
        ("numpy", None),
        ("pandas", None),
        ("sklearn", None),
        ("scipy", None),
    ]
    
    print("\n📦 Testing Basic Imports:")
    for module_name, class_name in basic_imports:
        test_import(module_name, class_name)
        time.sleep(0.1)  # Small delay between imports
    
    # Test our custom modules
    custom_imports = [
        ("common.models", "BaseAgent"),
        ("common.data_adapters.polygon_adapter", "PolygonDataAdapter"),
        ("ml_models.advanced_ml_models", "AdvancedMLModels"),
        ("risk_management.factor_model", "FactorModel"),
        ("execution_algorithms.advanced_execution", "AdvancedExecution"),
        ("hft.low_latency_execution", "LowLatencyExecution"),
        ("hft.market_microstructure", "MarketMicrostructure"),
        ("hft.ultra_fast_models", "UltraFastModels"),
        ("common.evaluation.performance_metrics", "PerformanceMetrics"),
        ("agents.undervalued.agent_real_data", "RealDataUndervaluedAgent"),
    ]
    
    print("\n🔧 Testing Custom Module Imports:")
    for module_name, class_name in custom_imports:
        success = test_import(module_name, class_name)
        if not success:
            print(f"⚠️ Stopping at failed import: {module_name}")
            break
        time.sleep(0.1)  # Small delay between imports
    
    print("\n✅ Import test completed")

if __name__ == "__main__":
    main()
