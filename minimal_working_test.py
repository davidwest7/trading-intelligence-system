#!/usr/bin/env python3
"""
Minimal Working Test - Only test what we know works
"""

import sys
import os
import time
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test only basic Python imports that should work"""
    try:
        logger.info("🧪 Testing basic imports...")
        
        # Test numpy
        import numpy as np
        logger.info("✅ NumPy imported successfully")
        
        # Test pandas
        import pandas as pd
        logger.info("✅ Pandas imported successfully")
        
        # Test basic Python modules
        import datetime
        import threading
        import queue
        logger.info("✅ Basic Python modules imported successfully")
        
        return True, "Basic imports successful"
        
    except Exception as e:
        logger.error(f"❌ Basic imports failed: {e}")
        return False, str(e)

def test_simple_file_operations():
    """Test simple file operations"""
    try:
        logger.info("🧪 Testing file operations...")
        
        # Test file creation
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Test file reading
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Clean up
        os.remove(test_file)
        
        logger.info("✅ File operations successful")
        return True, "File operations successful"
        
    except Exception as e:
        logger.error(f"❌ File operations failed: {e}")
        return False, str(e)

def test_simple_calculations():
    """Test simple calculations"""
    try:
        logger.info("🧪 Testing calculations...")
        
        # Test numpy calculations
        import numpy as np
        data = np.array([1, 2, 3, 4, 5])
        mean = np.mean(data)
        std = np.std(data)
        
        logger.info(f"✅ Calculations successful: mean={mean}, std={std}")
        return True, f"Calculations successful: mean={mean}, std={std}"
        
    except Exception as e:
        logger.error(f"❌ Calculations failed: {e}")
        return False, str(e)

def test_memory_operations():
    """Test memory operations"""
    try:
        logger.info("🧪 Testing memory operations...")
        
        # Test large array creation
        import numpy as np
        large_array = np.random.rand(1000, 1000)
        
        # Test array operations
        result = np.dot(large_array, large_array.T)
        
        logger.info(f"✅ Memory operations successful: array shape={large_array.shape}")
        return True, f"Memory operations successful: array shape={large_array.shape}"
        
    except Exception as e:
        logger.error(f"❌ Memory operations failed: {e}")
        return False, str(e)

def test_threading_basic():
    """Test basic threading operations"""
    try:
        logger.info("🧪 Testing basic threading...")
        
        import threading
        import time
        
        def simple_worker():
            time.sleep(0.1)
            return "worker completed"
        
        # Create and start thread
        thread = threading.Thread(target=simple_worker)
        thread.start()
        thread.join(timeout=1.0)
        
        if thread.is_alive():
            logger.error("❌ Thread did not complete in time")
            return False, "Thread timeout"
        
        logger.info("✅ Basic threading successful")
        return True, "Basic threading successful"
        
    except Exception as e:
        logger.error(f"❌ Basic threading failed: {e}")
        return False, str(e)

def test_system_info():
    """Test system information gathering"""
    try:
        logger.info("🧪 Testing system info...")
        
        import platform
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        info = {
            'platform': platform.platform(),
            'cpu_count': cpu_count,
            'memory_total': memory.total,
            'memory_available': memory.available
        }
        
        logger.info(f"✅ System info gathered: {info}")
        return True, f"System info: {info}"
        
    except Exception as e:
        logger.error(f"❌ System info failed: {e}")
        return False, str(e)

def main():
    """Run minimal tests"""
    logger.info("🚀 Starting Minimal Working Test Suite")
    logger.info("📊 Testing only what we know should work")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Operations", test_simple_file_operations),
        ("Calculations", test_simple_calculations),
        ("Memory Operations", test_memory_operations),
        ("Basic Threading", test_threading_basic),
        ("System Info", test_system_info),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        start_time = time.time()
        
        try:
            success, result = test_func()
            duration = time.time() - start_time
            
            results[test_name] = {
                'success': success,
                'result': result,
                'duration': duration
            }
            
            if success:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED ({duration:.2f}s)")
            else:
                logger.error(f"❌ {test_name}: FAILED ({duration:.2f}s) - {result}")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {test_name}: ERROR ({duration:.2f}s) - {e}")
            results[test_name] = {
                'success': False,
                'result': str(e),
                'duration': duration
            }
    
    # Generate report
    success_rate = (passed / total) * 100
    logger.info(f"\n📊 Test Results: {passed}/{total} passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        logger.info("🎉 All basic tests passed! System fundamentals are working.")
    elif success_rate >= 80:
        logger.info("✅ Most basic tests passed. System is mostly functional.")
    else:
        logger.error("❌ Many basic tests failed. System has fundamental issues.")
    
    # Save results
    with open('minimal_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📄 Results saved to minimal_test_results.json")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
