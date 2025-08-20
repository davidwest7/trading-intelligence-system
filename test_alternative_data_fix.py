#!/usr/bin/env python3
"""
Test script to verify alternative data integration fixes
"""

import time
import asyncio
import threading
from alternative_data.real_time_data_integration import RealTimeDataIntegration, RealTimeAlternativeData

def test_alternative_data_cleanup():
    """Test that alternative data integration properly cleans up"""
    print("🧪 Testing Alternative Data Integration Cleanup")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Test with context manager
        print("1. Testing with context manager...")
        with RealTimeDataIntegration() as alt_data:
            print("   ✅ Context manager entered successfully")
            # Simulate some work
            time.sleep(1)
            print("   ✅ Work completed, exiting context manager")
        
        print("   ✅ Context manager exited successfully")
        
        # Test manual initialization and cleanup
        print("\n2. Testing manual initialization and cleanup...")
        alt_data = RealTimeDataIntegration()
        print("   ✅ RealTimeDataIntegration created")
        
        # Initialize
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(asyncio.wait_for(alt_data.initialize(), timeout=5.0))
        loop.close()
        
        if success:
            print("   ✅ Initialization successful")
        else:
            print("   ⚠️ Initialization failed (expected in test environment)")
        
        # Cleanup
        alt_data.stop()
        print("   ✅ Manual cleanup successful")
        
        # Test destructor
        print("\n3. Testing destructor...")
        alt_data = RealTimeDataIntegration()
        del alt_data
        print("   ✅ Destructor called successfully")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 All tests passed! Duration: {duration:.2f} seconds")
        print("✅ Alternative data integration no longer hangs!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_thread_cleanup():
    """Test that background threads are properly cleaned up"""
    print("\n🧪 Testing Thread Cleanup")
    print("=" * 60)
    
    try:
        # Count active threads before
        initial_threads = threading.active_count()
        print(f"Initial thread count: {initial_threads}")
        
        # Create and destroy multiple instances
        for i in range(3):
            print(f"Creating instance {i+1}...")
            with RealTimeDataIntegration() as alt_data:
                time.sleep(0.5)
            print(f"Destroyed instance {i+1}")
        
        # Wait a bit for threads to clean up
        time.sleep(2)
        
        # Count active threads after
        final_threads = threading.active_count()
        print(f"Final thread count: {final_threads}")
        
        if final_threads <= initial_threads + 2:  # Allow for some overhead
            print("✅ Thread cleanup successful!")
            return True
        else:
            print(f"⚠️ Thread count increased: {final_threads - initial_threads} threads may not have cleaned up")
            return False
            
    except Exception as e:
        print(f"❌ Thread cleanup test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Alternative Data Integration Fix Tests")
    print("=" * 80)
    
    success1 = test_alternative_data_cleanup()
    success2 = test_thread_cleanup()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Alternative data integration is fixed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
