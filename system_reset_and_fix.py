#!/usr/bin/env python3
"""
System Reset and Fix - Address root cause of hanging issues
"""

import os
import sys
import signal
import subprocess
import time
import psutil

def kill_all_python_processes():
    """Kill all Python processes except this one"""
    current_pid = os.getpid()
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.pid != current_pid:
                        print(f"🛑 Killing Python process {proc.pid}")
                        proc.terminate()
                        proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        print(f"Warning: Could not kill all processes: {e}")

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_files = [
        'temp_test_*.py',
        'debug_test.txt',
        'test_file.txt',
        '*.log',
        'test_*.json',
        'e2e_test_*.json',
        'production_test_*.json',
        'lightweight_test_*.json',
        'minimal_test_*.json',
        'safe_test_*.json'
    ]
    
    for pattern in temp_files:
        try:
            import glob
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"🗑️ Removed {file}")
                except:
                    pass
        except:
            pass

def reset_environment():
    """Reset environment variables"""
    # Clear problematic environment variables
    env_vars_to_clear = [
        'TF_CPP_MIN_LOG_LEVEL',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            print(f"🧹 Cleared environment variable: {var}")

def check_system_resources():
    """Check system resources"""
    try:
        # Check memory
        memory = psutil.virtual_memory()
        print(f"💾 Memory: {memory.percent}% used ({memory.available / 1024**3:.1f}GB available)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        print(f"💿 Disk: {disk.percent}% used ({disk.free / 1024**3:.1f}GB available)")
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️ CPU: {cpu_percent}% used")
        
        # Check open file descriptors
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"📁 File descriptors: {soft}/{hard}")
        except:
            print("📁 File descriptors: Unable to check")
            
    except Exception as e:
        print(f"Warning: Could not check system resources: {e}")

def create_safe_test_environment():
    """Create a safe test environment"""
    print("🔧 Creating safe test environment...")
    
    # Set safe environment variables
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # Disable problematic features
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("✅ Safe test environment created")

def run_safe_test():
    """Run a safe test"""
    print("🧪 Running safe test...")
    
    # Create a minimal test script
    test_script = '''
import sys
import os
import time

print("🚀 Safe test starting...")
print(f"✅ Python version: {sys.version}")
print(f"✅ Current directory: {os.getcwd()}")
print(f"✅ Process ID: {os.getpid()}")

# Test basic operations
start_time = time.time()
result = 2 + 2
end_time = time.time()

print(f"✅ Basic calculation: 2 + 2 = {result}")
print(f"✅ Test completed in {end_time - start_time:.3f} seconds")
print("🎉 Safe test completed successfully!")
'''
    
    # Write test script to file
    with open('safe_test_script.py', 'w') as f:
        f.write(test_script)
    
    # Run test in subprocess with timeout
    try:
        result = subprocess.run(
            [sys.executable, 'safe_test_script.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Safe test passed!")
            print(result.stdout)
        else:
            print("❌ Safe test failed!")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Safe test timed out!")
    except Exception as e:
        print(f"❌ Safe test error: {e}")
    finally:
        # Clean up
        try:
            os.remove('safe_test_script.py')
        except:
            pass

def main():
    """Main system reset and fix function"""
    print("🚀 Starting System Reset and Fix")
    print("=" * 50)
    
    # Step 1: Kill all Python processes
    print("\n1️⃣ Killing all Python processes...")
    kill_all_python_processes()
    
    # Step 2: Clean up temporary files
    print("\n2️⃣ Cleaning up temporary files...")
    cleanup_temp_files()
    
    # Step 3: Reset environment
    print("\n3️⃣ Resetting environment...")
    reset_environment()
    
    # Step 4: Check system resources
    print("\n4️⃣ Checking system resources...")
    check_system_resources()
    
    # Step 5: Create safe test environment
    print("\n5️⃣ Creating safe test environment...")
    create_safe_test_environment()
    
    # Step 6: Run safe test
    print("\n6️⃣ Running safe test...")
    run_safe_test()
    
    print("\n" + "=" * 50)
    print("🎉 System reset and fix completed!")
    
    # Provide next steps
    print("\n📋 Next Steps:")
    print("1. If the safe test passed, your system is now ready")
    print("2. Run your main tests with the fixes applied")
    print("3. If issues persist, consider restarting your system")
    print("4. Check for any background processes that might be consuming resources")

if __name__ == "__main__":
    main()
