#!/usr/bin/env python3
"""
Save Current State Script
Saves the current state of Learning Agent files before restarting terminal
"""

import os
import shutil
from datetime import datetime
import json

def save_current_state():
    """Save current state of Learning Agent files"""
    print("💾 Saving current state...")
    
    # Create backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_learning_agent_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "agents/learning/advanced_learning_methods_fixed.py",
        "agents/learning/enhanced_logging.py",
        "agents/learning/enhanced_backtesting.py",
        "agents/learning/autonomous_code_generation.py",
        "agents/learning/agent_enhanced_backtesting.py",
        "comprehensive_learning_agent_test.py",
        "test_learning_agent_small.py",
        "simple_learning_agent_demo.py",
        "LEARNING_AGENT_CODE_REVIEW.md",
        "ENHANCED_LOGGING_AND_TESTING_SUMMARY.md"
    ]
    
    # Save files
    saved_files = []
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            try:
                # Create subdirectories if needed
                backup_path = os.path.join(backup_dir, file_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, backup_path)
                saved_files.append(file_path)
                print(f"✅ Saved: {file_path}")
            except Exception as e:
                print(f"❌ Failed to save {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Create state summary
    state_summary = {
        "timestamp": timestamp,
        "backup_directory": backup_dir,
        "saved_files": saved_files,
        "total_files": len(saved_files),
        "system_info": {
            "python_version": os.sys.version,
            "current_directory": os.getcwd()
        }
    }
    
    # Save state summary
    summary_file = os.path.join(backup_dir, "state_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(state_summary, f, indent=2)
    
    print(f"\n💾 State saved successfully!")
    print(f"📁 Backup directory: {backup_dir}")
    print(f"📊 Files saved: {len(saved_files)}")
    print(f"📋 Summary: {summary_file}")
    
    return backup_dir

def create_restart_script(backup_dir):
    """Create a script to restore state after restart"""
    restart_script = f"""
#!/usr/bin/env python3
"""
Restart and Test Learning Agent Script
Restores state and runs comprehensive tests
"""

import os
import sys
import subprocess
from datetime import datetime

def restore_and_test():
    print("🔄 Restarting Learning Agent testing...")
    print(f"📁 Restoring from: {backup_dir}")
    
    # Check if backup exists
    if not os.path.exists("{backup_dir}"):
        print(f"❌ Backup directory not found: {backup_dir}")
        return False
    
    # Restore files if needed
    print("📋 Checking file integrity...")
    
    # Run comprehensive test
    print("🧪 Running comprehensive Learning Agent test...")
    try:
        result = subprocess.run([
            sys.executable, "comprehensive_learning_agent_test.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("📊 Test output:")
            print(result.stdout)
        else:
            print("❌ Test failed!")
            print("📋 Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    restore_and_test()
"""
    
    # Write restart script
    with open("restart_learning_agent_test.py", 'w') as f:
        f.write(restart_script)
    
    print(f"📝 Created restart script: restart_learning_agent_test.py")

if __name__ == "__main__":
    backup_dir = save_current_state()
    create_restart_script(backup_dir)
    
    print("\n🔄 Ready to restart terminal!")
    print("📋 After restart, run: python restart_learning_agent_test.py")
