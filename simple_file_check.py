#!/usr/bin/env python3
"""
Simple File Check
Only checks if files exist - no imports or complex operations
"""
import os

def check_files():
    """Check if required files exist"""
    print("🔍 SIMPLE FILE CHECK")
    print("=" * 40)
    
    files_to_check = [
        'env_real_keys.env',
        'common/data_adapters/fred_adapter.py',
        'agents/macro/agent_real_data.py',
        'test_fred_integration.py',
        'fix_mutex_issues.py',
        'verify_data_mapping.py'
    ]
    
    all_exist = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    print("\n" + "=" * 40)
    
    if all_exist:
        print("🎉 ALL FILES EXIST!")
        print("✅ FRED API Integration Files Created")
        print("✅ Ready for Manual Testing")
    else:
        print("❌ SOME FILES MISSING")
    
    print("=" * 40)
    
    return all_exist

if __name__ == "__main__":
    check_files()
