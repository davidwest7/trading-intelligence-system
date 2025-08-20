#!/usr/bin/env python3
"""
Debug Hang - Identify exactly what's causing the hang
"""

import sys
import os

print("🚀 Starting debug...")

# Test 1: Basic print
print("✅ Basic print works")

# Test 2: Import sys
print("✅ Sys import works")

# Test 3: Import os
print("✅ OS import works")

# Test 4: Check current directory
print(f"✅ Current directory: {os.getcwd()}")

# Test 5: List files
try:
    files = os.listdir('.')
    print(f"✅ Directory listing works: {len(files)} files")
except Exception as e:
    print(f"❌ Directory listing failed: {e}")

# Test 6: Check Python version
print(f"✅ Python version: {sys.version}")

# Test 7: Check environment
print(f"✅ Python executable: {sys.executable}")

# Test 8: Check if we can write files
try:
    with open('debug_test.txt', 'w') as f:
        f.write('test')
    os.remove('debug_test.txt')
    print("✅ File operations work")
except Exception as e:
    print(f"❌ File operations failed: {e}")

print("🎉 Debug complete - no hangs detected!")
