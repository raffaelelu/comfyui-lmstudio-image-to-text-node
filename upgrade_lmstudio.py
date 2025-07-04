#!/usr/bin/env python3
"""
LM Studio SDK Upgrade Helper

This script helps users upgrade their LM Studio SDK when compatibility issues arise.
Run this script when you encounter errors like:
- "Object missing required field `bosToken`"
- "jinjaPromptTemplate" errors
- Other version compatibility issues

Usage:
    python upgrade_lmstudio.py
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("LM Studio SDK Upgrade Helper")
    print("=" * 40)
    print()
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print(f"✓ Virtual environment detected: {sys.prefix}")
    else:
        print("⚠️  No virtual environment detected. Consider using a virtual environment.")
    
    print()
    
    # Check current LM Studio SDK version
    print("Checking current LM Studio SDK version...")
    success, output = run_command("pip show lmstudio")
    
    if success:
        for line in output.split('\n'):
            if line.startswith('Version:'):
                current_version = line.split(':')[1].strip()
                print(f"Current version: {current_version}")
                break
    else:
        print("LM Studio SDK not found or not installed properly")
    
    print()
    
    # Upgrade LM Studio SDK
    print("Upgrading LM Studio SDK...")
    upgrade_command = "pip install lmstudio --upgrade"
    
    print(f"Running: {upgrade_command}")
    success, output = run_command(upgrade_command)
    
    if success:
        print("✓ LM Studio SDK upgraded successfully!")
        print()
        print("Output:")
        print(output)
    else:
        print("✗ Failed to upgrade LM Studio SDK")
        print("Error:")
        print(output)
        return 1
    
    print()
    
    # Check new version
    print("Checking new LM Studio SDK version...")
    success, output = run_command("pip show lmstudio")
    
    if success:
        for line in output.split('\n'):
            if line.startswith('Version:'):
                new_version = line.split(':')[1].strip()
                print(f"New version: {new_version}")
                break
    
    print()
    print("Upgrade complete! Your LM Studio nodes should now work with the latest version.")
    print()
    print("If you still encounter issues:")
    print("1. Restart ComfyUI completely")
    print("2. Make sure LM Studio is updated to the latest version")
    print("3. Check that your models are loaded in LM Studio")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
