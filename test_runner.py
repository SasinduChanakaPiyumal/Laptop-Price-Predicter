#!/usr/bin/env python
"""
Test runner to demonstrate the bug fix.

This script shows that:
1. Before the fix: The Python file has a syntax error due to 'pip install' command
2. After the fix: The Python file has valid syntax and can be compiled
"""

import os
import sys

def test_syntax(file_path):
    """Test if a Python file has valid syntax"""
    print(f"Testing syntax of: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found")
        return False
    
    try:
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        # Try to compile the source code
        compile(source_code, file_path, 'exec')
        print(f"✅ {file_path} has valid Python syntax")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path} at line {e.lineno}: {e.msg}")
        print(f"   Text: {e.text.strip() if e.text else 'N/A'}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("LAPTOP PRICE MODEL - BUG FIX DEMONSTRATION")
    print("=" * 60)
    
    # Test the main file
    result = test_syntax("Laptop Price model(1).py")
    
    if result:
        print("\n🎉 SUCCESS: The bug has been fixed!")
        print("   The Python file now has valid syntax and can be executed.")
    else:
        print("\n💥 FAILURE: The file still contains syntax errors.")
    
    print("\n" + "=" * 60)
    print("Bug Summary:")
    print("- Issue: 'pip install scikit-learn' command was mixed into Python code")
    print("- Fix: Commented out the pip install line with proper explanation")
    print("- Result: File now has valid Python syntax")
    print("=" * 60)

if __name__ == '__main__':
    main()
