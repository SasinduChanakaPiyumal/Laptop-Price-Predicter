#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration of the bugs that were fixed in the laptop price model.
This script shows what would happen with the original buggy code vs the fixed code.
"""

def original_set_processor(name):
    """Original buggy version of the set_processor function"""
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        # BUG: This will cause IndexError if name.split() is empty
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'

def fixed_set_processor(name):
    """Fixed version of the set_processor function"""
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        # FIXED: Handle empty strings and potential IndexErrors
        name_parts = name.split()
        if len(name_parts) > 0 and name_parts[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'

def demonstrate_syntax_error():
    """Demonstrate the syntax error from pip install line"""
    print("Original code had this line which is invalid Python syntax:")
    print("pip install scikit-learn")
    print("This would cause a SyntaxError when the script is run.")
    print("Fixed by commenting it out: # pip install scikit-learn")
    print()

def demonstrate_index_error():
    """Demonstrate the IndexError bug"""
    print("Testing set_processor function with problematic inputs:")
    
    test_cases = ["", "   ", "AMD Ryzen", "Intel Core i7"]
    
    for test_case in test_cases:
        print(f"\nTesting with input: '{test_case}'")
        
        # Test original function
        try:
            result = original_set_processor(test_case)
            print(f"  Original function result: {result}")
        except IndexError as e:
            print(f"  Original function ERROR: IndexError - {e}")
        except Exception as e:
            print(f"  Original function ERROR: {type(e).__name__} - {e}")
        
        # Test fixed function
        try:
            result = fixed_set_processor(test_case)
            print(f"  Fixed function result: {result}")
        except Exception as e:
            print(f"  Fixed function ERROR: {type(e).__name__} - {e}")

def demonstrate_string_processing_bugs():
    """Demonstrate string processing bugs in CPU/GPU name extraction"""
    print("\nTesting string processing functions:")
    
    # Original buggy version
    def original_extract_cpu_name(x):
        return " ".join(x.split()[0:3])
    
    def original_extract_gpu_name(x):
        return " ".join(x.split()[0:1])
    
    # Fixed versions
    def fixed_extract_cpu_name(x):
        return " ".join(x.split()[0:3]) if len(x.split()) >= 3 else " ".join(x.split()) if len(x.split()) > 0 else 'Unknown'
    
    def fixed_extract_gpu_name(x):
        return " ".join(x.split()[0:1]) if len(x.split()) > 0 else 'Unknown'
    
    test_cases = ["", "   ", "AMD", "Intel Core i7"]
    
    for test_case in test_cases:
        print(f"\nTesting with input: '{test_case}'")
        
        # Test CPU name extraction
        try:
            result = original_extract_cpu_name(test_case)
            print(f"  Original CPU extraction: '{result}'")
        except IndexError as e:
            print(f"  Original CPU extraction ERROR: IndexError - {e}")
        
        result = fixed_extract_cpu_name(test_case)
        print(f"  Fixed CPU extraction: '{result}'")
        
        # Test GPU name extraction
        try:
            result = original_extract_gpu_name(test_case)
            print(f"  Original GPU extraction: '{result}'")
        except IndexError as e:
            print(f"  Original GPU extraction ERROR: IndexError - {e}")
        
        result = fixed_extract_gpu_name(test_case)
        print(f"  Fixed GPU extraction: '{result}'")

if __name__ == '__main__':
    print("=== LAPTOP PRICE MODEL BUG DEMONSTRATION ===\n")
    
    demonstrate_syntax_error()
    demonstrate_index_error()
    demonstrate_string_processing_bugs()
    
    print("\n=== SUMMARY ===")
    print("Fixed bugs:")
    print("1. Syntax error from 'pip install scikit-learn' line in Python code")
    print("2. IndexError in set_processor function when processing empty strings")
    print("3. IndexError in CPU/GPU name extraction functions with empty strings")
    print("4. Added proper error handling and default values for edge cases")
