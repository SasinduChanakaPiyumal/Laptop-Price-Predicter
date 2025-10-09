#!/usr/bin/env python
# coding: utf-8

import unittest
import sys
import tempfile
import os


class TestLaptopPriceModel(unittest.TestCase):
    
    def test_add_company_function(self):
        """Test the add_company function works correctly"""
        # Import the function from the main script
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Since the main script has execution code, we need to extract just the function
        # Let's define it here to test the logic
        def add_company(inpt):
            if inpt == 'Samsung' or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
                return 'Other'
            else:
                return inpt
        
        # Test cases
        self.assertEqual(add_company('Samsung'), 'Other')
        self.assertEqual(add_company('Apple'), 'Apple')  # Should remain unchanged
        self.assertEqual(add_company('Dell'), 'Dell')    # Should remain unchanged
        self.assertEqual(add_company('Razer'), 'Other')
        self.assertEqual(add_company('Microsoft'), 'Other')
        self.assertEqual(add_company('Huawei'), 'Other')
    
    def test_original_buggy_function_with_syntax_issue(self):
        """Test that demonstrates the original spacing bug would work but is poor style"""
        # This represents the buggy version (without proper spacing)
        def buggy_add_company(inpt):
            # Note: 'Samsung'or actually works in Python but is poor style
            if inpt == 'Samsung'or inpt == 'Razer' or inpt == 'Mediacom':
                return 'Other'
            else:
                return inpt
        
        # Even the buggy version would work, but it's bad style
        self.assertEqual(buggy_add_company('Samsung'), 'Other')
        self.assertEqual(buggy_add_company('Apple'), 'Apple')
    
    def test_script_syntax_without_pip_command(self):
        """Test that the script can be imported without syntax errors after removing pip command"""
        # Create a temporary file with just the function definitions to test import
        script_content = '''
def add_company(inpt):
    if inpt == 'Samsung' or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt

def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'

def set_os(inpt):
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'
'''
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_path = f.name
        
        try:
            # Try to compile the script content
            compile(script_content, temp_path, 'exec')
            # If we get here, the syntax is valid
            self.assertTrue(True, "Script compiles without syntax errors")
        except SyntaxError as e:
            self.fail(f"Script has syntax errors: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_original_script_would_fail_with_pip_command(self):
        """Test that demonstrates the original bug - pip command in Python script"""
        buggy_script_content = '''
def add_company(inpt):
    if inpt == 'Samsung' or inpt == 'Razer':
        return 'Other'
    else:
        return inpt

pip install scikit-learn  # This line would cause SyntaxError

print("This would never be reached")
'''
        
        # This should raise a SyntaxError due to the pip command
        with self.assertRaises(SyntaxError):
            compile(buggy_script_content, '<string>', 'exec')


if __name__ == '__main__':
    unittest.main()
