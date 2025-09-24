#!/usr/bin/env python
"""
Unit test for the laptop price model to verify it can be imported and executed
without syntax errors.

This test specifically checks that the Python file can be compiled and imported,
which will fail if there are syntax errors like the 'pip install' command
mixed into the Python code.
"""

import unittest
import sys
import os
import importlib.util


class TestLaptopModel(unittest.TestCase):
    
    def test_python_file_syntax_valid(self):
        """
        Test that the laptop price model Python file has valid syntax
        and can be imported without syntax errors.
        
        This test will fail if there are syntax errors in the file,
        such as shell commands mixed into Python code.
        """
        # Path to the laptop price model file
        file_path = "Laptop Price model(1).py"
        
        # Check if file exists
        self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        
        try:
            # Try to compile the file to check for syntax errors
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # This will raise SyntaxError if there are syntax issues
            compile(source_code, file_path, 'exec')
            
            # If we get here, the syntax is valid
            print("✓ Python file has valid syntax and can be compiled")
            
        except SyntaxError as e:
            self.fail(f"Syntax error in {file_path} at line {e.lineno}: {e.msg}")
        except Exception as e:
            self.fail(f"Unexpected error when checking syntax: {e}")
    
    def test_file_can_be_imported(self):
        """
        Test that the laptop price model file can be imported as a module.
        This is a more comprehensive test that will catch runtime issues too.
        """
        file_path = "Laptop Price model(1).py"
        
        try:
            # Load the module spec
            spec = importlib.util.spec_from_file_location("laptop_model", file_path)
            self.assertIsNotNone(spec, "Could not create module spec")
            
            # Create the module
            module = importlib.util.module_from_spec(spec)
            
            # This is where the syntax error will be caught
            spec.loader.exec_module(module)
            
            print("✓ Python file can be imported successfully")
            
        except SyntaxError as e:
            self.fail(f"Syntax error when importing {file_path} at line {e.lineno}: {e.msg}")
        except FileNotFoundError:
            # Expected if CSV file is missing, but that's OK for syntax testing
            print("✓ Syntax is valid (FileNotFoundError is expected due to missing CSV)")
        except Exception as e:
            # Check if it's just a missing CSV file issue
            if "laptop_price.csv" in str(e):
                print("✓ Syntax is valid (CSV file missing but that's OK for syntax testing)")
            else:
                self.fail(f"Unexpected error when importing: {e}")


if __name__ == '__main__':
    unittest.main()
