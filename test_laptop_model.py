#!/usr/bin/env python
"""
Unit test for the laptop price model script.
This test checks that the script can be successfully compiled and executed
without syntax errors.
"""

import unittest
import sys
import ast
import tempfile
import os

class TestLaptopModel(unittest.TestCase):
    
    def test_syntax_validation(self):
        """
        Test that the laptop model script has valid Python syntax.
        This test would fail before the fix due to the invalid 'pip install scikit-learn' line.
        """
        # Read the script file
        script_path = "Laptop Price model(1).py"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Try to parse the script as valid Python syntax
        try:
            ast.parse(script_content)
            # If we reach here, the syntax is valid
            self.assertTrue(True, "Script has valid Python syntax")
        except SyntaxError as e:
            self.fail(f"Script contains syntax error: {e}")
    
    def test_script_execution_without_data(self):
        """
        Test that the script can be executed without immediately crashing on syntax errors.
        This focuses on the syntax validation rather than full execution since we don't
        want to depend on the CSV file or require all sklearn dependencies.
        """
        script_path = "Laptop Price model(1).py"
        
        # Create a minimal version that stops before data loading
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the line that loads the dataset and stop before it
        test_lines = []
        for line in lines:
            if 'pd.read_csv' in line:
                break
            test_lines.append(line)
        
        # Add an early exit to avoid data dependencies
        test_lines.append('\n# Test exit to avoid data dependencies\nexit(0)\n')
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.writelines(test_lines)
            temp_file_path = temp_file.name
        
        try:
            # Try to execute the script
            exit_code = os.system(f'python "{temp_file_path}"')
            # Should exit cleanly (exit code 0)
            self.assertEqual(exit_code, 0, "Script should execute without syntax errors")
        finally:
            # Clean up
            os.unlink(temp_file_path)

if __name__ == '__main__':
    unittest.main()
