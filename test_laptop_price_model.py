#!/usr/bin/env python3
"""
Unit tests for laptop price model script.
Tests that demonstrate the SyntaxError bug and verify it's fixed after patching.
"""
import unittest
import subprocess
import sys
import os
import tempfile
import shutil


class TestLaptopPriceModel(unittest.TestCase):
    """Test cases for the laptop price model script."""
    
    def setUp(self):
        """Set up test environment."""
        self.script_path = "Laptop Price model(1).py"
    
    def test_script_can_be_compiled_after_patch(self):
        """
        Test that after patching, the script can be compiled without syntax errors.
        
        Before the patch: The script contained 'pip install scikit-learn' on line 256
        which is not valid Python syntax and would cause a SyntaxError.
        
        After the patch: This line is commented out, so the script compiles successfully.
        """
        # Try to compile the script to check for syntax errors
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # After the patch, this should not raise a SyntaxError
        try:
            compiled_code = compile(content, self.script_path, 'exec')
            # If we get here, compilation succeeded
            self.assertTrue(True, "Script compiled successfully without SyntaxError")
        except SyntaxError as e:
            self.fail(f"Script has SyntaxError: {e}")
    
    def test_pip_install_line_is_commented(self):
        """
        Test that the problematic pip install line has been properly commented out.
        """
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # Check that the pip install line is now commented
        lines = content.split('\n')
        pip_install_lines = [line for line in lines if 'pip install scikit-learn' in line]
        
        # Should find the line, and it should be commented
        self.assertTrue(len(pip_install_lines) > 0, "pip install line not found")
        for line in pip_install_lines:
            self.assertTrue(line.strip().startswith('#'), 
                          f"pip install line should be commented: {line}")
    
    def test_script_basic_structure_intact(self):
        """
        Test that the basic structure of the script is intact after the patch.
        """
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # Check that essential imports are still there
        self.assertIn('import pandas as pd', content)
        self.assertIn('import numpy as np', content)
        self.assertIn('from sklearn.model_selection import train_test_split', content)
        
        # Check that key functions are still there
        self.assertIn('def add_company(inpt):', content)
        self.assertIn('def set_processor(name):', content)
        self.assertIn('def set_os(inpt):', content)
        self.assertIn('def model_acc(model):', content)


class TestOriginalBugDemonstration(unittest.TestCase):
    """
    This test class demonstrates what the bug was by creating a minimal example.
    """
    
    def test_pip_install_causes_syntax_error(self):
        """
        Demonstrate that having 'pip install scikit-learn' in Python code causes SyntaxError.
        This shows what the bug was before it was fixed.
        """
        # This is what was in the original script that caused the bug
        problematic_code = """
import pandas as pd
pip install scikit-learn
from sklearn.model_selection import train_test_split
"""
        
        # This should raise a SyntaxError
        with self.assertRaises(SyntaxError) as context:
            compile(problematic_code, '<string>', 'exec')
        
        # Verify it's a syntax error related to the pip install line
        self.assertIn('invalid syntax', str(context.exception).lower())
    
    def test_commented_pip_install_works(self):
        """
        Demonstrate that commenting out the pip install line fixes the syntax error.
        This shows what the fix does.
        """
        # This is the fixed version
        fixed_code = """
import pandas as pd
# pip install scikit-learn
from sklearn.model_selection import train_test_split
"""
        
        # This should NOT raise a SyntaxError
        try:
            compiled_code = compile(fixed_code, '<string>', 'exec')
            self.assertTrue(True, "Fixed code compiles successfully")
        except SyntaxError as e:
            self.fail(f"Fixed code should not have SyntaxError: {e}")


if __name__ == '__main__':
    unittest.main()
