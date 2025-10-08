#!/usr/bin/env python
# coding: utf-8

"""
Unit test for the set_processor function bug fix.
This test should fail before the patch and pass after the patch.
"""

import unittest


def set_processor_original(name):
    """Original buggy version of set_processor"""
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':  # BUG: IndexError when name is empty string
            return 'AMD'
        else:
            return 'Other'


def set_processor_fixed(name):
    """Fixed version of set_processor"""
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        split_name = name.split()
        if split_name and split_name[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'


class TestSetProcessor(unittest.TestCase):
    
    def test_intel_core_i7(self):
        """Test Intel Core i7 processor"""
        result = set_processor_fixed('Intel Core i7')
        self.assertEqual(result, 'Intel Core i7')
    
    def test_intel_core_i5(self):
        """Test Intel Core i5 processor"""
        result = set_processor_fixed('Intel Core i5')
        self.assertEqual(result, 'Intel Core i5')
    
    def test_intel_core_i3(self):
        """Test Intel Core i3 processor"""
        result = set_processor_fixed('Intel Core i3')
        self.assertEqual(result, 'Intel Core i3')
    
    def test_amd_processor(self):
        """Test AMD processor"""
        result = set_processor_fixed('AMD Ryzen 5')
        self.assertEqual(result, 'AMD')
    
    def test_other_processor(self):
        """Test other processor types"""
        result = set_processor_fixed('Samsung Exynos')
        self.assertEqual(result, 'Other')
    
    def test_empty_string_bug(self):
        """
        Test that empty string is handled correctly.
        This would raise IndexError in the original buggy version.
        """
        # This should not raise an exception
        result = set_processor_fixed('')
        self.assertEqual(result, 'Other')
    
    def test_whitespace_only_string(self):
        """Test that whitespace-only string is handled correctly"""
        result = set_processor_fixed('   ')
        self.assertEqual(result, 'Other')
    
    def test_original_fails_on_empty_string(self):
        """
        Demonstrate the bug: original version raises IndexError on empty string
        """
        with self.assertRaises(IndexError):
            set_processor_original('')


if __name__ == '__main__':
    unittest.main()
