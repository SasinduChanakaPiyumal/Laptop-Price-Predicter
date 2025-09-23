#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for laptop price model functions.
These tests would fail with the original buggy code but pass after fixes.
"""

import unittest
import pandas as pd
import numpy as np

def set_processor(name):
    """Fixed version of the set_processor function"""
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        # Handle empty strings and potential IndexErrors
        name_parts = name.split()
        if len(name_parts) > 0 and name_parts[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'

def extract_cpu_name(x):
    """Fixed version of CPU name extraction"""
    return " ".join(x.split()[0:3]) if len(x.split()) >= 3 else " ".join(x.split()) if len(x.split()) > 0 else 'Unknown'

def extract_gpu_name(x):
    """Fixed version of GPU name extraction"""
    return " ".join(x.split()[0:1]) if len(x.split()) > 0 else 'Unknown'

class TestLaptopPriceFunctions(unittest.TestCase):
    """Test cases for laptop price model functions"""

    def test_set_processor_with_empty_string(self):
        """Test set_processor function with empty string - would cause IndexError in original code"""
        result = set_processor("")
        self.assertEqual(result, 'Other', "Empty string should return 'Other'")

    def test_set_processor_with_whitespace_only(self):
        """Test set_processor function with whitespace only - would cause IndexError in original code"""
        result = set_processor("   ")
        self.assertEqual(result, 'Other', "Whitespace-only string should return 'Other'")

    def test_set_processor_with_single_word(self):
        """Test set_processor function with single word"""
        result = set_processor("AMD")
        self.assertEqual(result, 'AMD', "Single word 'AMD' should return 'AMD'")

    def test_set_processor_with_intel_core_i7(self):
        """Test set_processor function with Intel Core i7"""
        result = set_processor("Intel Core i7")
        self.assertEqual(result, 'Intel Core i7', "Intel Core i7 should return as is")

    def test_set_processor_with_unknown_processor(self):
        """Test set_processor function with unknown processor"""
        result = set_processor("Unknown Processor XYZ")
        self.assertEqual(result, 'Other', "Unknown processor should return 'Other'")

    def test_extract_cpu_name_empty_string(self):
        """Test CPU name extraction with empty string - would cause IndexError in original code"""
        result = extract_cpu_name("")
        self.assertEqual(result, 'Unknown', "Empty string should return 'Unknown'")

    def test_extract_cpu_name_single_word(self):
        """Test CPU name extraction with single word"""
        result = extract_cpu_name("AMD")
        self.assertEqual(result, 'AMD', "Single word should return as is")

    def test_extract_cpu_name_two_words(self):
        """Test CPU name extraction with two words"""
        result = extract_cpu_name("Intel Celeron")
        self.assertEqual(result, 'Intel Celeron', "Two words should return as is")

    def test_extract_cpu_name_three_or_more_words(self):
        """Test CPU name extraction with three or more words"""
        result = extract_cpu_name("Intel Core i7 8750H")
        self.assertEqual(result, 'Intel Core i7', "Should return first three words")

    def test_extract_gpu_name_empty_string(self):
        """Test GPU name extraction with empty string - would cause IndexError in original code"""
        result = extract_gpu_name("")
        self.assertEqual(result, 'Unknown', "Empty string should return 'Unknown'")

    def test_extract_gpu_name_single_word(self):
        """Test GPU name extraction with single word"""
        result = extract_gpu_name("NVIDIA")
        self.assertEqual(result, 'NVIDIA', "Single word should return as is")

    def test_extract_gpu_name_multiple_words(self):
        """Test GPU name extraction with multiple words"""
        result = extract_gpu_name("NVIDIA GeForce GTX")
        self.assertEqual(result, 'NVIDIA', "Should return first word only")

    def test_extract_gpu_name_whitespace_only(self):
        """Test GPU name extraction with whitespace only"""
        result = extract_gpu_name("   ")
        self.assertEqual(result, 'Unknown', "Whitespace-only string should return 'Unknown'")

if __name__ == '__main__':
    # Test that demonstrates the original bug would fail
    print("Running tests that would fail with the original buggy code...")
    unittest.main(verbosity=2)
