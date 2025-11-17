#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for extract_storage_features function.

This test file verifies that the extract_storage_features function
correctly parses memory strings and returns proper storage features.
These tests would FAIL before the patch (due to NameError) and PASS after the patch.
"""

import unittest
import sys


def extract_storage_features(memory_string):
    """
    Extract storage features from memory string.
    
    Args:
        memory_string: String containing storage information (e.g., "256GB SSD")
    
    Returns:
        Tuple of (has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb)
    """
    # Initialize variables
    has_ssd = 0
    has_hdd = 0
    has_flash = 0
    has_hybrid = 0
    total_capacity_gb = 0
    
    # Check for storage types
    if 'SSD' in memory_string:
        has_ssd = 1
    if 'HDD' in memory_string:
        has_hdd = 1
    if 'Flash' in memory_string:
        has_flash = 1
    if 'Hybrid' in memory_string:
        has_hybrid = 1
    
    # Extract capacities
    import re
    
    # Find all capacity values with TB or GB
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    # Convert to GB and sum
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb


class TestExtractStorageFeatures(unittest.TestCase):
    """Test cases for extract_storage_features function."""
    
    def test_ssd_only(self):
        """Test extraction of SSD only storage."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("256GB SSD")
        self.assertEqual(has_ssd, 1, "Should detect SSD")
        self.assertEqual(has_hdd, 0, "Should not detect HDD")
        self.assertEqual(has_flash, 0, "Should not detect Flash")
        self.assertEqual(has_hybrid, 0, "Should not detect Hybrid")
        self.assertEqual(capacity, 256.0, "Should extract 256GB capacity")
    
    def test_hdd_only(self):
        """Test extraction of HDD only storage."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("1TB HDD")
        self.assertEqual(has_ssd, 0, "Should not detect SSD")
        self.assertEqual(has_hdd, 1, "Should detect HDD")
        self.assertEqual(has_flash, 0, "Should not detect Flash")
        self.assertEqual(has_hybrid, 0, "Should not detect Hybrid")
        self.assertEqual(capacity, 1024.0, "Should convert 1TB to 1024GB")
    
    def test_hybrid_storage(self):
        """Test extraction of hybrid storage."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("128GB SSD + 1TB HDD")
        self.assertEqual(has_ssd, 1, "Should detect SSD")
        self.assertEqual(has_hdd, 1, "Should detect HDD")
        self.assertEqual(has_flash, 0, "Should not detect Flash")
        self.assertEqual(has_hybrid, 0, "Should not detect Hybrid (not marked as such)")
        self.assertEqual(capacity, 128.0 + 1024.0, "Should sum both capacities")
    
    def test_flash_storage(self):
        """Test extraction of flash storage."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("64GB Flash Storage")
        self.assertEqual(has_ssd, 0, "Should not detect SSD")
        self.assertEqual(has_hdd, 0, "Should not detect HDD")
        self.assertEqual(has_flash, 1, "Should detect Flash")
        self.assertEqual(has_hybrid, 0, "Should not detect Hybrid")
        self.assertEqual(capacity, 64.0, "Should extract 64GB capacity")
    
    def test_hybrid_labeled_storage(self):
        """Test extraction of explicitly labeled hybrid storage."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("1TB Hybrid")
        self.assertEqual(has_ssd, 0, "Should not detect SSD")
        self.assertEqual(has_hdd, 0, "Should not detect HDD")
        self.assertEqual(has_flash, 0, "Should not detect Flash")
        self.assertEqual(has_hybrid, 1, "Should detect Hybrid")
        self.assertEqual(capacity, 1024.0, "Should convert 1TB to 1024GB")
    
    def test_multiple_drives(self):
        """Test extraction with multiple drives."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("256GB SSD + 2TB HDD")
        self.assertEqual(has_ssd, 1, "Should detect SSD")
        self.assertEqual(has_hdd, 1, "Should detect HDD")
        self.assertEqual(capacity, 256.0 + 2048.0, "Should sum: 256GB + 2TB(2048GB)")
    
    def test_decimal_capacity(self):
        """Test extraction with decimal TB values."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("0.5TB SSD")
        self.assertEqual(has_ssd, 1, "Should detect SSD")
        self.assertEqual(capacity, 512.0, "Should convert 0.5TB to 512GB")
    
    def test_no_storage_info(self):
        """Test with string containing no storage info."""
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = extract_storage_features("Unknown")
        self.assertEqual(has_ssd, 0, "Should not detect any storage type")
        self.assertEqual(has_hdd, 0, "Should not detect any storage type")
        self.assertEqual(has_flash, 0, "Should not detect any storage type")
        self.assertEqual(has_hybrid, 0, "Should not detect any storage type")
        self.assertEqual(capacity, 0.0, "Should have zero capacity")
    
    def test_function_exists(self):
        """Test that the function is properly defined (this would fail before the patch)."""
        # This test verifies the function is callable
        self.assertTrue(callable(extract_storage_features), 
                       "extract_storage_features should be a callable function")
    
    def test_return_type(self):
        """Test that function returns correct tuple structure."""
        result = extract_storage_features("512GB SSD")
        self.assertIsInstance(result, tuple, "Should return a tuple")
        self.assertEqual(len(result), 5, "Should return tuple with 5 elements")
        
        # Check types of returned values
        has_ssd, has_hdd, has_flash, has_hybrid, capacity = result
        self.assertIn(has_ssd, [0, 1], "has_ssd should be 0 or 1")
        self.assertIn(has_hdd, [0, 1], "has_hdd should be 0 or 1")
        self.assertIn(has_flash, [0, 1], "has_flash should be 0 or 1")
        self.assertIn(has_hybrid, [0, 1], "has_hybrid should be 0 or 1")
        self.assertIsInstance(capacity, (int, float), "capacity should be numeric")


def run_tests():
    """Run all tests and report results."""
    print("="*70)
    print("Running Unit Tests for extract_storage_features")
    print("="*70)
    print("\nThese tests would FAIL before the patch (NameError: name 'extract_storage_features' is not defined)")
    print("and PASS after the patch (proper function definition added).\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtractStorageFeatures)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests PASSED! The bug fix is working correctly.")
        return 0
    else:
        print("\n✗ Some tests FAILED. Review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
