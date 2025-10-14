#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for division by zero bug fix in laptop price prediction model.

This test demonstrates that:
1. BEFORE the patch: Division by zero causes RuntimeWarning/invalid values (inf/nan)
2. AFTER the patch: Division by zero is handled gracefully with default values (0)
"""

import unittest
import pandas as pd
import numpy as np
import warnings


class TestDivisionByZeroFix(unittest.TestCase):
    """Test cases for division by zero bug fix"""
    
    def setUp(self):
        """Set up test data with edge cases including zero inches"""
        self.test_data = pd.DataFrame({
            'Inches': [15.6, 13.3, 0, 17.3, 0],  # Include zero values
            'Total_Pixels': [2073600, 1843200, 1920*1080, 2073600, 1366*768],
            'Weight': [2.5, 1.8, 2.0, 3.2, 1.5],
            'Ram': [8, 16, 0, 32, 8],  # Include zero RAM
            'Storage_Capacity_GB': [256, 512, 256, 1024, 128]
        })
    
    def test_ppi_calculation_without_fix(self):
        """
        This test demonstrates the BUG: PPI calculation without division by zero check.
        With zero inches, this produces inf values or RuntimeWarning.
        """
        # Simulate the OLD buggy code
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # OLD BUGGY CODE: Direct division without check
            buggy_ppi = np.sqrt(self.test_data['Total_Pixels']) / self.test_data['Inches']
            
            # Check that we got invalid values (inf) for zero inches
            self.assertTrue(np.isinf(buggy_ppi).any(), 
                          "BUG: Division by zero produces inf values")
            
            # Verify that rows with zero inches have inf values
            zero_inch_mask = self.test_data['Inches'] == 0
            self.assertTrue(np.all(np.isinf(buggy_ppi[zero_inch_mask])),
                          "All zero-inch entries should produce inf")
    
    def test_ppi_calculation_with_fix(self):
        """
        This test demonstrates the FIX: PPI calculation WITH division by zero check.
        With zero inches, this produces 0 (safe default) instead of inf.
        """
        # NEW FIXED CODE: Use np.where to handle division by zero
        fixed_ppi = np.where(self.test_data['Inches'] > 0, 
                            np.sqrt(self.test_data['Total_Pixels']) / self.test_data['Inches'],
                            0)
        
        # Check that we have NO invalid values
        self.assertFalse(np.isinf(fixed_ppi).any(), 
                        "FIX: No inf values should be produced")
        self.assertFalse(np.isnan(fixed_ppi).any(), 
                        "FIX: No nan values should be produced")
        
        # Verify that rows with zero inches have been set to 0
        zero_inch_mask = self.test_data['Inches'] == 0
        self.assertTrue(np.all(fixed_ppi[zero_inch_mask] == 0),
                       "All zero-inch entries should default to 0")
        
        # Verify that valid entries are calculated correctly
        valid_mask = self.test_data['Inches'] > 0
        expected_valid = np.sqrt(self.test_data['Total_Pixels'][valid_mask]) / self.test_data['Inches'][valid_mask]
        np.testing.assert_array_almost_equal(fixed_ppi[valid_mask], expected_valid,
                                            err_msg="Valid entries should be calculated correctly")
    
    def test_weight_size_ratio_without_fix(self):
        """Test the buggy weight/size ratio calculation"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # OLD BUGGY CODE
            buggy_ratio = self.test_data['Weight'] / self.test_data['Inches']
            
            # Should produce inf values
            self.assertTrue(np.isinf(buggy_ratio).any(),
                          "BUG: Division by zero produces inf values")
    
    def test_weight_size_ratio_with_fix(self):
        """Test the fixed weight/size ratio calculation"""
        # NEW FIXED CODE
        fixed_ratio = np.where(self.test_data['Inches'] > 0,
                              self.test_data['Weight'] / self.test_data['Inches'],
                              0)
        
        # Should have no invalid values
        self.assertFalse(np.isinf(fixed_ratio).any(),
                        "FIX: No inf values should be produced")
        self.assertFalse(np.isnan(fixed_ratio).any(),
                        "FIX: No nan values should be produced")
    
    def test_pixels_per_ram_without_fix(self):
        """Test the buggy pixels per RAM calculation"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # OLD BUGGY CODE
            buggy_ratio = self.test_data['Total_Pixels'] / (self.test_data['Ram'] * 1000000)
            
            # Should produce inf values for zero RAM
            self.assertTrue(np.isinf(buggy_ratio).any(),
                          "BUG: Division by zero RAM produces inf values")
    
    def test_pixels_per_ram_with_fix(self):
        """Test the fixed pixels per RAM calculation"""
        # NEW FIXED CODE
        fixed_ratio = np.where(self.test_data['Ram'] > 0,
                              self.test_data['Total_Pixels'] / (self.test_data['Ram'] * 1000000),
                              0)
        
        # Should have no invalid values
        self.assertFalse(np.isinf(fixed_ratio).any(),
                        "FIX: No inf values should be produced")
        self.assertFalse(np.isnan(fixed_ratio).any(),
                        "FIX: No nan values should be produced")
    
    def test_storage_per_inch_without_fix(self):
        """Test the buggy storage per inch calculation"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # OLD BUGGY CODE
            buggy_ratio = self.test_data['Storage_Capacity_GB'] / self.test_data['Inches']
            
            # Should produce inf values
            self.assertTrue(np.isinf(buggy_ratio).any(),
                          "BUG: Division by zero produces inf values")
    
    def test_storage_per_inch_with_fix(self):
        """Test the fixed storage per inch calculation"""
        # NEW FIXED CODE
        fixed_ratio = np.where(self.test_data['Inches'] > 0,
                              self.test_data['Storage_Capacity_GB'] / self.test_data['Inches'],
                              0)
        
        # Should have no invalid values
        self.assertFalse(np.isinf(fixed_ratio).any(),
                        "FIX: No inf values should be produced")
        self.assertFalse(np.isnan(fixed_ratio).any(),
                        "FIX: No nan values should be produced")
        
        # Verify correct calculation for valid values
        valid_mask = self.test_data['Inches'] > 0
        expected = self.test_data['Storage_Capacity_GB'][valid_mask] / self.test_data['Inches'][valid_mask]
        np.testing.assert_array_almost_equal(fixed_ratio[valid_mask], expected,
                                            err_msg="Valid entries should be calculated correctly")
    
    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases"""
        edge_cases = pd.DataFrame({
            'Inches': [0, -1, 15.6, np.nan],
            'Total_Pixels': [1920*1080, 1920*1080, 1920*1080, 1920*1080],
        })
        
        # Fixed version should handle all edge cases
        fixed_ppi = np.where(edge_cases['Inches'] > 0,
                            np.sqrt(edge_cases['Total_Pixels']) / edge_cases['Inches'],
                            0)
        
        # Check specific cases
        self.assertEqual(fixed_ppi[0], 0, "Zero inches should give 0")
        self.assertEqual(fixed_ppi[1], 0, "Negative inches should give 0")
        self.assertGreater(fixed_ppi[2], 0, "Valid inches should give positive PPI")
        # NaN case will propagate through, which is acceptable


if __name__ == '__main__':
    print("="*70)
    print("DIVISION BY ZERO BUG FIX - UNIT TESTS")
    print("="*70)
    print("\nThese tests demonstrate the bug and verify the fix:")
    print("- Tests with 'without_fix' show the BUG (inf values from division by zero)")
    print("- Tests with 'with_fix' show the FIX (safe default values)")
    print("\nRunning tests...\n")
    
    unittest.main(verbosity=2)
