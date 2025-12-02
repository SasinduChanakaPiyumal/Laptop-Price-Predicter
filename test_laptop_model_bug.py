#!/usr/bin/env python3
"""
Unit test for the duplicate column drop bug in Laptop Price model(1).py

This test reproduces the bug where columns are dropped twice:
1. Line 290 drops columns including 'laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', and 'Inches'
2. Line 485 tries to drop some of the same columns again, causing a KeyError

The test simulates the same sequence of operations to demonstrate the bug.
"""

import pandas as pd
import numpy as np
import unittest


class TestLaptopModelColumnDropBug(unittest.TestCase):
    """Test case for the duplicate column drop bug."""
    
    def setUp(self):
        """Create a minimal test dataset similar to the laptop dataset."""
        # Create a minimal dataset with the relevant columns
        self.test_data = pd.DataFrame({
            'laptop_ID': [1, 2, 3, 4, 5],
            'Company': ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus'],
            'Product': ['Laptop1', 'Laptop2', 'Laptop3', 'Laptop4', 'Laptop5'],
            'TypeName': ['Notebook', 'Notebook', 'Ultrabook', 'Ultrabook', 'Gaming'],
            'Inches': [15.6, 13.3, 14.0, 13.3, 17.3],
            'ScreenResolution': ['1920x1080', '1366x768', '1920x1080 IPS', '2560x1600', '1920x1080 IPS Touchscreen'],
            'Cpu': ['Intel Core i5', 'Intel Core i3', 'Intel Core i7', 'Intel Core i5', 'AMD Ryzen'],
            'Gpu': ['Intel HD', 'Intel HD', 'Intel Iris', 'Intel Iris', 'Nvidia GTX'],
            'Ram': ['8GB', '4GB', '16GB', '8GB', '16GB'],
            'Memory': ['256GB SSD', '1TB HDD', '512GB SSD', '256GB SSD', '1TB HDD + 256GB SSD'],
            'OpSys': ['Windows', 'Windows', 'Mac', 'Mac', 'Windows'],
            'Weight': ['1.8kg', '2.1kg', '1.5kg', '1.3kg', '2.7kg'],
            'Price_euros': [800.0, 450.0, 1200.0, 1500.0, 1100.0]
        })
    
    def test_duplicate_column_drop_bug(self):
        """
        Test that demonstrates the bug where dropping columns twice causes KeyError.
        
        This test simulates the problematic code flow BEFORE the patch:
        1. First drop at line 290 (removes several columns including 'Inches')
        2. Storage feature extraction (happens after the first drop)
        3. Second drop at line 485 tries to drop already-dropped columns -> KeyError
        
        This test should PASS (i.e., correctly detect the KeyError) to demonstrate the bug exists.
        """
        dataset = self.test_data.copy()
        
        # Simulate feature engineering steps that happen before line 290
        dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
        dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float32')
        dataset['Touchscreen'] = dataset['ScreenResolution'].str.contains('Touchscreen', case=False, regex=False).astype('int8')
        dataset['IPS'] = dataset['ScreenResolution'].str.contains('IPS', case=False, regex=False).astype('int8')
        
        # Extract screen resolution
        resolution_pattern = r'(\d{3,4})x(\d{3,4})'
        extracted = dataset['ScreenResolution'].str.extract(resolution_pattern)
        dataset['Screen_Width'] = pd.to_numeric(extracted[0], errors='coerce').fillna(1366).astype('int16')
        dataset['Screen_Height'] = pd.to_numeric(extracted[1], errors='coerce').fillna(768).astype('int16')
        dataset['Total_Pixels'] = (dataset['Screen_Width'] * dataset['Screen_Height']).astype('int32')
        
        # CPU and GPU feature extraction
        dataset['Cpu_name'] = dataset['Cpu'].str.split().str[0:3].str.join(' ')
        dataset['Gpu_name'] = dataset['Gpu'].str.split().str[0]
        
        # FIRST DROP (line 290) - This is the BUGGY line (now removed/commented in the fix)
        # This drops columns including 'Inches' which is needed later
        dataset = dataset.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
        
        # Storage feature extraction (happens AFTER the first drop)
        dataset['Has_SSD'] = dataset['Memory'].str.contains('SSD', case=False, regex=False).astype('int8')
        dataset['Has_HDD'] = dataset['Memory'].str.contains('HDD', case=False, regex=False).astype('int8')
        
        # SECOND DROP (line 485) - This will fail because some columns were already dropped
        # This should raise a KeyError for columns like 'laptop_ID', 'Product', etc.
        with self.assertRaises(KeyError):
            dataset = dataset.drop(columns=['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory'])
        
        print("✓ Bug confirmed: Attempting to drop already-dropped columns raises KeyError")
    
    def test_inches_column_availability(self):
        """
        Test that demonstrates 'Inches' is needed after line 290.
        
        Before the patch: 'Inches' would be dropped at line 290, causing issues 
        when trying to create interaction features that depend on it (lines 532, 553, 561).
        """
        dataset = self.test_data.copy()
        
        # Feature engineering
        dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
        dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float32')
        
        resolution_pattern = r'(\d{3,4})x(\d{3,4})'
        extracted = dataset['ScreenResolution'].str.extract(resolution_pattern)
        dataset['Screen_Width'] = pd.to_numeric(extracted[0], errors='coerce').fillna(1366).astype('int16')
        dataset['Screen_Height'] = pd.to_numeric(extracted[1], errors='coerce').fillna(768).astype('int16')
        dataset['Total_Pixels'] = (dataset['Screen_Width'] * dataset['Screen_Height']).astype('int32')
        
        # BUGGY DROP (line 290) - drops 'Inches'
        dataset = dataset.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
        
        # Try to create interaction feature that needs 'Inches' (like at line 532)
        # This should fail because 'Inches' was dropped
        self.assertNotIn('Inches', dataset.columns)
        
        # Attempting to create Screen_Quality would fail or produce incorrect results
        # because 'Inches' is missing
        with self.assertRaises(KeyError):
            screen_quality = dataset['Total_Pixels'] / 1000000 * dataset['Inches']
        
        print("✓ Bug confirmed: 'Inches' column is prematurely dropped and unavailable for interaction features")
    
    def test_correct_column_drop_sequence(self):
        """
        Test that demonstrates the CORRECT way to handle column dropping.
        
        This shows that if we skip the first premature drop, everything works fine.
        """
        dataset = self.test_data.copy()
        
        # Simulate feature engineering steps
        dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
        dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float32')
        dataset['Touchscreen'] = dataset['ScreenResolution'].str.contains('Touchscreen', case=False, regex=False).astype('int8')
        dataset['IPS'] = dataset['ScreenResolution'].str.contains('IPS', case=False, regex=False).astype('int8')
        
        # Extract screen resolution
        resolution_pattern = r'(\d{3,4})x(\d{3,4})'
        extracted = dataset['ScreenResolution'].str.extract(resolution_pattern)
        dataset['Screen_Width'] = pd.to_numeric(extracted[0], errors='coerce').fillna(1366).astype('int16')
        dataset['Screen_Height'] = pd.to_numeric(extracted[1], errors='coerce').fillna(768).astype('int16')
        dataset['Total_Pixels'] = (dataset['Screen_Width'] * dataset['Screen_Height']).astype('int32')
        
        # CPU and GPU feature extraction
        dataset['Cpu_name'] = dataset['Cpu'].str.split().str[0:3].str.join(' ')
        dataset['Gpu_name'] = dataset['Gpu'].str.split().str[0]
        
        # SKIP the first premature drop (removing line 290)
        # Instead, go directly to storage feature extraction
        
        # Storage feature extraction
        dataset['Has_SSD'] = dataset['Memory'].str.contains('SSD', case=False, regex=False).astype('int8')
        dataset['Has_HDD'] = dataset['Memory'].str.contains('HDD', case=False, regex=False).astype('int8')
        
        # NOW do the drop at the end (line 485) - this should work without errors
        # This should succeed because all columns still exist
        dataset = dataset.drop(columns=['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory'])
        
        # Verify that the expected columns were dropped
        self.assertNotIn('laptop_ID', dataset.columns)
        self.assertNotIn('Product', dataset.columns)
        self.assertNotIn('ScreenResolution', dataset.columns)
        self.assertNotIn('Cpu', dataset.columns)
        self.assertNotIn('Gpu', dataset.columns)
        self.assertNotIn('Memory', dataset.columns)
        
        # Verify that 'Inches' is still present (it wasn't dropped in the corrected version)
        self.assertIn('Inches', dataset.columns)
        
        # Verify that the derived features are present
        self.assertIn('Has_SSD', dataset.columns)
        self.assertIn('Has_HDD', dataset.columns)
        self.assertIn('Total_Pixels', dataset.columns)


if __name__ == '__main__':
    print("="*70)
    print("UNIT TEST FOR LAPTOP MODEL BUG FIX")
    print("="*70)
    print("\nBug Description:")
    print("  Line 290 in 'Laptop Price model(1).py' drops columns prematurely.")
    print("  This causes two problems:")
    print("  1. Line 485 tries to drop the same columns again -> KeyError")
    print("  2. 'Inches' column is dropped but needed later for interaction features")
    print("\nFix:")
    print("  Line 290 has been commented out to prevent premature column dropping.")
    print("  All columns are now dropped at line 485 after feature extraction is complete.")
    print("\nTest Results:")
    print("-"*70)
    
    # Run the tests
    unittest.main(verbosity=2, exit=True)
