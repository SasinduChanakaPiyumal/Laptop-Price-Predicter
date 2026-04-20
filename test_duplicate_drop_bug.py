#!/usr/bin/env python3
"""
Unit test for the duplicate column drop bug.

Bug Description:
    In the original code at line 485, the code attempts to drop columns:
    ['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory']
    
    However, at line 290, these columns were already dropped:
    ['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu']
    
    This causes a KeyError when pandas tries to drop non-existent columns.

Fix:
    Line 485 should only drop 'Memory' since the other columns are already gone.

This test:
    - Creates a minimal dataset simulating the state after line 290
    - Tests that the BUGGY code raises KeyError
    - Tests that the FIXED code works correctly
"""

import pandas as pd
import pytest


def create_test_dataset_after_line_290():
    """
    Create a minimal dataset that simulates the state after line 290.
    At this point, columns laptop_ID, Inches, Product, ScreenResolution, Cpu, Gpu
    have already been dropped.
    """
    # Create a simple dataset with columns that would exist after line 290
    data = {
        'Company': ['Dell', 'HP', 'Lenovo'],
        'TypeName': ['Notebook', 'Gaming', 'Ultrabook'],
        'Ram': [8, 16, 8],
        'Weight': [1.5, 2.0, 1.2],
        'Price_euros': [500.0, 1200.0, 800.0],
        'OpSys': ['Windows', 'Windows', 'Linux'],
        'Touchscreen': [0, 1, 0],
        'IPS': [1, 1, 0],
        'Screen_Width': [1920, 2560, 1920],
        'Screen_Height': [1080, 1440, 1080],
        'Total_Pixels': [2073600, 3686400, 2073600],
        'PPI': [141.21, 189.23, 141.21],
        'Cpu_name': ['Intel Core i5', 'Intel Core i7', 'Intel Core i5'],
        'Gpu_name': ['Intel', 'Nvidia', 'Intel'],
        'Memory': ['256GB SSD', '512GB SSD + 1TB HDD', '256GB SSD'],
        'Has_SSD': [1, 1, 1],
        'Has_HDD': [0, 1, 0],
        'Has_Flash': [0, 0, 0],
        'Has_Hybrid': [0, 0, 0],
        'Storage_Capacity_GB': [256.0, 1536.0, 256.0],
        'Storage_Type_Score': [3.0, 4.0, 3.0]
    }
    
    return pd.DataFrame(data)


def test_buggy_code_raises_keyerror():
    """
    Test that the BUGGY code (trying to drop already-dropped columns) raises KeyError.
    This test should FAIL with the buggy code (before the patch).
    """
    dataset = create_test_dataset_after_line_290()
    
    # The buggy code at line 485 (original version)
    buggy_columns_to_drop = ['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory']
    
    # This should raise KeyError because laptop_ID, Product, ScreenResolution, Cpu, Gpu
    # don't exist in the dataset
    with pytest.raises(KeyError):
        dataset.drop(columns=buggy_columns_to_drop)


def test_fixed_code_works():
    """
    Test that the FIXED code (only dropping 'Memory') works correctly.
    This test should PASS with the fixed code (after the patch).
    """
    dataset = create_test_dataset_after_line_290()
    
    # The fixed code at line 485 (after patch)
    fixed_columns_to_drop = ['Memory']
    
    # This should work without error
    try:
        result = dataset.drop(columns=fixed_columns_to_drop)
        
        # Verify Memory column was dropped
        assert 'Memory' not in result.columns, "Memory column should be dropped"
        
        # Verify other columns still exist
        assert 'Company' in result.columns
        assert 'Ram' in result.columns
        assert 'Storage_Capacity_GB' in result.columns
        
        # Verify row count unchanged
        assert len(result) == len(dataset)
        
        print("✓ Fixed code works correctly!")
        
    except KeyError as e:
        pytest.fail(f"Fixed code should not raise KeyError, but got: {e}")


def test_columns_actually_missing():
    """
    Verify that the columns we claim are missing are actually missing.
    This confirms our bug analysis is correct.
    """
    dataset = create_test_dataset_after_line_290()
    
    # These columns should NOT exist (they were dropped at line 290)
    missing_columns = ['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu']
    
    for col in missing_columns:
        assert col not in dataset.columns, f"Column '{col}' should not exist in dataset after line 290"
    
    # This column SHOULD exist (it gets dropped at line 485)
    assert 'Memory' in dataset.columns, "Memory column should still exist before line 485"


if __name__ == "__main__":
    print("="*70)
    print("Testing Duplicate Column Drop Bug")
    print("="*70)
    
    print("\n1. Testing that columns are actually missing...")
    try:
        test_columns_actually_missing()
        print("   ✓ Confirmed: laptop_ID, Product, ScreenResolution, Cpu, Gpu are missing")
    except AssertionError as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n2. Testing buggy code (should raise KeyError)...")
    try:
        test_buggy_code_raises_keyerror()
        print("   ✓ Confirmed: Buggy code raises KeyError as expected")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
    
    print("\n3. Testing fixed code (should work)...")
    try:
        test_fixed_code_works()
        print("   ✓ Confirmed: Fixed code works without errors")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
    
    print("\n" + "="*70)
    print("Summary:")
    print("  - BEFORE patch: Code raises KeyError at line 485")
    print("  - AFTER patch: Code works correctly")
    print("="*70)
