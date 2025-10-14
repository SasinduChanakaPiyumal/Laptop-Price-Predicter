#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for storage feature extraction function.

This test suite provides comprehensive coverage for the extract_storage_features
function used in the laptop price prediction model.
"""

import pytest
import re


def extract_storage_features(memory_string):
    """
    Extract storage type and total capacity from memory string.
    Examples: "256GB SSD", "1TB HDD", "128GB SSD +  1TB HDD", "256GB Flash Storage"
    """
    memory_string = str(memory_string)
    
    # Initialize features
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
    # Find all capacity values with TB or GB
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    # Convert to GB and sum
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb


class TestExtractStorageFeatures:
    """Test suite for extract_storage_features function."""
    
    def test_ssd_only_gb(self):
        """Test SSD only with GB capacity."""
        result = extract_storage_features("256GB SSD")
        assert result == (1, 0, 0, 0, 256.0), "Should detect SSD and 256GB capacity"
    
    def test_ssd_only_tb(self):
        """Test SSD only with TB capacity."""
        result = extract_storage_features("1TB SSD")
        assert result == (1, 0, 0, 0, 1024.0), "Should detect SSD and convert 1TB to 1024GB"
    
    def test_hdd_only_gb(self):
        """Test HDD only with GB capacity."""
        result = extract_storage_features("500GB HDD")
        assert result == (0, 1, 0, 0, 500.0), "Should detect HDD and 500GB capacity"
    
    def test_hdd_only_tb(self):
        """Test HDD only with TB capacity."""
        result = extract_storage_features("2TB HDD")
        assert result == (0, 1, 0, 0, 2048.0), "Should detect HDD and convert 2TB to 2048GB"
    
    def test_mixed_ssd_hdd(self):
        """Test mixed SSD + HDD configuration."""
        result = extract_storage_features("128GB SSD +  1TB HDD")
        assert result == (1, 1, 0, 0, 1152.0), "Should detect both SSD and HDD, total 1152GB"
    
    def test_mixed_ssd_hdd_alternative(self):
        """Test another mixed configuration."""
        result = extract_storage_features("256GB SSD + 2TB HDD")
        assert result == (1, 1, 0, 0, 2304.0), "Should detect both SSD and HDD, total 2304GB"
    
    def test_flash_storage(self):
        """Test Flash storage detection."""
        result = extract_storage_features("256GB Flash Storage")
        assert result == (0, 0, 1, 0, 256.0), "Should detect Flash storage"
    
    def test_flash_storage_alternative(self):
        """Test Flash storage with different formatting."""
        result = extract_storage_features("128GB Flash")
        assert result == (0, 0, 1, 0, 128.0), "Should detect Flash storage"
    
    def test_hybrid_drive(self):
        """Test Hybrid drive detection."""
        result = extract_storage_features("1TB Hybrid")
        assert result == (0, 0, 0, 1, 1024.0), "Should detect Hybrid drive"
    
    def test_large_capacity_ssd(self):
        """Test large SSD capacity."""
        result = extract_storage_features("2TB SSD")
        assert result == (1, 0, 0, 0, 2048.0), "Should handle large SSD capacity"
    
    def test_multiple_ssds(self):
        """Test configuration with multiple SSDs."""
        result = extract_storage_features("512GB SSD + 512GB SSD")
        assert result == (1, 1, 0, 0, 1024.0), "Should sum multiple storage capacities"
    
    def test_decimal_tb(self):
        """Test decimal TB values."""
        result = extract_storage_features("1.5TB HDD")
        assert result == (0, 1, 0, 0, 1536.0), "Should handle decimal TB values (1.5TB = 1536GB)"
    
    def test_empty_string(self):
        """Test empty string input."""
        result = extract_storage_features("")
        assert result == (0, 0, 0, 0, 0.0), "Should handle empty string gracefully"
    
    def test_no_storage_info(self):
        """Test string with no storage information."""
        result = extract_storage_features("Unknown")
        assert result == (0, 0, 0, 0, 0.0), "Should return zeros for unknown storage"
    
    def test_none_input(self):
        """Test None input (converted to string 'None')."""
        result = extract_storage_features(None)
        # When None is converted to string, it becomes 'None'
        assert result == (0, 0, 0, 0, 0.0), "Should handle None input"
    
    def test_numeric_input(self):
        """Test numeric input converted to string."""
        result = extract_storage_features(512)
        assert result == (0, 0, 0, 0, 0.0), "Should handle numeric input (no GB/TB units)"
    
    def test_triple_storage(self):
        """Test configuration with three storage devices."""
        result = extract_storage_features("128GB SSD + 256GB SSD + 1TB HDD")
        assert result == (1, 1, 0, 0, 1408.0), "Should sum all three storage capacities"
    
    def test_case_sensitivity(self):
        """Test that detection is case-sensitive (as implemented)."""
        result = extract_storage_features("256gb ssd")
        # Current implementation is case-sensitive, so lowercase won't match
        assert result == (0, 0, 0, 0, 0.0), "Current implementation is case-sensitive"
    
    def test_mixed_case_capacity(self):
        """Test mixed case in capacity units (GB/gb)."""
        result = extract_storage_features("256gb SSD")
        # The regex looks for GB (uppercase), so gb won't be captured
        assert result == (1, 0, 0, 0, 0.0), "SSD detected but capacity not captured due to lowercase gb"
    
    def test_ssd_hdd_different_order(self):
        """Test SSD + HDD in different order."""
        result = extract_storage_features("1TB HDD + 256GB SSD")
        assert result == (1, 1, 0, 0, 1280.0), "Should work regardless of order"
    
    def test_extra_whitespace(self):
        """Test handling of extra whitespace."""
        result = extract_storage_features("256GB   SSD")
        assert result == (1, 0, 0, 0, 256.0), "Should handle extra whitespace"
    
    def test_no_space_between_number_and_unit(self):
        """Test when there's no space between number and GB/TB."""
        result = extract_storage_features("256GBSSD")
        # The regex allows optional whitespace
        assert result == (1, 0, 0, 0, 256.0), "Should handle no space between capacity and unit"


# Pytest will automatically discover and run these tests when you run:
# pytest test_storage_features.py -v

if __name__ == "__main__":
    # Allow running tests directly with: python test_storage_features.py
    pytest.main([__file__, "-v"])
