#!/usr/bin/env python
# coding: utf-8

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
import os

# Import the utility functions
from laptop_price_utils import (
    add_company, set_processor, set_os, model_acc,
    clean_ram_data, clean_weight_data,
    extract_touchscreen_feature, extract_ips_feature,
    extract_cpu_name, extract_gpu_name
)


class TestLaptopPriceFunctions(unittest.TestCase):
    """Test suite for laptop price model functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample test data
        self.sample_data = {
            'laptop_ID': [1, 2, 3, 4, 5],
            'Company': ['Apple', 'Samsung', 'HP', 'Dell', 'Microsoft'],
            'Product': ['MacBook Pro', 'Galaxy Book', 'Pavilion', 'Inspiron', 'Surface'],
            'TypeName': ['Notebook', 'Notebook', 'Gaming', 'Notebook', '2 in 1 Convertible'],
            'Inches': [13.3, 15.6, 15.6, 14.0, 13.5],
            'ScreenResolution': ['IPS Panel Retina Display 2560x1600', 
                                'Full HD 1920x1080', 
                                'Full HD 1920x1080',
                                'Touchscreen Full HD 1920x1080',
                                'IPS Panel Touchscreen 2256x1504'],
            'Cpu': ['Intel Core i7 2.9GHz', 'Intel Core i5 2.3GHz', 'AMD Ryzen 5', 
                    'Intel Core i3 2.1GHz', 'Intel Core m3 1.2GHz'],
            'Ram': ['8GB', '16GB', '8GB', '4GB', '8GB'],
            'Memory': ['512GB SSD', '256GB SSD', '1TB HDD', '500GB HDD', '128GB SSD'],
            'Gpu': ['Intel Iris Plus Graphics 650', 'Intel UHD Graphics 620', 
                    'AMD Radeon RX 560X', 'Intel HD Graphics 620', 'Intel HD Graphics 615'],
            'OpSys': ['macOS', 'Windows 10', 'Windows 10', 'Linux', 'Windows 10 S'],
            'Weight': ['1.37kg', '1.8kg', '2.5kg', '1.65kg', '1.1kg'],
            'Price_euros': [1339.69, 898.95, 575.00, 379.94, 1499.00]
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_add_company_function(self):
        """Test the add_company function for proper company categorization."""
        
        # Test companies that should be categorized as 'Other'
        self.assertEqual(add_company('Samsung'), 'Other')
        self.assertEqual(add_company('Microsoft'), 'Other')
        self.assertEqual(add_company('Google'), 'Other')
        self.assertEqual(add_company('LG'), 'Other')
        self.assertEqual(add_company('Huawei'), 'Other')
        
        # Test companies that should remain unchanged
        self.assertEqual(add_company('Apple'), 'Apple')
        self.assertEqual(add_company('HP'), 'HP')
        self.assertEqual(add_company('Dell'), 'Dell')
        self.assertEqual(add_company('Lenovo'), 'Lenovo')
        
        # Test edge cases
        self.assertEqual(add_company(''), '')
        self.assertEqual(add_company('Unknown Brand'), 'Unknown Brand')
    
    def test_set_processor_function(self):
        """Test the set_processor function for CPU name processing."""
        
        # Test Intel processors that should remain unchanged
        self.assertEqual(set_processor('Intel Core i7'), 'Intel Core i7')
        self.assertEqual(set_processor('Intel Core i5'), 'Intel Core i5')
        self.assertEqual(set_processor('Intel Core i3'), 'Intel Core i3')
        
        # Test AMD processors
        self.assertEqual(set_processor('AMD Ryzen 5'), 'AMD')
        self.assertEqual(set_processor('AMD FX-8350'), 'AMD')
        self.assertEqual(set_processor('AMD A10'), 'AMD')
        
        # Test other processors
        self.assertEqual(set_processor('Intel Core m3'), 'Other')
        self.assertEqual(set_processor('Intel Celeron'), 'Other')
        self.assertEqual(set_processor('ARM Cortex'), 'Other')
        
        # Test edge cases
        self.assertEqual(set_processor(''), 'Other')
    
    def test_set_os_function(self):
        """Test the set_os function for operating system categorization."""
        
        # Test Windows variants
        self.assertEqual(set_os('Windows 10'), 'Windows')
        self.assertEqual(set_os('Windows 7'), 'Windows')
        self.assertEqual(set_os('Windows 10 S'), 'Windows')
        
        # Test Mac variants
        self.assertEqual(set_os('macOS'), 'Mac')
        self.assertEqual(set_os('Mac OS X'), 'Mac')
        
        # Test Linux
        self.assertEqual(set_os('Linux'), 'Linux')
        
        # Test other OS
        self.assertEqual(set_os('Chrome OS'), 'Other')
        self.assertEqual(set_os('FreeDOS'), 'Other')
        self.assertEqual(set_os('No OS'), 'Other')
        
        # Test edge cases
        self.assertEqual(set_os(''), 'Other')
    
    def test_model_acc_function(self):
        """Test the model_acc function for model evaluation."""
        
        # Create simple test data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([1, 2, 3, 4, 5, 6])
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        # Test with LinearRegression
        model = LinearRegression()
        score = model_acc(model, x_train, x_test, y_train, y_test)
        
        # Score should be a float between 0 and 1 (or potentially negative for very poor fits)
        self.assertIsInstance(score, float)
        # For this simple linear relationship, score should be very high
        self.assertGreater(score, 0.8)
    
    def test_data_preprocessing_ram_cleaning(self):
        """Test RAM data cleaning functionality."""
        # Test RAM cleaning - removing 'GB' and converting to int
        test_ram = pd.Series(['8GB', '16GB', '4GB', '32GB'])
        expected_ram = pd.Series([8, 16, 4, 32], dtype='int32')
        
        cleaned_ram = clean_ram_data(test_ram)
        pd.testing.assert_series_equal(cleaned_ram, expected_ram)
    
    def test_data_preprocessing_weight_cleaning(self):
        """Test Weight data cleaning functionality."""
        # Test Weight cleaning - removing 'kg' and converting to float
        test_weight = pd.Series(['1.37kg', '2.5kg', '1.8kg', '1.1kg'])
        expected_weight = pd.Series([1.37, 2.5, 1.8, 1.1], dtype='float64')
        
        cleaned_weight = clean_weight_data(test_weight)
        pd.testing.assert_series_equal(cleaned_weight, expected_weight)
    
    def test_touchscreen_extraction(self):
        """Test touchscreen feature extraction from screen resolution."""
        test_resolutions = pd.Series([
            'IPS Panel Retina Display 2560x1600',
            'Touchscreen Full HD 1920x1080',
            'IPS Panel Touchscreen 2256x1504',
            'Full HD 1920x1080'
        ])
        
        touchscreen_flags = extract_touchscreen_feature(test_resolutions)
        expected_flags = pd.Series([0, 1, 1, 0])
        
        pd.testing.assert_series_equal(touchscreen_flags, expected_flags)
    
    def test_ips_extraction(self):
        """Test IPS feature extraction from screen resolution."""
        test_resolutions = pd.Series([
            'IPS Panel Retina Display 2560x1600',
            'Touchscreen Full HD 1920x1080',
            'IPS Panel Touchscreen 2256x1504',
            'Full HD 1920x1080'
        ])
        
        ips_flags = extract_ips_feature(test_resolutions)
        expected_flags = pd.Series([1, 0, 1, 0])
        
        pd.testing.assert_series_equal(ips_flags, expected_flags)
    
    def test_cpu_name_extraction(self):
        """Test CPU name extraction logic."""
        test_cpus = pd.Series([
            'Intel Core i7 2.9GHz',
            'AMD Ryzen 5 3500U 2.1GHz',
            'Intel Core i5 8250U 1.6GHz',
            'ARM Cortex A72'
        ])
        
        cpu_names = extract_cpu_name(test_cpus)
        expected_names = pd.Series([
            'Intel Core i7',
            'AMD Ryzen 5',
            'Intel Core i5',
            'ARM Cortex A72'
        ])
        
        pd.testing.assert_series_equal(cpu_names, expected_names)
    
    def test_gpu_name_extraction(self):
        """Test GPU name extraction logic."""
        test_gpus = pd.Series([
            'Intel Iris Plus Graphics 650',
            'NVIDIA GeForce GTX 1060',
            'AMD Radeon RX 560X',
            'Intel HD Graphics 620'
        ])
        
        gpu_names = extract_gpu_name(test_gpus)
        expected_names = pd.Series(['Intel', 'NVIDIA', 'AMD', 'Intel'])
        
        pd.testing.assert_series_equal(gpu_names, expected_names)


if __name__ == '__main__':
    unittest.main()
