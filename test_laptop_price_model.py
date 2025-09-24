"""
Unit tests for laptop price prediction model functions
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from laptop_price_model import (
    add_company, set_processor, set_os, model_acc, preprocess_data
)


class TestAddCompany(unittest.TestCase):
    """Test cases for add_company function"""
    
    def test_add_company_other_brands(self):
        """Test that specified brands are mapped to 'Other'"""
        other_brands = ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                       'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']
        
        for brand in other_brands:
            self.assertEqual(add_company(brand), 'Other')
    
    def test_add_company_major_brands(self):
        """Test that major brands are preserved"""
        major_brands = ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI']
        
        for brand in major_brands:
            self.assertEqual(add_company(brand), brand)
    
    def test_add_company_case_sensitivity(self):
        """Test case sensitivity"""
        self.assertEqual(add_company('samsung'), 'samsung')  # lowercase not mapped
        self.assertEqual(add_company('Samsung'), 'Other')     # exact case mapped
    
    def test_add_company_invalid_input(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            add_company(123)
        with self.assertRaises(ValueError):
            add_company(None)
        with self.assertRaises(ValueError):
            add_company([])


class TestSetProcessor(unittest.TestCase):
    """Test cases for set_processor function"""
    
    def test_set_processor_intel(self):
        """Test Intel processor categorization"""
        intel_cpus = ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']
        
        for cpu in intel_cpus:
            self.assertEqual(set_processor(cpu), cpu)
    
    def test_set_processor_amd(self):
        """Test AMD processor categorization"""
        amd_cpus = ['AMD Ryzen 5', 'AMD A10', 'AMD FX']
        
        for cpu in amd_cpus:
            self.assertEqual(set_processor(cpu), 'AMD')
    
    def test_set_processor_other(self):
        """Test other processor categorization"""
        other_cpus = ['ARM Cortex', 'Qualcomm Snapdragon', 'Custom Processor']
        
        for cpu in other_cpus:
            self.assertEqual(set_processor(cpu), 'Other')
    
    def test_set_processor_edge_cases(self):
        """Test edge cases"""
        self.assertEqual(set_processor(''), 'Other')
        self.assertEqual(set_processor('   '), 'Other')
        self.assertEqual(set_processor('AMD'), 'AMD')
    
    def test_set_processor_invalid_input(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            set_processor(123)
        with self.assertRaises(ValueError):
            set_processor(None)


class TestSetOS(unittest.TestCase):
    """Test cases for set_os function"""
    
    def test_set_os_windows(self):
        """Test Windows OS categorization"""
        windows_os = ['Windows 10', 'Windows 7', 'Windows 10 S']
        
        for os in windows_os:
            self.assertEqual(set_os(os), 'Windows')
    
    def test_set_os_mac(self):
        """Test Mac OS categorization"""
        mac_os = ['macOS', 'Mac OS X']
        
        for os in mac_os:
            self.assertEqual(set_os(os), 'Mac')
    
    def test_set_os_linux(self):
        """Test Linux OS categorization"""
        self.assertEqual(set_os('Linux'), 'Linux')
    
    def test_set_os_other(self):
        """Test other OS categorization"""
        other_os = ['Chrome OS', 'Android', 'FreeBSD', 'Unknown']
        
        for os in other_os:
            self.assertEqual(set_os(os), 'Other')
    
    def test_set_os_edge_cases(self):
        """Test edge cases"""
        self.assertEqual(set_os(''), 'Other')
        self.assertEqual(set_os('   '), 'Other')
        self.assertEqual(set_os('windows 10'), 'Other')  # case sensitive
    
    def test_set_os_invalid_input(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            set_os(123)
        with self.assertRaises(ValueError):
            set_os(None)


class TestModelAcc(unittest.TestCase):
    """Test cases for model_acc function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    def test_model_acc_linear_regression(self):
        """Test model accuracy with Linear Regression"""
        model = LinearRegression()
        acc = model_acc(model, self.x_train, self.y_train, self.x_test, self.y_test)
        
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, -1)  # R² can be negative for bad models
    
    def test_model_acc_decision_tree(self):
        """Test model accuracy with Decision Tree"""
        model = DecisionTreeRegressor(random_state=42)
        acc = model_acc(model, self.x_train, self.y_train, self.x_test, self.y_test)
        
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, -1)
    
    def test_model_acc_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        model = LinearRegression()
        
        with self.assertRaises(ValueError):
            model_acc(model, None, self.y_train, self.x_test, self.y_test)
        
        with self.assertRaises(ValueError):
            model_acc(model, self.x_train, None, self.x_test, self.y_test)
        
        with self.assertRaises(ValueError):
            model_acc(model, self.x_train, self.y_train, None, self.y_test)
        
        with self.assertRaises(ValueError):
            model_acc(model, self.x_train, self.y_train, self.x_test, None)


class TestPreprocessData(unittest.TestCase):
    """Test cases for preprocess_data function"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'laptop_ID': [1, 2, 3],
            'Company': ['Apple', 'Samsung', 'Dell'],
            'Product': ['MacBook Pro', 'Galaxy Book', 'XPS 13'],
            'Ram': ['8GB', '16GB', '32GB'],
            'Weight': ['1.5kg', '2.1kg', '1.2kg'],
            'ScreenResolution': ['1920x1080 IPS', '1920x1080 Touchscreen', '3840x2160'],
            'Cpu': ['Intel Core i7-8750H 2.2GHz', 'Intel Core i5-8250U 1.6GHz', 'AMD Ryzen 7'],
            'Gpu': ['Intel UHD Graphics 630', 'NVIDIA GTX 1050', 'AMD Radeon'],
            'OpSys': ['macOS', 'Windows 10', 'Linux'],
            'Price_euros': [1200, 800, 1500]
        })
    
    def test_preprocess_data_basic_functionality(self):
        """Test basic preprocessing functionality"""
        result = preprocess_data(self.sample_data)
        
        # Check that RAM and Weight are processed
        self.assertTrue(result['Ram'].dtype in ['int32'])
        self.assertTrue(result['Weight'].dtype in ['float64'])
        
        # Check that new columns are created
        self.assertIn('Touchscreen', result.columns)
        self.assertIn('IPS', result.columns)
        self.assertIn('Cpu_name', result.columns)
        self.assertIn('Gpu_name', result.columns)
    
    def test_preprocess_data_company_mapping(self):
        """Test company mapping in preprocessing"""
        result = preprocess_data(self.sample_data)
        
        # Samsung should be mapped to 'Other'
        self.assertEqual(result.loc[1, 'Company'], 'Other')
        # Apple and Dell should remain unchanged
        self.assertEqual(result.loc[0, 'Company'], 'Apple')
        self.assertEqual(result.loc[2, 'Company'], 'Dell')
    
    def test_preprocess_data_feature_extraction(self):
        """Test feature extraction"""
        result = preprocess_data(self.sample_data)
        
        # Check touchscreen detection
        self.assertEqual(result.loc[1, 'Touchscreen'], 1)  # Galaxy Book has touchscreen
        self.assertEqual(result.loc[0, 'Touchscreen'], 0)  # MacBook doesn't
        
        # Check IPS detection
        self.assertEqual(result.loc[0, 'IPS'], 1)  # MacBook has IPS
        self.assertEqual(result.loc[1, 'IPS'], 0)  # Galaxy Book doesn't mention IPS
    
    def test_preprocess_data_invalid_input(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            preprocess_data("not a dataframe")
        
        with self.assertRaises(ValueError):
            preprocess_data(None)
    
    def test_preprocess_data_original_unchanged(self):
        """Test that original dataframe is not modified"""
        original_data = self.sample_data.copy()
        preprocess_data(self.sample_data)
        
        # Original should remain unchanged
        pd.testing.assert_frame_equal(self.sample_data, original_data)


if __name__ == '__main__':
    unittest.main()
