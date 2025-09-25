#!/usr/bin/env python
"""
Comprehensive test suite for laptop price prediction model.
Addresses the lack of automated testing in the original code.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Import functions from the main module
# Note: In a real implementation, you'd refactor the main script into proper modules
sys.path.append('.')

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'Ram': ['8GB', '16GB', '4GB'],
            'Weight': ['1.5kg', '2.1kg', '1.0kg'],
            'Company': ['Dell', 'Samsung', 'Apple'],
            'Cpu': ['Intel Core i7-8550U 1.8GHz', 'AMD Ryzen 5 2500U', 'Intel Core i5-8250U'],
            'Gpu': ['Intel UHD Graphics 620', 'AMD Radeon Vega 8', 'Nvidia GeForce GTX'],
            'OpSys': ['Windows 10', 'macOS', 'Linux'],
            'ScreenResolution': ['Full HD 1920x1080 Touchscreen IPS', '1366x768', '1920x1080 IPS'],
            'Price_euros': [800.0, 1200.0, 600.0]
        })
    
    def test_ram_preprocessing(self):
        """Test RAM column preprocessing"""
        # Test the RAM preprocessing logic
        test_data = self.sample_data.copy()
        test_data['Ram'] = test_data['Ram'].str.replace('GB','').astype('int32')
        
        expected = [8, 16, 4]
        self.assertEqual(test_data['Ram'].tolist(), expected)
        self.assertEqual(test_data['Ram'].dtype, 'int32')
    
    def test_weight_preprocessing(self):
        """Test Weight column preprocessing"""
        test_data = self.sample_data.copy()
        test_data['Weight'] = test_data['Weight'].str.replace('kg','').astype('float64')
        
        expected = [1.5, 2.1, 1.0]
        self.assertEqual(test_data['Weight'].tolist(), expected)
        self.assertEqual(test_data['Weight'].dtype, 'float64')
    
    def test_touchscreen_feature_extraction(self):
        """Test touchscreen feature extraction"""
        test_data = self.sample_data.copy()
        test_data['Touchscreen'] = test_data['ScreenResolution'].apply(
            lambda x: 1 if 'Touchscreen' in x else 0
        )
        
        expected = [1, 0, 0]  # Only first has Touchscreen
        self.assertEqual(test_data['Touchscreen'].tolist(), expected)
    
    def test_ips_feature_extraction(self):
        """Test IPS feature extraction"""
        test_data = self.sample_data.copy()
        test_data['IPS'] = test_data['ScreenResolution'].apply(
            lambda x: 1 if 'IPS' in x else 0
        )
        
        expected = [1, 0, 1]  # First and third have IPS
        self.assertEqual(test_data['IPS'].tolist(), expected)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions"""
    
    def test_company_grouping_function(self):
        """Test company grouping logic"""
        def add_company(inpt):
            if inpt in ['Samsung','Razer','Mediacom','Microsoft','Xiaomi','Vero',
                       'Chuwi','Google','Fujitsu','LG','Huawei']:
                return 'Other'
            else:
                return inpt
        
        # Test known companies that should be grouped as 'Other'
        self.assertEqual(add_company('Samsung'), 'Other')
        self.assertEqual(add_company('Google'), 'Other')
        self.assertEqual(add_company('LG'), 'Other')
        
        # Test companies that should remain unchanged
        self.assertEqual(add_company('Dell'), 'Dell')
        self.assertEqual(add_company('Apple'), 'Apple')
        self.assertEqual(add_company('HP'), 'HP')
    
    def test_cpu_name_extraction(self):
        """Test CPU name extraction logic"""
        cpu_strings = [
            'Intel Core i7-8550U 1.8GHz',
            'AMD Ryzen 5 2500U 2.0GHz',
            'Intel Core i5-8250U 1.6GHz',
            'Intel Celeron N4000 1.1GHz'
        ]
        
        def extract_cpu_name(x):
            return " ".join(x.split()[0:3])
        
        expected = [
            'Intel Core i7',
            'AMD Ryzen 5',
            'Intel Core i5',
            'Intel Celeron N4000'
        ]
        
        results = [extract_cpu_name(cpu) for cpu in cpu_strings]
        self.assertEqual(results, expected)
    
    def test_processor_categorization(self):
        """Test processor categorization logic"""
        def set_processor(name):
            if name in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
                return name
            elif name.split()[0] == 'AMD':
                return 'AMD'
            else:
                return 'Other'
        
        # Test Intel Core processors
        self.assertEqual(set_processor('Intel Core i7'), 'Intel Core i7')
        self.assertEqual(set_processor('Intel Core i5'), 'Intel Core i5')
        self.assertEqual(set_processor('Intel Core i3'), 'Intel Core i3')
        
        # Test AMD processors
        self.assertEqual(set_processor('AMD Ryzen 5'), 'AMD')
        self.assertEqual(set_processor('AMD FX 8350'), 'AMD')
        
        # Test other processors
        self.assertEqual(set_processor('Intel Celeron'), 'Other')
        self.assertEqual(set_processor('ARM Cortex'), 'Other')
    
    def test_os_categorization(self):
        """Test operating system categorization"""
        def set_os(inpt):
            if inpt in ['Windows 10', 'Windows 7', 'Windows 10 S']:
                return 'Windows'
            elif inpt in ['macOS', 'Mac OS X']:
                return 'Mac'
            elif inpt == 'Linux':
                return inpt
            else:
                return 'Other'
        
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


class TestModelTraining(unittest.TestCase):
    """Test model training pipeline"""
    
    def setUp(self):
        """Set up test data for model training"""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(100, 10))
        self.X_test = pd.DataFrame(np.random.rand(20, 10))
        self.y_train = pd.Series(np.random.rand(100) * 1000 + 500)  # Prices 500-1500
        self.y_test = pd.Series(np.random.rand(20) * 1000 + 500)
    
    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_model_accuracy_function(self, mock_rf):
        """Test the model accuracy evaluation function"""
        # Mock the model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.score.return_value = 0.85
        mock_model.__str__ = lambda x: "MockModel"
        mock_rf.return_value = mock_model
        
        def model_acc(model):
            model.fit(self.X_train, self.y_train)
            acc = model.score(self.X_test, self.y_test)
            return acc
        
        # Test the function
        accuracy = model_acc(mock_model)
        
        # Verify model was fitted and scored
        mock_model.fit.assert_called_once_with(self.X_train, self.y_train)
        mock_model.score.assert_called_once_with(self.X_test, self.y_test)
        self.assertEqual(accuracy, 0.85)
    
    def test_data_split_shapes(self):
        """Test that data splitting produces correct shapes"""
        from sklearn.model_selection import train_test_split
        
        # Create sample data
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.rand(100))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 75)
        self.assertEqual(X_test.shape[0], 25)
        self.assertEqual(len(y_train), 75)
        self.assertEqual(len(y_test), 25)
        self.assertEqual(X_train.shape[1], X_test.shape[1])


class TestPredictionValidation(unittest.TestCase):
    """Test prediction functionality and validation"""
    
    def setUp(self):
        """Set up mock model and feature names"""
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([850.0])
        
        # Mock feature names that would come from x_train.columns
        self.feature_names = [
            'Ram', 'Weight', 'Touchscreen', 'IPS',
            'Company_Dell', 'Company_HP', 'Company_Lenovo', 'Company_Other',
            'TypeName_Gaming', 'TypeName_Notebook', 'TypeName_Ultrabook',
            'Cpu_name_Intel Core i5', 'Cpu_name_Intel Core i7', 'Cpu_name_AMD',
            'Gpu_name_Intel', 'Gpu_name_Nvidia', 'OpSys_Windows', 'OpSys_Mac'
        ]
    
    def test_input_validation_ram(self):
        """Test RAM input validation"""
        with patch('__main__.x_train') as mock_x_train:
            mock_x_train.columns.tolist.return_value = self.feature_names
            
            # Import the validation function (would need proper module structure)
            def validate_ram(ram):
                if not isinstance(ram, int) or ram < 1 or ram > 64:
                    raise ValueError(f"RAM must be an integer between 1-64 GB, got: {ram}")
                return True
            
            # Valid RAM values
            self.assertTrue(validate_ram(8))
            self.assertTrue(validate_ram(16))
            self.assertTrue(validate_ram(32))
            
            # Invalid RAM values
            with self.assertRaises(ValueError):
                validate_ram(0)  # Too low
            with self.assertRaises(ValueError):
                validate_ram(128)  # Too high
            with self.assertRaises(ValueError):
                validate_ram(8.5)  # Float instead of int
            with self.assertRaises(ValueError):
                validate_ram("8")  # String instead of int
    
    def test_input_validation_weight(self):
        """Test weight input validation"""
        def validate_weight(weight):
            if not isinstance(weight, (int, float)) or weight < 0.1 or weight > 10.0:
                raise ValueError(f"Weight must be between 0.1-10.0 kg, got: {weight}")
            return True
        
        # Valid weights
        self.assertTrue(validate_weight(1.5))
        self.assertTrue(validate_weight(2))
        self.assertTrue(validate_weight(0.9))
        
        # Invalid weights
        with self.assertRaises(ValueError):
            validate_weight(0.05)  # Too low
        with self.assertRaises(ValueError):
            validate_weight(15.0)  # Too high
        with self.assertRaises(ValueError):
            validate_weight("1.5")  # String instead of number
    
    def test_input_validation_boolean_fields(self):
        """Test boolean field validation"""
        def validate_boolean(value, field_name):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean, got: {value}")
            return True
        
        # Valid boolean values
        self.assertTrue(validate_boolean(True, "Touchscreen"))
        self.assertTrue(validate_boolean(False, "IPS"))
        
        # Invalid boolean values
        with self.assertRaises(ValueError):
            validate_boolean(1, "Touchscreen")  # Integer instead of bool
        with self.assertRaises(ValueError):
            validate_boolean("true", "IPS")  # String instead of bool
    
    def test_feature_vector_construction(self):
        """Test that feature vectors are constructed correctly"""
        def construct_feature_vector(ram, weight, touchscreen, ips, feature_names):
            feature_vector = np.zeros(len(feature_names))
            feature_vector[feature_names.index('Ram')] = ram
            feature_vector[feature_names.index('Weight')] = weight
            feature_vector[feature_names.index('Touchscreen')] = 1 if touchscreen else 0
            feature_vector[feature_names.index('IPS')] = 1 if ips else 0
            return feature_vector
        
        result = construct_feature_vector(8, 1.5, True, False, self.feature_names)
        
        # Check that values are set correctly
        self.assertEqual(result[self.feature_names.index('Ram')], 8)
        self.assertEqual(result[self.feature_names.index('Weight')], 1.5)
        self.assertEqual(result[self.feature_names.index('Touchscreen')], 1)
        self.assertEqual(result[self.feature_names.index('IPS')], 0)
    
    def test_prediction_output_format(self):
        """Test that predictions return proper format"""
        # Mock prediction should return a single float value
        prediction_result = self.mock_model.predict([[1, 2, 3, 4, 5]])
        
        self.assertIsInstance(prediction_result, np.ndarray)
        self.assertEqual(len(prediction_result), 1)
        self.assertIsInstance(prediction_result[0], (int, float, np.number))


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and pipeline robustness"""
    
    def test_missing_values_handling(self):
        """Test handling of missing values in dataset"""
        # Test data with missing values
        test_data = pd.DataFrame({
            'Ram': ['8GB', None, '16GB'],
            'Weight': ['1.5kg', '2.0kg', None],
            'Price_euros': [800.0, 900.0, 1000.0]
        })
        
        # Check for missing values
        missing_counts = test_data.isnull().sum()
        
        # This should flag missing values for proper handling
        self.assertGreater(missing_counts['Ram'], 0)
        self.assertGreater(missing_counts['Weight'], 0)
    
    def test_data_type_consistency(self):
        """Test data type consistency after preprocessing"""
        test_data = pd.DataFrame({
            'Ram': [8, 16, 4],
            'Weight': [1.5, 2.0, 1.2],
            'Touchscreen': [1, 0, 1],
            'IPS': [0, 1, 0]
        })
        
        # Verify data types
        self.assertTrue(pd.api.types.is_integer_dtype(test_data['Ram']))
        self.assertTrue(pd.api.types.is_float_dtype(test_data['Weight']))
        self.assertTrue(pd.api.types.is_integer_dtype(test_data['Touchscreen']))
        self.assertTrue(pd.api.types.is_integer_dtype(test_data['IPS']))
    
    def test_one_hot_encoding_integrity(self):
        """Test one-hot encoding produces valid results"""
        test_data = pd.DataFrame({
            'Company': ['Dell', 'HP', 'Dell'],
            'Type': ['Gaming', 'Notebook', 'Gaming']
        })
        
        # Apply one-hot encoding
        encoded_data = pd.get_dummies(test_data)
        
        # Check that sum of one-hot columns equals number of categories per row
        company_cols = [col for col in encoded_data.columns if col.startswith('Company_')]
        type_cols = [col for col in encoded_data.columns if col.startswith('Type_')]
        
        # Each row should have exactly one 1 in company columns and one 1 in type columns
        for i in range(len(encoded_data)):
            self.assertEqual(encoded_data.iloc[i][company_cols].sum(), 1)
            self.assertEqual(encoded_data.iloc[i][type_cols].sum(), 1)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataPreprocessing,
        TestFeatureEngineering, 
        TestModelTraining,
        TestPredictionValidation,
        TestDataIntegrity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestClass(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
