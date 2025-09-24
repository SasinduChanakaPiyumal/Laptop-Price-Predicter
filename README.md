# Laptop Price Prediction Model - Test Coverage Implementation

## Overview
This project implements a machine learning model for predicting laptop prices based on various features. The original code has been refactored to improve testability and comprehensive unit tests have been added to address test coverage issues.

## Test Coverage Improvements

### Files Added/Modified
1. **`laptop_price_model.py`** - Refactored functions with improved error handling and testability
2. **`test_laptop_price_model.py`** - Comprehensive unit tests covering all main functions
3. **`run_tests.py`** - Test runner script
4. **`Laptop Price model(1).py`** - Updated to use refactored functions

### Functions Under Test
1. **`add_company(inpt)`** - Maps company names to categories
2. **`set_processor(name)`** - Categorizes CPU names
3. **`set_os(inpt)`** - Categorizes operating systems
4. **`model_acc(model, x_train, y_train, x_test, y_test)`** - Evaluates model accuracy
5. **`preprocess_data(dataset)`** - Complete data preprocessing pipeline

### Test Coverage Areas
- **Functionality Tests**: Verify correct categorization and processing
- **Edge Cases**: Handle empty strings, whitespace, case sensitivity
- **Error Handling**: Input validation for None, wrong types, invalid data
- **Data Integrity**: Ensure original data is not modified during processing
- **Integration Testing**: Test complete preprocessing pipeline

### Running Tests
```bash
python run_tests.py
```

Or using unittest directly:
```bash
python -m unittest test_laptop_price_model -v
```

## Key Improvements
1. **Enhanced Error Handling**: Functions now validate inputs and raise appropriate exceptions
2. **Better Testability**: Functions are pure and don't rely on global state
3. **Comprehensive Coverage**: Tests cover normal cases, edge cases, and error conditions
4. **Documentation**: Functions include docstrings explaining parameters and return values
5. **Code Quality**: Refactored code follows better practices and is more maintainable

## Test Statistics
The test suite includes:
- 20+ individual test methods
- Coverage of all main processing functions
- Edge case and error condition testing
- Integration testing for the complete preprocessing pipeline

This addresses the original test coverage issues by providing comprehensive unit tests for all critical functions in the laptop price prediction model.
