"""
Configuration file for laptop price prediction model.
Contains all constants, file paths, and hyperparameters.
"""

# File paths
DATA_FILE = "laptop_price.csv"
DATA_ENCODING = "latin-1"
MODEL_OUTPUT_FILE = "predictor.pickle"

# Data processing constants
RAM_UNIT = "GB"
WEIGHT_UNIT = "kg"
DEFAULT_RESOLUTION = (1366, 768)

# Companies to group as 'Other'
COMPANIES_TO_GROUP = [
    'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
    'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'
]

# Primary Intel CPU models to keep
PRIMARY_INTEL_CPUS = ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']

# GPU brands to exclude
GPU_BRANDS_TO_EXCLUDE = ['ARM']

# Operating systems mapping
OS_WINDOWS_VARIANTS = ['Windows 10', 'Windows 7', 'Windows 10 S']
OS_MAC_VARIANTS = ['macOS', 'Mac OS X']

# Columns to drop during preprocessing
COLUMNS_TO_DROP = ['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu']

# Feature engineering
KEY_NUMERIC_FEATURES = ['Ram', 'Weight', 'Inches', 'Touchscreen', 'IPS', 
                        'Screen_Width', 'Screen_Height', 'Total_Pixels', 'PPI']
POLYNOMIAL_FEATURES = ['Ram', 'Weight', 'Total_Pixels', 'PPI']

# Model training parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter tuning
N_ITER_RANDOM_SEARCH = 50

# Random Forest parameters for tuning
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Gradient Boosting parameters for tuning
GB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Target variable
TARGET_VARIABLE = 'Price_euros'
