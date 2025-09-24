"""
Laptop Price Prediction Model - Refactored functions for better testability
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle


def add_company(inpt):
    """
    Categorize laptop companies into groups, mapping less common brands to 'Other'
    
    Args:
        inpt (str): Company name
        
    Returns:
        str: Either the original company name or 'Other' for less common brands
    """
    if not isinstance(inpt, str):
        raise ValueError("Input must be a string")
    
    other_companies = {
        'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
        'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'
    }
    
    return 'Other' if inpt in other_companies else inpt


def set_processor(name):
    """
    Categorize CPU names into standard groups
    
    Args:
        name (str): CPU name string
        
    Returns:
        str: Categorized processor name
    """
    if not isinstance(name, str):
        raise ValueError("Input must be a string")
        
    if not name.strip():
        return 'Other'
    
    intel_processors = {'Intel Core i7', 'Intel Core i5', 'Intel Core i3'}
    
    if name in intel_processors:
        return name
    elif name.split()[0] == 'AMD':
        return 'AMD'
    else:
        return 'Other'


def set_os(inpt):
    """
    Categorize operating systems into standard groups
    
    Args:
        inpt (str): Operating system name
        
    Returns:
        str: Categorized OS name
    """
    if not isinstance(inpt, str):
        raise ValueError("Input must be a string")
        
    if not inpt.strip():
        return 'Other'
    
    if inpt in {'Windows 10', 'Windows 7', 'Windows 10 S'}:
        return 'Windows'
    elif inpt in {'macOS', 'Mac OS X'}:
        return 'Mac'
    elif inpt == 'Linux':
        return 'Linux'
    else:
        return 'Other'


def model_acc(model, x_train, y_train, x_test, y_test):
    """
    Train a model and evaluate its accuracy
    
    Args:
        model: sklearn model instance
        x_train: Training features
        y_train: Training target
        x_test: Test features  
        y_test: Test target
        
    Returns:
        float: Model accuracy score
    """
    if x_train is None or y_train is None or x_test is None or y_test is None:
        raise ValueError("Training and test data cannot be None")
        
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


def preprocess_data(dataset):
    """
    Apply all preprocessing steps to the dataset
    
    Args:
        dataset (pd.DataFrame): Raw laptop dataset
        
    Returns:
        pd.DataFrame: Processed dataset ready for modeling
    """
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
        
    # Create a copy to avoid modifying original data
    df = dataset.copy()
    
    # Clean RAM column
    if 'Ram' in df.columns:
        df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    
    # Clean Weight column
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].str.replace('kg', '').astype('float64')
    
    # Apply company categorization
    if 'Company' in df.columns:
        df['Company'] = df['Company'].apply(add_company)
    
    # Extract touchscreen and IPS features
    if 'ScreenResolution' in df.columns:
        df['Touchscreen'] = df['ScreenResolution'].apply(
            lambda x: 1 if 'Touchscreen' in x else 0
        )
        df['IPS'] = df['ScreenResolution'].apply(
            lambda x: 1 if 'IPS' in x else 0
        )
    
    # Process CPU information
    if 'Cpu' in df.columns:
        df['Cpu_name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
        df['Cpu_name'] = df['Cpu_name'].apply(set_processor)
    
    # Process GPU information
    if 'Gpu' in df.columns:
        df['Gpu_name'] = df['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
        # Remove ARM entries
        df = df[df['Gpu_name'] != 'ARM']
    
    # Process operating system
    if 'OpSys' in df.columns:
        df['OpSys'] = df['OpSys'].apply(set_os)
    
    return df
