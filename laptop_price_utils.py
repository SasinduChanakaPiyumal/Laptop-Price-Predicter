#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for laptop price prediction model.
These functions handle data preprocessing and categorization.
"""

import pandas as pd
import numpy as np


def add_company(inpt):
    """
    Categorize laptop companies, grouping less common brands as 'Other'.
    
    Args:
        inpt (str): Company name
    
    Returns:
        str: Original company name or 'Other' for less common brands
    """
    if inpt in ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']:
        return 'Other'
    else:
        return inpt


def set_processor(name):
    """
    Categorize processor names into standard groups.
    
    Args:
        name (str): Processor name (first 3 words of CPU string)
    
    Returns:
        str: Categorized processor name ('Intel Core i7', 'Intel Core i5', 
             'Intel Core i3', 'AMD', or 'Other')
    """
    if name in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'


def set_os(inpt):
    """
    Categorize operating systems into standard groups.
    
    Args:
        inpt (str): Operating system name
    
    Returns:
        str: Categorized OS name ('Windows', 'Mac', 'Linux', or 'Other')
    """
    if inpt in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inpt in ['macOS', 'Mac OS X']:
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'


def model_acc(model, x_train, x_test, y_train, y_test):
    """
    Train a model and evaluate its accuracy.
    
    Args:
        model: Scikit-learn model instance
        x_train: Training features
        x_test: Test features  
        y_train: Training target
        y_test: Test target
    
    Returns:
        float: Model accuracy score
    """
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


def clean_ram_data(ram_series):
    """
    Clean RAM data by removing 'GB' suffix and converting to integer.
    
    Args:
        ram_series (pd.Series): Series with RAM values like '8GB', '16GB'
    
    Returns:
        pd.Series: Cleaned RAM values as integers
    """
    return ram_series.str.replace('GB', '').astype('int32')


def clean_weight_data(weight_series):
    """
    Clean Weight data by removing 'kg' suffix and converting to float.
    
    Args:
        weight_series (pd.Series): Series with weight values like '1.37kg'
    
    Returns:
        pd.Series: Cleaned weight values as floats
    """
    return weight_series.str.replace('kg', '').astype('float64')


def extract_touchscreen_feature(screen_resolution_series):
    """
    Extract touchscreen feature from screen resolution strings.
    
    Args:
        screen_resolution_series (pd.Series): Series with screen resolution descriptions
    
    Returns:
        pd.Series: Binary series (1 if touchscreen, 0 otherwise)
    """
    return screen_resolution_series.apply(lambda x: 1 if 'Touchscreen' in x else 0)


def extract_ips_feature(screen_resolution_series):
    """
    Extract IPS feature from screen resolution strings.
    
    Args:
        screen_resolution_series (pd.Series): Series with screen resolution descriptions
    
    Returns:
        pd.Series: Binary series (1 if IPS, 0 otherwise)
    """
    return screen_resolution_series.apply(lambda x: 1 if 'IPS' in x else 0)


def extract_cpu_name(cpu_series):
    """
    Extract CPU name (first 3 words) from CPU descriptions.
    
    Args:
        cpu_series (pd.Series): Series with full CPU descriptions
    
    Returns:
        pd.Series: Series with extracted CPU names
    """
    return cpu_series.apply(lambda x: " ".join(x.split()[0:3]))


def extract_gpu_name(gpu_series):
    """
    Extract GPU brand (first word) from GPU descriptions.
    
    Args:
        gpu_series (pd.Series): Series with full GPU descriptions
    
    Returns:
        pd.Series: Series with extracted GPU brands
    """
    return gpu_series.apply(lambda x: " ".join(x.split()[0:1]))
