import numpy as np
import pandas as pd
from typing import Tuple


def optimized_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vectorized, memory-friendly preprocessing for the laptop price dataset.

    Steps:
    - Parse Ram and Weight with minimal dtypes
    - Create Touchscreen and IPS flags via str.contains (vectorized)
    - Normalize Cpu_name into {"Intel Core i7", "Intel Core i5", "Intel Core i3", "AMD", "Other"}
    - Extract Gpu_name (first vendor token) and drop ARM rows
    - Consolidate OpSys into {"Windows", "Mac", "Linux", "Other"}
    - Consolidate rare Company values into "Other"
    - Drop unused columns
    - One-hot encode categoricals with compact dtype
    Returns X (features) and y (target)
    """
    df = df.copy()

    # Parse numeric columns with smaller dtypes
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int16')
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

    # Flags from screen resolution
    sr = df['ScreenResolution'].astype(str)
    df['Touchscreen'] = sr.str.contains('Touchscreen', na=False).astype('uint8')
    df['IPS'] = sr.str.contains('IPS', na=False).astype('uint8')

    # CPU name normalization
    cpu_name = df['Cpu'].astype(str).str.split().str[:3].str.join(' ')
    intel_set = {"Intel Core i7", "Intel Core i5", "Intel Core i3"}
    is_intel_family = cpu_name.isin(intel_set)
    is_amd = cpu_name.str.startswith('AMD')
    df['Cpu_name'] = np.where(is_intel_family, cpu_name,
                              np.where(is_amd, 'AMD', 'Other'))

    # GPU vendor
    df['Gpu_name'] = df['Gpu'].astype(str).str.split().str[0]
    df = df[df['Gpu_name'] != 'ARM']

    # OS consolidation
    ops = df['OpSys'].astype(str)
    df['OpSys'] = np.select(
        [ops.isin(['Windows 10', 'Windows 7', 'Windows 10 S']),
         ops.isin(['macOS', 'Mac OS X']),
         ops.eq('Linux')],
        ['Windows', 'Mac', 'Linux'],
        default='Other'
    )

    # Company consolidation
    rare_companies = {
        'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero',
        'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'
    }
    df['Company'] = np.where(df['Company'].isin(rare_companies), 'Other', df['Company'])

    # Drop unused columns
    df = df.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])

    # One-hot encode categoricals with compact dtype
    df = pd.get_dummies(df, dtype=np.uint8)

    # Split features/target
    y = df['Price_euros']
    X = df.drop(columns=['Price_euros'])
    return X, y


def train_fast_random_forest(X: pd.DataFrame, y: pd.Series):
    """Train a fast, reproducible RandomForestRegressor."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        criterion='squared_error'
    )
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    return rf, score
