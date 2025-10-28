"""
Laptop Price Model module.

This module refactors the original Jupyter notebook style script
"Laptop Price model(1).py" into an importable Python module that exposes
reusable data preparation utilities and model training helpers, while ensuring
that importing this module does not trigger model training. Running the module
as a script executes the end-to-end pipeline.

Key exported utilities:
- add_company
- extract_resolution
- set_processor
- set_os
- extract_storage_features
- model_acc
- load_and_prepare_data
- split_data
- train_model
- save_model

Notes:
- Optional dependencies like xgboost and lightgbm are imported lazily with
  try/except so importing this module will not fail if they are missing.
- The original file "Laptop Price model(1).py" is left untouched for reference.
"""

# Standard library imports
from __future__ import annotations
import os
import pickle
import re
import tempfile
from typing import Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Scikit-learn core imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Optional third-party libs (handled lazily in code paths)
try:  # lightgbm is optional
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:  # xgboost is optional
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


# =========================
# Utility functions (verbatim logic from original script)
# =========================

def add_company(inpt: str) -> str:
    """Consolidate rare company names under 'Other'.

    This preserves the exact branching logic from the original script.

    Parameters
    - inpt: str - Original company name

    Returns
    - str: Consolidated company name

    Examples
    >>> add_company('Samsung')
    'Other'
    >>> add_company('Lenovo')
    'Lenovo'
    """
    if inpt == 'Samsung'or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt


def extract_resolution(res_string: Optional[str]) -> Tuple[int, int, int]:
    """Parse resolution string to width, height, total pixels.

    This reproduces the robust parsing and defaulting behavior from the
    original notebook.

    Parameters
    - res_string: str | None - String containing values like "1920x1080".

    Returns
    - (width, height, total_pixels)

    Examples
    >>> extract_resolution('1920x1080 IPS')
    (1920, 1080, 2073600)
    >>> extract_resolution(None)
    (1366, 768, 1049088)
    """
    # Default resolution values
    DEFAULT_WIDTH = 1366
    DEFAULT_HEIGHT = 768
    DEFAULT_PIXELS = DEFAULT_WIDTH * DEFAULT_HEIGHT

    # Handle None or invalid input
    if res_string is None or not isinstance(res_string, str):
        # keep side-effectful prints consistent with original
        print(f"WARNING: Invalid resolution input (None or non-string): {type(res_string)}")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS

    # Handle empty string
    if not res_string.strip():
        print("WARNING: Empty resolution string provided")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS

    try:
        # Find pattern like "1920x1080" or "3840x2160"
        match = re.search(r'(\d+)x(\d+)', res_string)

        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))

                # Validate reasonable resolution values
                if width < 640 or width > 7680 or height < 480 or height > 4320:
                    print(f"WARNING: Unusual resolution detected: {width}x{height}. Using default.")
                    return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS

                return width, height, width * height
            except (ValueError, AttributeError) as e:
                print(f"WARNING: Could not convert resolution values to integers in '{res_string}': {e}")
                return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS
        else:
            # No match found, use default silently (common case for non-standard formats)
            return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS

    except re.error as e:  # pragma: no cover - defensive
        print(f"ERROR: Regex error in extract_resolution for '{res_string}': {e}")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS
    except Exception as e:  # pragma: no cover - defensive
        print(f"ERROR: Unexpected error in extract_resolution for '{res_string}': {e}")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PIXELS


def set_processor(name: str) -> str:
    """Categorize CPU names following original mapping.

    Parameters
    - name: str - CPU name prefix (e.g., 'Intel Core i7')

    Returns
    - str: One of 'Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD', 'Other'
    """
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'


def set_os(inpt: str) -> str:
    """Map operating system strings to coarse categories.

    Parameters
    - inpt: str - Original OS string

    Returns
    - str: 'Windows', 'Mac', 'Linux', or 'Other'
    """
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'


def extract_storage_features(memory_string: Optional[str]):
    """Extract storage type flags and total capacity in GB from memory string.

    This preserves the detection of SSD/HDD/Flash/Hybrid and capacity summing in
    GB and TB from the original script.

    Parameters
    - memory_string: str | None

    Returns
    - tuple: (has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb)

    Examples
    >>> extract_storage_features('256GB SSD + 1TB HDD')
    (1, 1, 0, 0, 1280.0)
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


# =========================
# Pipeline helpers
# =========================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load CSV and perform feature engineering to return encoded DataFrame.

    This function mirrors the original pipeline: type cleaning, feature
    extraction, dummy encoding, and creation of interaction features. The
    returned DataFrame contains the target column 'Price_euros'.

    Parameters
    - filepath: str - Path to the CSV (e.g., 'laptop_price.csv')

    Returns
    - pandas.DataFrame: Fully processed dataset ready for modeling
    """
    # Load CSV with robust encoding fallbacks similar to the original
    try:
        dataset = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            dataset = pd.read_csv(filepath, encoding='utf-8')
        except Exception:
            dataset = pd.read_csv(filepath, encoding='iso-8859-1')

    # Ram to int
    dataset['Ram'] = dataset['Ram'].str.replace('GB', '').astype('int32')

    # Weight conversion (replicates original logic but minimal verbosity)
    def convert_weight_to_kg(weight_str):
        if pd.isnull(weight_str):
            return np.nan
        weight_str = str(weight_str).strip().lower()
        try:
            if 'kg' in weight_str:
                return float(weight_str.replace('kg', '').strip())
            elif 'g' in weight_str and 'kg' not in weight_str:
                return float(weight_str.replace('g', '').strip()) / 1000.0
            elif 'lb' in weight_str or 'lbs' in weight_str:
                weight_val = weight_str.replace('lbs', '').replace('lb', '').strip()
                return float(weight_val) * 0.453592
            else:
                return float(weight_str)
        except (ValueError, AttributeError):
            return np.nan

    dataset['Weight'] = dataset['Weight'].apply(convert_weight_to_kg)

    # Company consolidation
    dataset['Company'] = dataset['Company'].apply(add_company)

    # Screen features
    dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    dataset['Screen_Width'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[0])
    dataset['Screen_Height'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[1])
    dataset['Total_Pixels'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[2])
    dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']

    # CPU name engineering
    dataset['Cpu_name'] = dataset['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)

    # GPU name prefix and filter ARM (as original)
    dataset['Gpu_name'] = dataset['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
    dataset = dataset[dataset['Gpu_name'] != 'ARM']

    # OS mapping
    dataset['OpSys'] = dataset['OpSys'].apply(set_os)

    # Storage features
    storage_features = dataset['Memory'].apply(extract_storage_features)
    dataset['Has_SSD'] = storage_features.apply(lambda x: x[0])
    dataset['Has_HDD'] = storage_features.apply(lambda x: x[1])
    dataset['Has_Flash'] = storage_features.apply(lambda x: x[2])
    dataset['Has_Hybrid'] = storage_features.apply(lambda x: x[3])
    dataset['Storage_Capacity_GB'] = storage_features.apply(lambda x: x[4])
    dataset['Storage_Type_Score'] = (
        dataset['Has_SSD'] * 3 +
        dataset['Has_Flash'] * 2.5 +
        dataset['Has_Hybrid'] * 2 +
        dataset['Has_HDD'] * 1
    )

    # Drop redundant columns
    dataset = dataset.drop(columns=['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory'])

    # One-hot encode
    dataset = pd.get_dummies(dataset)

    # Split features/target to add interactions
    x = dataset.drop('Price_euros', axis=1)
    y = dataset['Price_euros']

    # Add selected engineered interaction features (as in original)
    if 'Ram' in x.columns:
        x['Ram_squared'] = x['Ram'] ** 2
    if 'Total_Pixels' in x.columns and 'Inches' in x.columns:
        x['Screen_Quality'] = x['Total_Pixels'] / 1000000 * x['Inches']
    if 'Storage_Capacity_GB' in x.columns and 'Has_SSD' in x.columns:
        x['Premium_Storage'] = x['Storage_Capacity_GB'] * (x['Has_SSD'] + 1) / 1000
    if 'Ram' in x.columns and 'Storage_Type_Score' in x.columns:
        x['RAM_Storage_Quality'] = x['Ram'] * x['Storage_Type_Score']
    if 'PPI' in x.columns and 'Storage_Type_Score' in x.columns:
        x['Display_Storage_Premium'] = x['PPI'] * x['Storage_Type_Score']
    if 'Weight' in x.columns and 'Inches' in x.columns:
        x['Weight_Size_Ratio'] = x['Weight'] / x['Inches']
    if 'Total_Pixels' in x.columns and 'Ram' in x.columns:
        x['Pixels_Per_RAM'] = x['Total_Pixels'] / (x['Ram'] * 1000000)
    if 'Storage_Capacity_GB' in x.columns and 'Inches' in x.columns:
        x['Storage_Per_Inch'] = x['Storage_Capacity_GB'] / x['Inches']

    # Recombine
    processed = pd.concat([x, y], axis=1)
    return processed


def split_data(dataframe: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """Split processed dataframe into train/test parts.

    Parameters
    - dataframe: pd.DataFrame - Output of load_and_prepare_data (contains Price_euros)
    - test_size: float - Fraction for test split
    - random_state: int - RNG seed

    Returns
    - x_train, x_test, y_train, y_test
    """
    x = dataframe.drop('Price_euros', axis=1)
    y = dataframe['Price_euros']
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'xgboost'):
    """Train a model of the specified type and return the fitted estimator.

    Supported model_type values: 'linear', 'ridge', 'lasso', 'elasticnet',
    'random_forest', 'gbrt', 'lightgbm', 'xgboost'. Defaults to 'xgboost' if
    available, otherwise falls back to RandomForest.

    Parameters
    - x_train: pd.DataFrame
    - y_train: pd.Series
    - model_type: str

    Returns
    - Fitted estimator
    """
    model_type = (model_type or '').lower()

    if model_type in ('linear', 'ols'):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(x_train)
        model = LinearRegression()
        return model.fit(Xs, y_train)

    if model_type == 'ridge':
        scaler = StandardScaler()
        Xs = scaler.fit_transform(x_train)
        model = Ridge(alpha=1.0, random_state=42)
        return model.fit(Xs, y_train)

    if model_type == 'lasso':
        scaler = StandardScaler()
        Xs = scaler.fit_transform(x_train)
        model = Lasso(alpha=1.0, random_state=42)
        return model.fit(Xs, y_train)

    if model_type == 'elasticnet':
        scaler = StandardScaler()
        Xs = scaler.fit_transform(x_train)
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        return model.fit(Xs, y_train)

    if model_type in ('gbrt', 'gb', 'gradient_boosting'):
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        return model.fit(x_train, y_train)

    if model_type in ('lightgbm', 'lgb') and lgb is not None:
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        return model.fit(x_train, y_train)

    if model_type in ('xgboost', 'xgb') and xgb is not None:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
        return model.fit(x_train, y_train)

    # Default fallback to RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model.fit(x_train, y_train)


def model_acc(model, model_name: str = 'Model', use_scaled: bool = False,
              x_train: Optional[pd.DataFrame] = None,
              x_test: Optional[pd.DataFrame] = None,
              y_train: Optional[pd.Series] = None,
              y_test: Optional[pd.Series] = None):
    """Evaluate a model with multiple metrics.

    This function generalizes the original `model_acc` by allowing explicit
    datasets to be passed. If scaled data is requested, it will scale x_train
    and x_test internally. If explicit data is not provided, this function
    raises a ValueError to avoid reliance on globals.

    Parameters
    - model: Estimator implementing fit/predict
    - model_name: str
    - use_scaled: bool - Whether to scale features before fitting
    - x_train, x_test, y_train, y_test: Datasets

    Returns
    - (r2, mae, rmse, cv_mean)
    """
    if any(v is None for v in (x_train, x_test, y_train, y_test)):
        raise ValueError("Please pass x_train, x_test, y_train, y_test to model_acc().")

    X_train = x_train
    X_test = x_test

    if use_scaled:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        X_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} euros")
    print(f"  RMSE: {rmse:.2f} euros")
    print(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")

    return r2, mae, rmse, cv_mean


def save_model(model, filepath: str) -> None:
    """Serialize model to filepath using pickle with robust error handling.

    Parameters
    - model: Trained estimator
    - filepath: str - Destination path
    """
    temp_filename = None
    try:
        # Create temporary file first
        fd, temp_filename = tempfile.mkstemp(suffix='.pickle', dir=os.path.dirname(filepath) or '.')
        with os.fdopen(fd, 'wb') as temp_file:
            pickle.dump(model, temp_file)

        # Replace target file atomically
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except PermissionError:
                print(f"ERROR: Cannot remove existing file '{filepath}'. Permission denied.")
                raise
        os.rename(temp_filename, filepath)
        temp_filename = None
        print(f"Model successfully saved to '{filepath}'")
    except PermissionError:
        print(f"ERROR: Permission denied when trying to write to '{filepath}'.")
        raise
    except (IOError, OSError) as e:
        print(f"ERROR: I/O/OS error while writing model file: {e}")
        raise
    except pickle.PicklingError as e:
        print(f"ERROR: Failed to pickle the model: {e}")
        raise
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


if __name__ == '__main__':
    # Execute end-to-end training pipeline when run as a script.
    csv_path = 'laptop_price.csv'
    print('Loading and preparing data...')
    data = load_and_prepare_data(csv_path)

    print('Splitting data...')
    X_train, X_test, Y_train, Y_test = split_data(data, test_size=0.25, random_state=42)

    # Choose a sensible default model (prefer XGBoost if available)
    default_type = 'xgboost' if xgb is not None else 'random_forest'
    print(f'Training model ({default_type})...')
    model = train_model(X_train, Y_train, model_type=default_type)

    print('Evaluating model...')
    model_acc(model, model_name=f'Default {default_type.title()}', use_scaled=False,
              x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test)

    # Save model
    print('Saving model to predictor.pickle...')
    save_model(model, 'predictor.pickle')
