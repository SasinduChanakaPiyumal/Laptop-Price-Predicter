"""
Data loading and initial exploration module.
Handles reading the dataset and performing initial data quality checks.
"""

import pandas as pd
import config


def load_data(file_path=config.DATA_FILE, encoding=config.DATA_ENCODING):
    """
    Load the laptop price dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding to use
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        dataset = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def explore_data(dataset, verbose=True):
    """
    Perform initial data exploration and quality checks.
    
    Args:
        dataset (pd.DataFrame): The dataset to explore
        verbose (bool): Whether to print exploration results
        
    Returns:
        dict: Dictionary containing exploration statistics
    """
    exploration_stats = {
        'shape': dataset.shape,
        'columns': list(dataset.columns),
        'dtypes': dataset.dtypes.to_dict(),
        'missing_values': dataset.isnull().sum().to_dict(),
        'total_missing': dataset.isnull().sum().sum()
    }
    
    if verbose:
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        print(f"\nDataset Shape: {exploration_stats['shape']}")
        print(f"\nColumns: {exploration_stats['columns']}")
        print(f"\nData Types:\n{dataset.dtypes}")
        print(f"\nMissing Values:\n{dataset.isnull().sum()}")
        print(f"\nBasic Statistics:\n{dataset.describe()}")
        print(f"\nDataset Info:")
        dataset.info()
        
    return exploration_stats


def analyze_correlations(dataset, target_column=config.TARGET_VARIABLE):
    """
    Analyze correlations between numeric features and target variable.
    
    Args:
        dataset (pd.DataFrame): The dataset
        target_column (str): Name of the target variable
        
    Returns:
        pd.Series: Correlation values with target variable
    """
    non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
    numeric_dataset = dataset.drop(columns=non_numeric_columns)
    
    if target_column in numeric_dataset.columns:
        correlation = numeric_dataset.corr()[target_column].sort_values(ascending=False)
        print("\n" + "="*60)
        print(f"CORRELATION WITH {target_column}")
        print("="*60)
        print(correlation)
        return correlation
    else:
        print(f"Warning: Target column '{target_column}' not found in numeric columns")
        return None


if __name__ == "__main__":
    # Test the data loading functionality
    dataset = load_data()
    explore_data(dataset)
    analyze_correlations(dataset)
