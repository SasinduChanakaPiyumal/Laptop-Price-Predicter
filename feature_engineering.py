"""
Feature engineering module.
Handles data preprocessing, cleaning, and feature extraction.
"""

import pandas as pd
import numpy as np
import re
import config


def clean_ram_column(dataset):
    """
    Clean and convert RAM column to numeric format.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned RAM column
    """
    dataset['Ram'] = dataset['Ram'].str.replace(config.RAM_UNIT, '').astype('int32')
    print("✓ RAM column cleaned and converted to int32")
    return dataset


def clean_weight_column(dataset):
    """
    Clean and convert Weight column to numeric format.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned Weight column
    """
    dataset['Weight'] = dataset['Weight'].str.replace(config.WEIGHT_UNIT, '').astype('float64')
    print("✓ Weight column cleaned and converted to float64")
    return dataset


def group_companies(dataset):
    """
    Group less common companies into 'Other' category.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with grouped companies
    """
    def categorize_company(company):
        return 'Other' if company in config.COMPANIES_TO_GROUP else company
    
    dataset['Company'] = dataset['Company'].apply(categorize_company)
    print(f"✓ Companies grouped. Unique companies: {dataset['Company'].nunique()}")
    return dataset


def extract_cpu_features(dataset):
    """
    Extract and categorize CPU features.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with CPU name extracted and categorized
    """
    # Extract first 3 words of CPU name
    dataset['Cpu_name'] = dataset['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    
    # Categorize CPU
    def categorize_processor(name):
        if name in config.PRIMARY_INTEL_CPUS:
            return name
        elif name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
    
    dataset['Cpu_name'] = dataset['Cpu_name'].apply(categorize_processor)
    print(f"✓ CPU features extracted. Categories: {dataset['Cpu_name'].value_counts().to_dict()}")
    return dataset


def extract_gpu_features(dataset):
    """
    Extract GPU brand from GPU column and filter out unwanted brands.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with GPU features and filtered brands
    """
    # Extract first word (brand) from GPU
    dataset['Gpu_name'] = dataset['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
    
    # Filter out excluded GPU brands
    original_count = len(dataset)
    for brand in config.GPU_BRANDS_TO_EXCLUDE:
        dataset = dataset[dataset['Gpu_name'] != brand]
    
    filtered_count = original_count - len(dataset)
    if filtered_count > 0:
        print(f"✓ GPU features extracted. Filtered out {filtered_count} rows with excluded GPU brands")
    else:
        print(f"✓ GPU features extracted. GPU brands: {dataset['Gpu_name'].nunique()}")
    
    return dataset


def categorize_operating_system(dataset):
    """
    Categorize operating systems into major groups.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with categorized OS
    """
    def categorize_os(os):
        if os in config.OS_WINDOWS_VARIANTS:
            return 'Windows'
        elif os in config.OS_MAC_VARIANTS:
            return 'Mac'
        elif os == 'Linux':
            return 'Linux'
        else:
            return 'Other'
    
    dataset['OpSys'] = dataset['OpSys'].apply(categorize_os)
    print(f"✓ Operating systems categorized: {dataset['OpSys'].value_counts().to_dict()}")
    return dataset


def extract_screen_features(dataset):
    """
    Extract detailed screen features from ScreenResolution column.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with extracted screen features
    """
    # Extract Touchscreen feature
    dataset['Touchscreen'] = dataset['ScreenResolution'].apply(
        lambda x: 1 if 'Touchscreen' in x else 0
    )
    
    # Extract IPS feature
    dataset['IPS'] = dataset['ScreenResolution'].apply(
        lambda x: 1 if 'IPS' in x else 0
    )
    
    # Extract screen resolution dimensions
    def extract_resolution(res_string):
        """Extract width, height, and total pixels from resolution string."""
        match = re.search(r'(\d+)x(\d+)', res_string)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height, width * height
        # Return default resolution if pattern not found
        default_width, default_height = config.DEFAULT_RESOLUTION
        return default_width, default_height, default_width * default_height
    
    dataset['Screen_Width'] = dataset['ScreenResolution'].apply(
        lambda x: extract_resolution(x)[0]
    )
    dataset['Screen_Height'] = dataset['ScreenResolution'].apply(
        lambda x: extract_resolution(x)[1]
    )
    dataset['Total_Pixels'] = dataset['ScreenResolution'].apply(
        lambda x: extract_resolution(x)[2]
    )
    
    # Calculate PPI (Pixels Per Inch) - important quality metric
    dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']
    
    print("✓ Screen features extracted: Touchscreen, IPS, Width, Height, Total_Pixels, PPI")
    return dataset


def create_interaction_features(x):
    """
    Create interaction features for better model performance.
    
    Args:
        x (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Feature matrix with interaction features
    """
    # Add RAM squared (premium laptops may have non-linear pricing with RAM)
    if 'Ram' in x.columns:
        x['Ram_squared'] = x['Ram'] ** 2
        print("✓ Added Ram_squared feature")
    
    # Add interaction between screen quality and size
    if 'Total_Pixels' in x.columns and 'Inches' in x.columns:
        x['Screen_Quality'] = x['Total_Pixels'] / 1000000 * x['Inches']
        print("✓ Added Screen_Quality interaction feature")
    
    return x


def preprocess_dataset(dataset):
    """
    Apply all preprocessing steps to the dataset.
    
    Args:
        dataset (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Fully preprocessed dataset
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Clean numeric columns
    dataset = clean_ram_column(dataset)
    dataset = clean_weight_column(dataset)
    
    # Process categorical features
    dataset = group_companies(dataset)
    dataset = extract_screen_features(dataset)
    dataset = extract_cpu_features(dataset)
    dataset = extract_gpu_features(dataset)
    dataset = categorize_operating_system(dataset)
    
    # Drop redundant columns
    dataset = dataset.drop(columns=config.COLUMNS_TO_DROP)
    print(f"✓ Dropped redundant columns: {config.COLUMNS_TO_DROP}")
    
    print("\n" + "="*60)
    print(f"PREPROCESSING COMPLETE - Final shape: {dataset.shape}")
    print("="*60 + "\n")
    
    return dataset


def prepare_features(dataset, target_column=config.TARGET_VARIABLE):
    """
    Prepare features for modeling: one-hot encoding and interaction features.
    
    Args:
        dataset (pd.DataFrame): Preprocessed dataset
        target_column (str): Name of target variable
        
    Returns:
        tuple: (X, y) - feature matrix and target variable
    """
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60 + "\n")
    
    # Apply one-hot encoding to categorical variables
    dataset = pd.get_dummies(dataset)
    print(f"✓ One-hot encoding applied. New shape: {dataset.shape}")
    
    # Split features and target
    x = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    # Create interaction features
    x = create_interaction_features(x)
    
    print(f"\n✓ Feature preparation complete")
    print(f"  - Features: {x.shape[1]}")
    print(f"  - Samples: {x.shape[0]}")
    print(f"  - Target variable: {target_column}")
    print("="*60 + "\n")
    
    return x, y


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from data_loader import load_data
    
    dataset = load_data()
    dataset = preprocess_dataset(dataset)
    x, y = prepare_features(dataset)
    
    print(f"\nFinal feature matrix shape: {x.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"\nFeature columns:\n{list(x.columns)}")
