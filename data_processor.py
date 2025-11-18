import pandas as pd

def load_and_preprocess_data(file_path):
    """
    Loads the laptop price dataset and performs initial preprocessing steps.

    Args:
        file_path (str): The path to the laptop price CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    dataset = pd.read_csv(file_path, encoding='latin-1')
    
    # Preprocessing 'Ram' column
    dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
    
    # Preprocessing 'Weight' column
    dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float64')
    
    return dataset

def process_company_data(df):
    """
    Processes the 'Company' column, grouping less frequent companies into 'Other'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with processed 'Company' column.
    """
    OTHER_COMPANIES = {'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'}
    def add_company(inpt):
        return 'Other' if inpt in OTHER_COMPANIES else inpt
    df['Company'] = df['Company'].apply(add_company)
    return df

def extract_screen_features(df):
    """
    Extracts 'Touchscreen' and 'IPS' features from the 'ScreenResolution' column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new 'Touchscreen' and 'IPS' columns.
    """
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
    return df

def process_cpu_data(df):
    """
    Processes the 'Cpu' column to extract and categorize CPU names.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Cpu_name' column.
    """
    df['Cpu_name']= df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
    def set_processor(name):
        if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
            return name
        else:
            if name.split()[0] == 'AMD':
                return 'AMD'
            else:
                return 'Other'
    df['Cpu_name'] = df['Cpu_name'].apply(set_processor)
    return df

def process_gpu_data(df):
    """
    Processes the 'Gpu' column to extract GPU names and filter out 'ARM'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Gpu_name' column and filtered rows.
    """
    df['Gpu_name']= df['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))
    df = df[df['Gpu_name'] != 'ARM']
    return df

def process_os_data(df):
    """
    Processes the 'OpSys' column, categorizing operating systems.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with processed 'OpSys' column.
    """
    def set_os(inpt):
        if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
            return 'Windows'
        elif inpt == 'macOS' or inpt == 'Mac OS X':
            return 'Mac'
        elif inpt == 'Linux':
            return inpt
        else:
            return 'Other'
    df['OpSys']= df['OpSys'].apply(set_os)
    return df

import re

def extract_storage_features(memory_string):
    """
    Extracts storage features (SSD, HDD, Flash, Hybrid, and total capacity) from a memory string.
    """
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
    
    # Find all capacity values with TB or GB
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    # Convert to GB and sum
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb

def process_storage_data(df):
    """
    Applies storage feature extraction and creates derived storage features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new storage-related features.
    """
    storage_features = df['Memory'].apply(extract_storage_features)
    df['Has_SSD'] = storage_features.apply(lambda x: x[0])
    df['Has_HDD'] = storage_features.apply(lambda x: x[1])
    df['Has_Flash'] = storage_features.apply(lambda x: x[2])
    df['Has_Hybrid'] = storage_features.apply(lambda x: x[3])
    df['Storage_Capacity_GB'] = storage_features.apply(lambda x: x[4])

    df['Storage_Type_Score'] = (
        df['Has_SSD'] * 3 +      # SSD is premium
        df['Has_Flash'] * 2.5 +  # Flash is also premium
        df['Has_Hybrid'] * 2 +   # Hybrid is mid-range
        df['Has_HDD'] * 1        # HDD is budget
    )
    return df

def drop_unnecessary_columns(df):
    """
    Drops columns that are no longer needed after feature extraction.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    # Keep screen size and drop only redundant columns (now also drop Memory after feature extraction)
    df = df.drop(columns=['laptop_ID','Product','ScreenResolution','Cpu','Gpu','Memory'])
    return df

def one_hot_encode_features(df):
    """
    Performs one-hot encoding on categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded features.
    """
    df = pd.get_dummies(df)
    return df

def create_interaction_features(df):
    """
    Creates various interaction features for better predictions.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new interaction features.
    """
    # Add RAM squared (premium laptops may have non-linear pricing with RAM)
    if 'Ram' in df.columns:
        df['Ram_squared'] = df['Ram'] ** 2
        
    # Add interaction between screen quality and size
    # Note: 'Total_Pixels' and 'Inches' might be created in other processing steps. 
    # Ensure these columns exist before creating interaction features.
    if 'Total_Pixels' in df.columns and 'Inches' in df.columns:
        df['Screen_Quality'] = df['Total_Pixels'] / 1000000 * df['Inches']  # Normalized quality metric

    # Storage capacity * SSD indicator (SSD with high capacity is premium)
    if 'Storage_Capacity_GB' in df.columns and 'Has_SSD' in df.columns:
        df['Premium_Storage'] = df['Storage_Capacity_GB'] * (df['Has_SSD'] + 1) / 1000  # Normalized

    # RAM * Storage Type Score (high RAM + fast storage = workstation/gaming)
    if 'Ram' in df.columns and 'Storage_Type_Score' in df.columns:
        df['RAM_Storage_Quality'] = df['Ram'] * df['Storage_Type_Score']

    # Screen quality * Storage quality (premium display + premium storage)
    if 'PPI' in df.columns and 'Storage_Type_Score' in df.columns:
        df['Display_Storage_Premium'] = df['PPI'] * df['Storage_Type_Score']

    # Weight to size ratio (portability factor)
    if 'Weight' in df.columns and 'Inches' in df.columns:
        df['Weight_Size_Ratio'] = df['Weight'] / df['Inches']

    # Total pixels per RAM (graphics capability estimation)
    if 'Total_Pixels' in df.columns and 'Ram' in df.columns:
        df['Pixels_Per_RAM'] = df['Total_Pixels'] / (df['Ram'] * 1000000)

    # Storage per inch (how much storage per screen size)
    if 'Storage_Capacity_GB' in df.columns and 'Inches' in df.columns:
        df['Storage_Per_Inch'] = df['Storage_Capacity_GB'] / df['Inches']
        
    return df
