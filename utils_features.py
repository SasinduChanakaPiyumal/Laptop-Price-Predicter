"""
Utility functions for feature extraction and performance-optimized transforms.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Lightweight timer decorator for optional benchmarking in notebooks/scripts
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        dt = time.time() - t0
        print(f"⏱️  {func.__name__} took {dt:.2f} seconds")
        return res
    return wrapper


@timer
def extract_storage_features_vectorized(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Extract storage features using fully vectorized operations (no Python-level loops or .apply).
    Outputs:
      - Has_SSD, Has_HDD, Has_Flash, Has_Hybrid (int8)
      - Storage_Capacity_GB (float32)
    """
    # Vectorized storage type detection
    mem = dataset['Memory'].fillna("")
    dataset['Has_SSD'] = mem.str.contains('SSD', case=False, regex=False).astype('int8')
    dataset['Has_HDD'] = mem.str.contains('HDD', case=False, regex=False).astype('int8')
    dataset['Has_Flash'] = mem.str.contains('Flash', case=False, regex=False).astype('int8')
    dataset['Has_Hybrid'] = mem.str.contains('Hybrid', case=False, regex=False).astype('int8')

    # Vectorized capacity parsing
    pattern = r'(?i)(\d+(?:\.\d+)?)\s*(TB|GB)'
    extracted = mem.str.extractall(pattern)

    if extracted.empty:
        dataset['Storage_Capacity_GB'] = np.zeros(len(dataset), dtype='float32')
        return dataset

    values = extracted[0].astype('float32')
    units = extracted[1].str.upper().map({'GB': np.float32(1.0), 'TB': np.float32(1024.0)}).astype('float32')
    capacity_gb_per_match = values * units
    summed = capacity_gb_per_match.groupby(level=0).sum()
    dataset['Storage_Capacity_GB'] = (
        summed.reindex(dataset.index, fill_value=np.float32(0.0)).astype('float32')
    )
    return dataset
