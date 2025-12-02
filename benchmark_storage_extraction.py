#!/usr/bin/env python3
"""
Micro-benchmark: old vs new storage feature extraction

This script constructs a synthetic dataset by replicating rows from laptop_price.csv
(if available) or using a representative fallback list of Memory strings.
It measures runtime and peak memory for:
  - Old implementation (used .apply on lists)
  - New fully vectorized implementation using extractall + groupby

Usage:
  python benchmark_storage_extraction.py --scale 20000

Notes:
  - The larger the --scale, the clearer the difference. On 20k-100k rows,
    the new function should be significantly faster and more memory efficient.
  - The script prints timings and peak memory in MB.
"""
import argparse
import time
import os
import numpy as np
import pandas as pd

# Local import: load the optimized function from the main script by name
# Import the new implementation from the shared utility module
from utils_features import extract_storage_features_vectorized as new_extract

# Define a reference (old) implementation for comparison
# (Matches previous behavior but intentionally uses .apply with Python loops)
def old_extract_storage_features(dataset: pd.DataFrame) -> pd.DataFrame:
    s = dataset['Memory'].fillna('')
    dataset = dataset.copy()
    dataset['Has_SSD'] = s.str.contains('SSD', case=False, regex=False).astype('int8')
    dataset['Has_HDD'] = s.str.contains('HDD', case=False, regex=False).astype('int8')
    dataset['Has_Flash'] = s.str.contains('Flash', case=False, regex=False).astype('int8')
    dataset['Has_Hybrid'] = s.str.contains('Hybrid', case=False, regex=False).astype('int8')

    tb_matches = s.str.findall(r'(\d+(?:\.\d+)?)\s*TB')
    tb_capacity = tb_matches.apply(lambda x: sum([float(i) * 1024 for i in x]) if x else 0)
    gb_matches = s.str.findall(r'(\d+(?:\.\d+)?)\s*GB')
    gb_capacity = gb_matches.apply(lambda x: sum([float(i) for i in x]) if x else 0)

    dataset['Storage_Capacity_GB'] = (tb_capacity + gb_capacity).astype('float32')
    return dataset


def make_dataset(scale: int) -> pd.DataFrame:
    # Try to use the real CSV for realism
    csv_path = 'laptop_price.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='latin-1', usecols=['Memory'])
        if len(df) == 0:
            raise RuntimeError('laptop_price.csv has no rows')
        reps = int(np.ceil(scale / len(df)))
        df_big = pd.concat([df] * reps, ignore_index=True).iloc[:scale].copy()
        return df_big
    # Fallback crafted examples
    base = pd.DataFrame({
        'Memory': [
            '128 GB SSD', '256 GB SSD', '512 GB SSD', '1 TB HDD', '2 TB HDD',
            '256 GB SSD + 1 TB HDD', '512 GB SSD + 512 GB SSD', '32 GB Flash',
            '64 GB Flash + 256 GB SSD', '1.5 TB Hybrid', '500 GB HDD + 16 GB SSD Cache'
        ]
    })
    reps = int(np.ceil(scale / len(base)))
    return pd.concat([base] * reps, ignore_index=True).iloc[:scale].copy()


def bench(fn, df: pd.DataFrame, label: str):
    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(df.copy())
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Convert to MB
    peak_mb = peak / (1024 * 1024)
    print(f"{label:28s} time: {dt:.3f} s | peak: {peak_mb:.1f} MB | rows: {len(df)}")
    # Sanity check: columns exist
    for col in ['Has_SSD', 'Has_HDD', 'Has_Flash', 'Has_Hybrid', 'Storage_Capacity_GB']:
        assert col in out.columns
    return dt, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=20000, help='Number of rows to benchmark')
    args = parser.parse_args()

    df = make_dataset(args.scale)
    print(f"Dataset ready: {len(df)} rows")

    bench(old_extract_storage_features, df, 'old_apply_version')
    bench(new_extract, df, 'new_vectorized_version')


if __name__ == '__main__':
    main()
