#!/usr/bin/env python3
"""
Micro-benchmark for storage feature extraction

Compares runtime and basic memory footprint between:
- original apply-based extraction
- new vectorized extraction

Usage: python bench_storage_features.py [multiplier]
The optional multiplier replicates the dataset to amplify runtime.
"""
import sys
import time
import gc

import pandas as pd
import numpy as np

from utils.storage_features import (
    extract_storage_features_apply,
    extract_storage_features_vectorized,
)

CSV_PATH = "laptop_price.csv"


def load_memory_series() -> pd.Series:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    return df["Memory"]


def replicate(series: pd.Series, k: int) -> pd.Series:
    if k <= 1:
        return series.reset_index(drop=True)
    return pd.concat([series] * k, ignore_index=True)


def time_call(fn, *args, **kwargs):
    gc.collect()
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, res


def main():
    k = 1
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
        except ValueError:
            print("Invalid multiplier; using 1")
            k = 1

    mem_series = load_memory_series()
    mem_series = replicate(mem_series, k)
    n = len(mem_series)
    print(f"Rows: {n} (replication x{k})")

    # Warmup
    _ = extract_storage_features_vectorized(mem_series)

    t_vec, df_vec = time_call(extract_storage_features_vectorized, mem_series)
    print(f"Vectorized: {t_vec:.4f}s, shape={df_vec.shape}")

    t_app, df_app = time_call(extract_storage_features_apply, mem_series)
    print(f"Apply:      {t_app:.4f}s, shape={df_app.shape}")

    speedup = t_app / max(t_vec, 1e-9)
    print(f"Speedup (apply/vectorized): {speedup:.2f}x")

    # Basic correctness check (within tolerance for floats)
    try:
        assert df_vec[["Has_SSD", "Has_HDD", "Has_Flash", "Has_Hybrid"]].equals(
            df_app[["Has_SSD", "Has_HDD", "Has_Flash", "Has_Hybrid"]]
        )
        if not np.allclose(
            df_vec["Storage_Capacity_GB"].values,
            df_app["Storage_Capacity_GB"].values,
            rtol=1e-6,
            atol=1e-3,
            equal_nan=True,
        ):
            raise AssertionError("Storage_Capacity_GB mismatch")
        if not np.allclose(
            df_vec["Storage_Type_Score"].values,
            df_app["Storage_Type_Score"].values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ):
            raise AssertionError("Storage_Type_Score mismatch")
        print("Correctness: PASS")
    except AssertionError as e:
        print(f"Correctness: FAIL -> {e}")


if __name__ == "__main__":
    main()
