#!/usr/bin/env python
# coding: utf-8
"""
Micro-benchmark script to measure performance improvement of resolution extraction optimization.

This script compares the BEFORE and AFTER versions of the extract_resolution function
to demonstrate the performance gains from calling the function once per row instead of three times.
"""

import pandas as pd
import numpy as np
import time
import re

# Create sample data similar to the laptop dataset
def create_sample_data(n_rows=1000):
    """Create sample screen resolution data for benchmarking."""
    resolutions = [
        'Full HD 1920x1080',
        'IPS Panel Full HD 1920x1080',
        'Touchscreen IPS Panel Full HD 1920x1080',
        '4K Ultra HD 3840x2160',
        'Touchscreen 4K Ultra HD 3840x2160',
        'HD 1366x768',
        'IPS Panel HD 1366x768',
        'QHD 2560x1440',
        'Touchscreen QHD 2560x1440',
        'Full HD 1920x1200'
    ]
    
    data = {
        'ScreenResolution': np.random.choice(resolutions, size=n_rows),
        'Inches': np.random.uniform(13.0, 17.0, size=n_rows)
    }
    
    return pd.DataFrame(data)


# BEFORE: Original inefficient implementation (called 3 times per row)
def extract_resolution_old(res_string):
    """Original version with re import inside function."""
    import re
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height, width * height
    return 1366, 768, 1366*768


def benchmark_old_method(df):
    """Benchmark the OLD inefficient method (3 function calls per row)."""
    dataset = df.copy()
    
    start_time = time.time()
    
    # This is how it was done before: 3 separate apply calls
    dataset['Screen_Width'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution_old(x)[0])
    dataset['Screen_Height'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution_old(x)[1])
    dataset['Total_Pixels'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution_old(x)[2])
    dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return elapsed, dataset


# AFTER: Optimized implementation (called once per row)
def extract_resolution_new(res_string):
    """Optimized version with re import outside."""
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height, width * height
    return 1366, 768, 1366*768


def benchmark_new_method(df):
    """Benchmark the NEW optimized method (1 function call per row)."""
    dataset = df.copy()
    
    start_time = time.time()
    
    # Optimized: Call extract_resolution once per row
    resolution_data = dataset['ScreenResolution'].apply(extract_resolution_new)
    dataset['Screen_Width'] = resolution_data.apply(lambda x: x[0])
    dataset['Screen_Height'] = resolution_data.apply(lambda x: x[1])
    dataset['Total_Pixels'] = resolution_data.apply(lambda x: x[2])
    dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return elapsed, dataset


def verify_correctness(df_old, df_new):
    """Verify that both methods produce identical results."""
    columns_to_check = ['Screen_Width', 'Screen_Height', 'Total_Pixels', 'PPI']
    
    for col in columns_to_check:
        if not df_old[col].equals(df_new[col]):
            print(f"❌ MISMATCH in column {col}")
            return False
    
    print("✅ Results verified: Both methods produce identical output")
    return True


def run_benchmark(n_rows=1000, n_iterations=5):
    """Run the benchmark multiple times and report statistics."""
    print("="*70)
    print(f"MICRO-BENCHMARK: Resolution Extraction Optimization")
    print("="*70)
    print(f"\nDataset size: {n_rows} rows")
    print(f"Iterations: {n_iterations}")
    print("\n" + "-"*70)
    
    # Create sample data
    df = create_sample_data(n_rows)
    
    old_times = []
    new_times = []
    
    # Run multiple iterations
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}:")
        
        # Benchmark old method
        old_time, df_old = benchmark_old_method(df)
        old_times.append(old_time)
        print(f"  OLD method (3 calls/row): {old_time:.4f} seconds")
        
        # Benchmark new method
        new_time, df_new = benchmark_new_method(df)
        new_times.append(new_time)
        print(f"  NEW method (1 call/row):  {new_time:.4f} seconds")
        
        # Speedup for this iteration
        speedup = old_time / new_time
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify correctness on first iteration
        if i == 0:
            print("\n" + "-"*70)
            verify_correctness(df_old, df_new)
            print("-"*70)
    
    # Calculate statistics
    old_mean = np.mean(old_times)
    old_std = np.std(old_times)
    new_mean = np.mean(new_times)
    new_std = np.std(new_times)
    speedup_mean = old_mean / new_mean
    time_saved = old_mean - new_mean
    improvement_pct = (time_saved / old_mean) * 100
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"\nOLD method (3 calls per row):")
    print(f"  Mean time: {old_mean:.4f} ± {old_std:.4f} seconds")
    print(f"\nNEW method (1 call per row):")
    print(f"  Mean time: {new_mean:.4f} ± {new_std:.4f} seconds")
    print(f"\n{'='*70}")
    print(f"SPEEDUP: {speedup_mean:.2f}x faster")
    print(f"TIME SAVED: {time_saved:.4f} seconds ({improvement_pct:.1f}% improvement)")
    print(f"{'='*70}")
    
    # Memory efficiency note
    print(f"\nMemory Efficiency:")
    print(f"  - OLD: 3 × {n_rows} = {3*n_rows} regex operations")
    print(f"  - NEW: 1 × {n_rows} = {n_rows} regex operations")
    print(f"  - Reduction: {2*n_rows} fewer operations ({66.7:.1f}% reduction)")
    
    # Real-world impact
    print(f"\nReal-World Impact (for typical laptop dataset ~1000 rows):")
    estimated_saving = (time_saved / n_rows) * 1000
    print(f"  - Estimated time saved: ~{estimated_saving:.3f} seconds per run")
    print(f"  - For 100 runs (during development/testing): ~{estimated_saving*100:.2f} seconds saved")
    
    print("\n" + "="*70)
    print("OPTIMIZATION DETAILS")
    print("="*70)
    print("\nWhat was changed:")
    print("  1. Moved 're' module import outside the function")
    print("  2. Called extract_resolution() ONCE per row instead of 3 times")
    print("  3. Stored result in intermediate variable 'resolution_data'")
    print("  4. Extracted width, height, pixels from stored tuples")
    print("\nWhy it matters:")
    print("  - Eliminates redundant regex parsing (3N → N operations)")
    print("  - Removes repeated module imports (3N → 0 per-call imports)")
    print("  - Reduces function call overhead (3N → N calls)")
    print("  - Better CPU cache utilization")
    print("="*70)


if __name__ == "__main__":
    # Run benchmark with different dataset sizes
    print("\n🚀 Starting benchmark with 1000 rows (typical dataset size)...\n")
    run_benchmark(n_rows=1000, n_iterations=5)
    
    print("\n\n" + "🔬 Additional test with larger dataset (5000 rows)...\n")
    run_benchmark(n_rows=5000, n_iterations=3)
