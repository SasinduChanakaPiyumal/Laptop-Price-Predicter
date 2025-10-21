#!/usr/bin/env python
# coding: utf-8

"""
Performance Benchmark: Feature Extraction Optimization
========================================================

This script benchmarks the performance improvements made to the laptop price
prediction model's feature extraction functions.

BOTTLENECKS IDENTIFIED:
1. extract_resolution(): Called 3 times per row (3x regex operations per row)
2. extract_storage_features(): Applied once but then accessed 5 times with lambdas

OPTIMIZATIONS APPLIED:
1. extract_resolution(): Now returns pd.Series and unpacks all values at once
2. extract_storage_features(): Now returns pd.Series and unpacks all values at once

Expected Performance Gains:
- Screen resolution: ~66% faster (3 calls -> 1 call per row)
- Storage features: ~80% faster (6 operations -> 1 operation per row)
"""

import pandas as pd
import numpy as np
import time
import re
from memory_profiler import profile


# ============================================================================
# ORIGINAL (SLOW) IMPLEMENTATIONS
# ============================================================================

def extract_resolution_OLD(res_string):
    """Original implementation returning tuple"""
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height, width * height
    return 1366, 768, 1366*768


def extract_storage_features_OLD(memory_string):
    """Original implementation returning tuple"""
    memory_string = str(memory_string)
    has_ssd = 1 if 'SSD' in memory_string else 0
    has_hdd = 1 if 'HDD' in memory_string else 0
    has_flash = 1 if 'Flash' in memory_string else 0
    has_hybrid = 1 if 'Hybrid' in memory_string else 0
    total_capacity_gb = 0
    
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb


# ============================================================================
# OPTIMIZED (FAST) IMPLEMENTATIONS
# ============================================================================

def extract_resolution_NEW(res_string):
    """Optimized implementation returning pd.Series"""
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return pd.Series([width, height, width * height])
    return pd.Series([1366, 768, 1366*768])


def extract_storage_features_NEW(memory_string):
    """Optimized implementation returning pd.Series"""
    memory_string = str(memory_string)
    has_ssd = 1 if 'SSD' in memory_string else 0
    has_hdd = 1 if 'HDD' in memory_string else 0
    has_flash = 1 if 'Flash' in memory_string else 0
    has_hybrid = 1 if 'Hybrid' in memory_string else 0
    total_capacity_gb = 0
    
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return pd.Series([has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb])


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_resolution_old(df):
    """Original approach: Call function 3 times per row"""
    start = time.time()
    df['Screen_Width'] = df['ScreenResolution'].apply(lambda x: extract_resolution_OLD(x)[0])
    df['Screen_Height'] = df['ScreenResolution'].apply(lambda x: extract_resolution_OLD(x)[1])
    df['Total_Pixels'] = df['ScreenResolution'].apply(lambda x: extract_resolution_OLD(x)[2])
    end = time.time()
    return end - start


def benchmark_resolution_new(df):
    """Optimized approach: Call function once per row, unpack all values"""
    start = time.time()
    df[['Screen_Width', 'Screen_Height', 'Total_Pixels']] = df['ScreenResolution'].apply(extract_resolution_NEW)
    end = time.time()
    return end - start


def benchmark_storage_old(df):
    """Original approach: Apply function then extract each element with lambda"""
    start = time.time()
    storage_features = df['Memory'].apply(extract_storage_features_OLD)
    df['Has_SSD'] = storage_features.apply(lambda x: x[0])
    df['Has_HDD'] = storage_features.apply(lambda x: x[1])
    df['Has_Flash'] = storage_features.apply(lambda x: x[2])
    df['Has_Hybrid'] = storage_features.apply(lambda x: x[3])
    df['Storage_Capacity_GB'] = storage_features.apply(lambda x: x[4])
    end = time.time()
    return end - start


def benchmark_storage_new(df):
    """Optimized approach: Apply function once and unpack all values"""
    start = time.time()
    df[['Has_SSD', 'Has_HDD', 'Has_Flash', 'Has_Hybrid', 'Storage_Capacity_GB']] = df['Memory'].apply(extract_storage_features_NEW)
    end = time.time()
    return end - start


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_benchmark():
    """Run complete benchmark suite"""
    print("="*80)
    print("PERFORMANCE BENCHMARK: Feature Extraction Optimization")
    print("="*80)
    
    # Load the dataset
    try:
        dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
        print(f"\nDataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    except FileNotFoundError:
        print("\nERROR: laptop_price.csv not found. Creating synthetic data for testing...")
        # Create synthetic data for testing
        n_rows = 1000
        dataset = pd.DataFrame({
            'ScreenResolution': ['1920x1080'] * (n_rows // 2) + ['1366x768'] * (n_rows // 2),
            'Memory': ['256GB SSD'] * (n_rows // 4) + ['1TB HDD'] * (n_rows // 4) + 
                     ['128GB SSD +  1TB HDD'] * (n_rows // 4) + ['512GB Flash Storage'] * (n_rows // 4)
        })
        print(f"Synthetic dataset created: {dataset.shape[0]} rows")
    
    # Run benchmarks multiple times for statistical significance
    n_runs = 5
    
    print("\n" + "-"*80)
    print("BENCHMARK 1: Screen Resolution Extraction")
    print("-"*80)
    
    resolution_old_times = []
    resolution_new_times = []
    
    for i in range(n_runs):
        df_test = dataset.copy()
        time_old = benchmark_resolution_old(df_test)
        resolution_old_times.append(time_old)
        
        df_test = dataset.copy()
        time_new = benchmark_resolution_new(df_test)
        resolution_new_times.append(time_new)
        
        print(f"Run {i+1}: OLD={time_old:.4f}s, NEW={time_new:.4f}s, Speedup={time_old/time_new:.2f}x")
    
    avg_old = np.mean(resolution_old_times)
    avg_new = np.mean(resolution_new_times)
    speedup = avg_old / avg_new
    improvement = ((avg_old - avg_new) / avg_old) * 100
    
    print(f"\nRESULTS (Screen Resolution):")
    print(f"  Average OLD: {avg_old:.4f}s (±{np.std(resolution_old_times):.4f}s)")
    print(f"  Average NEW: {avg_new:.4f}s (±{np.std(resolution_new_times):.4f}s)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}% faster")
    
    print("\n" + "-"*80)
    print("BENCHMARK 2: Storage Feature Extraction")
    print("-"*80)
    
    storage_old_times = []
    storage_new_times = []
    
    for i in range(n_runs):
        df_test = dataset.copy()
        time_old = benchmark_storage_old(df_test)
        storage_old_times.append(time_old)
        
        df_test = dataset.copy()
        time_new = benchmark_storage_new(df_test)
        storage_new_times.append(time_new)
        
        print(f"Run {i+1}: OLD={time_old:.4f}s, NEW={time_new:.4f}s, Speedup={time_old/time_new:.2f}x")
    
    avg_old = np.mean(storage_old_times)
    avg_new = np.mean(storage_new_times)
    speedup = avg_old / avg_new
    improvement = ((avg_old - avg_new) / avg_old) * 100
    
    print(f"\nRESULTS (Storage Features):")
    print(f"  Average OLD: {avg_old:.4f}s (±{np.std(storage_old_times):.4f}s)")
    print(f"  Average NEW: {avg_new:.4f}s (±{np.std(storage_new_times):.4f}s)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}% faster")
    
    # Calculate total time saved
    print("\n" + "="*80)
    print("OVERALL IMPACT")
    print("="*80)
    
    total_old = np.mean(resolution_old_times) + np.mean(storage_old_times)
    total_new = np.mean(resolution_new_times) + np.mean(storage_new_times)
    total_speedup = total_old / total_new
    total_improvement = ((total_old - total_new) / total_old) * 100
    time_saved = total_old - total_new
    
    print(f"Total feature extraction time:")
    print(f"  OLD: {total_old:.4f}s")
    print(f"  NEW: {total_new:.4f}s")
    print(f"  Time saved: {time_saved:.4f}s ({time_saved*1000:.1f}ms)")
    print(f"  Overall speedup: {total_speedup:.2f}x")
    print(f"  Overall improvement: {total_improvement:.1f}% faster")
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print("\nORIGINAL IMPLEMENTATION ISSUES:")
    print("1. Screen Resolution: Function called 3x per row")
    print("   - Each call performs regex matching")
    print("   - Total regex operations: 3 × num_rows")
    print("\n2. Storage Features: Function applied once but accessed 5x")
    print("   - Creates intermediate Series of tuples")
    print("   - 5 additional apply() operations to extract values")
    print("   - Total operations: 6 × num_rows")
    
    print("\nOPTIMIZED IMPLEMENTATION:")
    print("1. Screen Resolution: Function called 1x per row")
    print("   - Returns pd.Series, unpacked directly")
    print("   - Total regex operations: 1 × num_rows")
    print("   - Reduction: 66.7% fewer operations")
    
    print("\n2. Storage Features: Function applied once, values unpacked")
    print("   - Returns pd.Series, unpacked directly")
    print("   - Total operations: 1 × num_rows")
    print("   - Reduction: 83.3% fewer operations")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("✓ Always return pd.Series from apply() when extracting multiple values")
    print("✓ Avoid calling functions multiple times per row")
    print("✓ Avoid intermediate tuple unpacking with multiple lambdas")
    print("✓ Profile code regularly to identify similar bottlenecks")
    print("\n")


if __name__ == "__main__":
    # Check if memory_profiler is available for memory usage tracking
    try:
        from memory_profiler import profile
        print("Note: memory_profiler is available. Install with: pip install memory-profiler")
    except ImportError:
        print("Note: memory_profiler not installed. Install with: pip install memory-profiler")
        print("      (Memory profiling will be skipped)\n")
    
    run_benchmark()
