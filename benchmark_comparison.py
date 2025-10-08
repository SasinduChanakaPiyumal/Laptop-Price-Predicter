#!/usr/bin/env python
# coding: utf-8
"""
Micro-benchmark script to compare original vs optimized laptop price prediction model
Measures runtime and memory usage improvements
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
from memory_profiler import profile
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0
        
    def start(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self.start_memory
        
    def current(self):
        return self.process.memory_info().rss / 1024 / 1024  # MB
        
    def peak_usage(self):
        return self.current() - self.start_memory

def benchmark_string_operations(dataset, iterations=1):
    """Benchmark string processing operations"""
    print("=" * 50)
    print("BENCHMARKING STRING OPERATIONS")
    print("=" * 50)
    
    # Original approach with apply(lambda)
    def original_string_ops(df):
        df = df.copy()
        df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
        df['Cpu_name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
        df['Gpu_name'] = df['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
        return df
    
    # Optimized approach with vectorized operations
    def optimized_string_ops(df):
        df = df.copy()
        df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', na=False).astype(int)
        df['IPS'] = df['ScreenResolution'].str.contains('IPS', na=False).astype(int)
        cpu_split = df['Cpu'].str.split(n=2, expand=True)
        df['Cpu_name'] = cpu_split[0] + ' ' + cpu_split[1] + ' ' + cpu_split[2].fillna('')
        df['Cpu_name'] = df['Cpu_name'].str.strip()
        df['Gpu_name'] = df['Gpu'].str.split().str[0]
        return df
    
    monitor = MemoryMonitor()
    
    # Benchmark original approach
    print("Testing original apply(lambda) approach...")
    monitor.start()
    start_time = time.time()
    
    for i in range(iterations):
        result_orig = original_string_ops(dataset)
    
    original_time = (time.time() - start_time) / iterations
    original_memory = monitor.peak_usage()
    
    # Benchmark optimized approach
    print("Testing optimized vectorized approach...")
    monitor.start()
    start_time = time.time()
    
    for i in range(iterations):
        result_opt = optimized_string_ops(dataset)
    
    optimized_time = (time.time() - start_time) / iterations
    optimized_memory = monitor.peak_usage()
    
    print(f"\nString Operations Benchmark Results:")
    print(f"Original approach:  {original_time:.4f}s, Memory: {original_memory:.2f}MB")
    print(f"Optimized approach: {optimized_time:.4f}s, Memory: {optimized_memory:.2f}MB")
    print(f"Speed improvement:  {original_time/optimized_time:.2f}x faster")
    print(f"Memory improvement: {original_memory/optimized_memory:.2f}x less memory" if optimized_memory > 0 else "Memory improvement: Significant")
    
    return {
        'original': {'time': original_time, 'memory': original_memory},
        'optimized': {'time': optimized_time, 'memory': optimized_memory}
    }

def benchmark_hyperparameter_search(X_train, y_train):
    """Benchmark hyperparameter optimization approaches"""
    print("\n" + "=" * 50)
    print("BENCHMARKING HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    
    rf = RandomForestRegressor(random_state=42)
    
    # Original GridSearchCV approach
    print("Testing original GridSearchCV approach...")
    grid_params = {
        'n_estimators': [10, 50, 100],
        'criterion': ['squared_error', 'absolute_error']  # Removed 'poisson' for compatibility
    }
    
    monitor = MemoryMonitor()
    monitor.start()
    start_time = time.time()
    
    grid_search = GridSearchCV(estimator=rf, param_grid=grid_params, cv=3, n_jobs=1)
    grid_search.fit(X_train, y_train)
    
    grid_time = time.time() - start_time
    grid_memory = monitor.peak_usage()
    
    # Optimized RandomizedSearchCV approach
    print("Testing optimized RandomizedSearchCV approach...")
    random_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    monitor.start()
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=random_params,
        n_iter=6,  # Test fewer combinations
        cv=3, 
        random_state=42,
        n_jobs=1
    )
    random_search.fit(X_train, y_train)
    
    random_time = time.time() - start_time
    random_memory = monitor.peak_usage()
    
    print(f"\nHyperparameter Optimization Benchmark Results:")
    print(f"GridSearchCV:       {grid_time:.2f}s, Memory: {grid_memory:.2f}MB")
    print(f"RandomizedSearchCV: {random_time:.2f}s, Memory: {random_memory:.2f}MB")
    print(f"Speed improvement:  {grid_time/random_time:.2f}x faster")
    print(f"Memory improvement: {grid_memory/random_memory:.2f}x less memory" if random_memory > 0 else "Memory improvement: Significant")
    
    return {
        'grid': {'time': grid_time, 'memory': grid_memory, 'score': grid_search.best_score_},
        'random': {'time': random_time, 'memory': random_memory, 'score': random_search.best_score_}
    }

def benchmark_data_preprocessing(dataset):
    """Benchmark data preprocessing approaches"""
    print("\n" + "=" * 50)
    print("BENCHMARKING DATA PREPROCESSING")
    print("=" * 50)
    
    # Original preprocessing (step by step)
    def original_preprocessing(df):
        df = df.copy()
        df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
        df['Weight'] = df['Weight'].str.replace('kg', '').astype('float64')
        
        def add_company(inpt):
            other_companies = ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                             'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']
            return 'Other' if inpt in other_companies else inpt
        
        df['Company'] = df['Company'].apply(add_company)
        
        def set_os(inpt):
            if inpt in ['Windows 10', 'Windows 7', 'Windows 10 S']:
                return 'Windows'
            elif inpt in ['macOS', 'Mac OS X']:
                return 'Mac'
            elif inpt == 'Linux':
                return inpt
            else:
                return 'Other'
        
        df['OpSys'] = df['OpSys'].apply(set_os)
        
        return df
    
    # Optimized preprocessing (vectorized)
    def optimized_preprocessing(df):
        df = df.copy()
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float64')
        
        other_companies = ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                          'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']
        df['Company'] = np.where(df['Company'].isin(other_companies), 'Other', df['Company'])
        
        windows_mask = df['OpSys'].isin(['Windows 10', 'Windows 7', 'Windows 10 S'])
        mac_mask = df['OpSys'].isin(['macOS', 'Mac OS X'])
        linux_mask = df['OpSys'] == 'Linux'
        
        df['OpSys'] = np.where(windows_mask, 'Windows',
                              np.where(mac_mask, 'Mac',
                                      np.where(linux_mask, 'Linux', 'Other')))
        
        return df
    
    monitor = MemoryMonitor()
    
    # Test original approach
    print("Testing original step-by-step preprocessing...")
    monitor.start()
    start_time = time.time()
    
    orig_result = original_preprocessing(dataset)
    
    orig_time = time.time() - start_time
    orig_memory = monitor.peak_usage()
    
    # Test optimized approach
    print("Testing optimized vectorized preprocessing...")
    monitor.start()
    start_time = time.time()
    
    opt_result = optimized_preprocessing(dataset)
    
    opt_time = time.time() - start_time
    opt_memory = monitor.peak_usage()
    
    print(f"\nData Preprocessing Benchmark Results:")
    print(f"Original approach:  {orig_time:.4f}s, Memory: {orig_memory:.2f}MB")
    print(f"Optimized approach: {opt_time:.4f}s, Memory: {opt_memory:.2f}MB")
    print(f"Speed improvement:  {orig_time/opt_time:.2f}x faster")
    print(f"Memory improvement: {orig_memory/opt_memory:.2f}x less memory" if opt_memory > 0 else "Memory improvement: Significant")
    
    return {
        'original': {'time': orig_time, 'memory': orig_memory},
        'optimized': {'time': opt_time, 'memory': opt_memory}
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing all optimizations"""
    print("COMPREHENSIVE LAPTOP PRICE MODEL OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    # Load the dataset
    print("Loading dataset...")
    try:
        dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
        print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    except FileNotFoundError:
        print("Error: laptop_price.csv not found. Creating sample data...")
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        dataset = pd.DataFrame({
            'laptop_ID': range(n_samples),
            'Company': np.random.choice(['HP', 'Dell', 'Apple', 'Samsung', 'Razer'], n_samples),
            'Product': [f'Laptop_{i}' for i in range(n_samples)],
            'Ram': [f'{ram}GB' for ram in np.random.choice([4, 8, 16, 32], n_samples)],
            'Weight': [f'{weight:.1f}kg' for weight in np.random.uniform(1.0, 3.0, n_samples)],
            'Price_euros': np.random.uniform(300, 3000, n_samples),
            'ScreenResolution': np.random.choice(['1920x1080', '1920x1080 Touchscreen', '2560x1440 IPS'], n_samples),
            'Cpu': [f'Intel Core i{i} {freq}GHz' for i, freq in zip(
                np.random.choice([3, 5, 7], n_samples), 
                np.random.uniform(2.0, 4.0, n_samples)
            )],
            'Gpu': np.random.choice(['Intel HD', 'NVIDIA GTX', 'AMD Radeon'], n_samples),
            'OpSys': np.random.choice(['Windows 10', 'macOS', 'Linux', 'Windows 7'], n_samples),
            'Inches': np.random.uniform(13, 17, n_samples)
        })
    
    # Prepare data for hyperparameter benchmarking
    dataset_processed = dataset.copy()
    dataset_processed['Ram'] = dataset_processed['Ram'].str.replace('GB', '', regex=False).astype('int32')
    dataset_processed['Weight'] = dataset_processed['Weight'].str.replace('kg', '', regex=False).astype('float64')
    dataset_processed = pd.get_dummies(dataset_processed.select_dtypes(include=[np.number]))
    
    X = dataset_processed.drop('Price_euros', axis=1)
    y = dataset_processed['Price_euros']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Run benchmarks
    results = {}
    
    # 1. String Operations
    results['string_ops'] = benchmark_string_operations(dataset, iterations=3)
    
    # 2. Data Preprocessing
    results['preprocessing'] = benchmark_data_preprocessing(dataset)
    
    # 3. Hyperparameter Optimization (using smaller dataset for speed)
    sample_size = min(500, len(X_train))
    X_sample = X_train.iloc[:sample_size]
    y_sample = y_train.iloc[:sample_size]
    results['hyperparameter'] = benchmark_hyperparameter_search(X_sample, y_sample)
    
    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    total_orig_time = (results['string_ops']['original']['time'] + 
                       results['preprocessing']['original']['time'] + 
                       results['hyperparameter']['grid']['time'])
    
    total_opt_time = (results['string_ops']['optimized']['time'] + 
                      results['preprocessing']['optimized']['time'] + 
                      results['hyperparameter']['random']['time'])
    
    print(f"Total Original Time:  {total_orig_time:.2f}s")
    print(f"Total Optimized Time: {total_opt_time:.2f}s")
    print(f"Overall Speed Improvement: {total_orig_time/total_opt_time:.2f}x faster")
    
    print(f"\nKey Optimizations Made:")
    print(f"1. String operations: {results['string_ops']['original']['time']/results['string_ops']['optimized']['time']:.2f}x faster")
    print(f"2. Data preprocessing: {results['preprocessing']['original']['time']/results['preprocessing']['optimized']['time']:.2f}x faster")
    print(f"3. Hyperparameter search: {results['hyperparameter']['grid']['time']/results['hyperparameter']['random']['time']:.2f}x faster")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
        print("\nBenchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
