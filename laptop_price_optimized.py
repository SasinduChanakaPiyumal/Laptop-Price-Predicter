#!/usr/bin/env python
# coding: utf-8
"""
Optimized Laptop Price Prediction Model
- Vectorized string operations instead of apply(lambda)
- Efficient hyperparameter search
- Batched data processing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import time

def load_and_preprocess_data(filename="laptop_price.csv"):
    """Load and preprocess the laptop dataset with optimized operations"""
    print("Loading dataset...")
    dataset = pd.read_csv(filename, encoding='latin-1')
    
    print("Preprocessing data...")
    
    # Optimize RAM and Weight conversion using vectorized operations
    dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
    dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float64')
    
    # Vectorized company categorization using np.where and isin()
    other_companies = ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                      'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']
    dataset['Company'] = np.where(dataset['Company'].isin(other_companies), 'Other', dataset['Company'])
    
    # Vectorized touchscreen and IPS detection using str.contains()
    dataset['Touchscreen'] = dataset['ScreenResolution'].str.contains('Touchscreen', na=False).astype(int)
    dataset['IPS'] = dataset['ScreenResolution'].str.contains('IPS', na=False).astype(int)
    
    # Optimized CPU name extraction using str.split() with vectorized operations
    cpu_split = dataset['Cpu'].str.split(n=2, expand=True)
    dataset['Cpu_name'] = cpu_split[0] + ' ' + cpu_split[1] + ' ' + cpu_split[2].fillna('')
    dataset['Cpu_name'] = dataset['Cpu_name'].str.strip()
    
    # Vectorized processor categorization
    intel_core_mask = dataset['Cpu_name'].isin(['Intel Core i7', 'Intel Core i5', 'Intel Core i3'])
    amd_mask = dataset['Cpu_name'].str.startswith('AMD', na=False)
    dataset['Cpu_name'] = np.where(intel_core_mask, dataset['Cpu_name'],
                                  np.where(amd_mask, 'AMD', 'Other'))
    
    # Optimized GPU name extraction
    dataset['Gpu_name'] = dataset['Gpu'].str.split().str[0]
    
    # Filter out ARM GPUs
    dataset = dataset[dataset['Gpu_name'] != 'ARM'].copy()
    
    # Vectorized OS categorization
    windows_mask = dataset['OpSys'].isin(['Windows 10', 'Windows 7', 'Windows 10 S'])
    mac_mask = dataset['OpSys'].isin(['macOS', 'Mac OS X'])
    linux_mask = dataset['OpSys'] == 'Linux'
    
    dataset['OpSys'] = np.where(windows_mask, 'Windows',
                               np.where(mac_mask, 'Mac',
                                       np.where(linux_mask, 'Linux', 'Other')))
    
    # Drop unnecessary columns
    dataset = dataset.drop(columns=['laptop_ID', 'Inches', 'Product', 
                                   'ScreenResolution', 'Cpu', 'Gpu'])
    
    # One-hot encoding
    dataset = pd.get_dummies(dataset, drop_first=True)
    
    return dataset

def train_models(X, y):
    """Train multiple models and return performance metrics"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42)
    }
    
    results = {}
    print("\nTraining models...")
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        score = model.score(X_test, y_test)
        results[name] = {'model': model, 'score': score, 'train_time': train_time}
        print(f"{name}: Score = {score:.4f}, Time = {train_time:.2f}s")
    
    return results, X_train, X_test, y_train, y_test

def optimize_best_model(best_model, X_train, X_test, y_train, y_test):
    """Optimize the best model using RandomizedSearchCV (more efficient than GridSearchCV)"""
    print("\nOptimizing RandomForest with RandomizedSearchCV...")
    
    # Reduced parameter space for faster search
    param_distributions = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Use RandomizedSearchCV instead of GridSearchCV for better efficiency
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=10,  # Only test 10 combinations instead of all
        cv=3,       # Reduce cross-validation folds
        random_state=42,
        n_jobs=-1   # Use all available CPU cores
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    optimization_time = time.time() - start_time
    
    best_optimized = random_search.best_estimator_
    best_score = best_optimized.score(X_test, y_test)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Optimized score: {best_score:.4f}")
    print(f"Optimization time: {optimization_time:.2f}s")
    
    return best_optimized

def save_model(model, filename='optimized_predictor.pickle'):
    """Save the trained model"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def main():
    """Main execution function"""
    print("Starting optimized laptop price prediction model...")
    
    # Load and preprocess data
    start_time = time.time()
    dataset = load_and_preprocess_data()
    preprocessing_time = time.time() - start_time
    print(f"Data preprocessing completed in {preprocessing_time:.2f}s")
    
    # Prepare features and target
    X = dataset.drop('Price_euros', axis=1)
    y = dataset['Price_euros']
    
    # Train models
    results, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Find best model
    best_name = max(results, key=lambda k: results[k]['score'])
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name} with score {results[best_name]['score']:.4f}")
    
    # Optimize the RandomForest (typically the best performer)
    if isinstance(best_model, RandomForestRegressor):
        optimized_model = optimize_best_model(best_model, X_train, X_test, y_train, y_test)
    else:
        optimized_model = best_model
    
    # Save the model
    save_model(optimized_model)
    
    # Test predictions
    print("\nSample predictions:")
    sample_data = [[8, 1.4, 1, 1, 0, 1] + [0] * (X.shape[1] - 6)]  # Adjust to match feature count
    if len(sample_data[0]) == X.shape[1]:
        prediction = optimized_model.predict(sample_data)
        print(f"Sample prediction: {prediction[0]:.2f} euros")
    
    return optimized_model, results

if __name__ == "__main__":
    model, results = main()
