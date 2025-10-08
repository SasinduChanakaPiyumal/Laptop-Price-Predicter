#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Laptop Price Prediction Model Comparison Demo
============================================

This script demonstrates the improvements made to the laptop price prediction model
by comparing the original implementation with the enhanced version.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def original_approach():
    """
    Replicate the original model approach for comparison.
    """
    print("="*60)
    print("ORIGINAL MODEL APPROACH")
    print("="*60)
    
    # Load data
    dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
    print(f"Original dataset shape: {dataset.shape}")
    
    # Basic preprocessing (from original code)
    dataset['Ram'] = dataset['Ram'].str.replace('GB','').astype('int32')
    dataset['Weight'] = dataset['Weight'].str.replace('kg','').astype('float64')
    
    # Basic feature extraction
    dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
    dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
    
    # Basic CPU processing
    dataset['Cpu_name'] = dataset['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    
    def set_processor(name):
        if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
            return name
        else:
            if name.split()[0] == 'AMD':
                return 'AMD'
            else:
                return 'Other'
    dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)
    
    # Basic GPU processing
    dataset['Gpu_name'] = dataset['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
    dataset = dataset[dataset['Gpu_name'] != 'ARM']
    
    # Basic OS processing
    def set_os(inpt):
        if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
            return 'Windows'
        elif inpt == 'macOS' or inpt == 'Mac OS X':
            return 'Mac'
        elif inpt == 'Linux':
            return inpt
        else:
            return 'Other'
    dataset['OpSys'] = dataset['OpSys'].apply(set_os)
    
    # Drop original columns
    dataset = dataset.drop(columns=['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu'])
    
    # One-hot encoding
    dataset = pd.get_dummies(dataset)
    
    # Split features and target
    X = dataset.drop('Price_euros', axis=1)
    y = dataset['Price_euros']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train basic Random Forest (similar to original)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nOriginal Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} euros")
    print(f"MAE: {mae:.2f} euros")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'num_features': X.shape[1],
        'model': rf
    }

def enhanced_approach():
    """
    Use the enhanced model approach.
    """
    print("\n" + "="*60)
    print("ENHANCED MODEL APPROACH")
    print("="*60)
    
    # Import the enhanced predictor
    from enhanced_laptop_price_model import EnhancedLaptopPricePredictor
    
    # Initialize and train the enhanced model
    predictor = EnhancedLaptopPricePredictor(random_state=42)
    
    try:
        results = predictor.fit(
            filepath='laptop_price.csv',
            test_size=0.25,  # Same as original for fair comparison
            optimize_hyperparams=True
        )
        
        print(f"Best Model: {results['best_model_name']}")
        print(f"Number of features: {results['data_info']['final_features']}")
        print(f"Training samples: {results['data_info']['training_samples']}")
        print(f"Test samples: {results['data_info']['test_samples']}")
        print(f"\nEnhanced Model Performance:")
        print(f"R² Score: {results['final_metrics']['r2']:.4f}")
        print(f"RMSE: {results['final_metrics']['rmse']:.2f} euros")
        print(f"MAE: {results['final_metrics']['mae']:.2f} euros")
        
        return {
            'r2': results['final_metrics']['r2'],
            'rmse': results['final_metrics']['rmse'],
            'mae': results['final_metrics']['mae'],
            'num_features': results['data_info']['final_features'],
            'best_model': results['best_model_name'],
            'feature_importance': results['feature_importance']
        }
        
    except Exception as e:
        print(f"Error in enhanced approach: {str(e)}")
        return None

def compare_results(original_results, enhanced_results):
    """
    Compare the results from both approaches.
    """
    if enhanced_results is None:
        print("Enhanced model failed to run, skipping comparison.")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"{'Metric':<15} {'Original':<12} {'Enhanced':<12} {'Improvement':<15}")
    print("-" * 60)
    
    # R² Score comparison
    r2_improvement = ((enhanced_results['r2'] - original_results['r2']) / original_results['r2']) * 100
    print(f"{'R² Score':<15} {original_results['r2']:<12.4f} {enhanced_results['r2']:<12.4f} {r2_improvement:>+13.2f}%")
    
    # RMSE comparison (lower is better, so calculate improvement differently)
    rmse_improvement = ((original_results['rmse'] - enhanced_results['rmse']) / original_results['rmse']) * 100
    print(f"{'RMSE (euros)':<15} {original_results['rmse']:<12.2f} {enhanced_results['rmse']:<12.2f} {rmse_improvement:>+13.2f}%")
    
    # MAE comparison (lower is better)
    mae_improvement = ((original_results['mae'] - enhanced_results['mae']) / original_results['mae']) * 100
    print(f"{'MAE (euros)':<15} {original_results['mae']:<12.2f} {enhanced_results['mae']:<12.2f} {mae_improvement:>+13.2f}%")
    
    # Feature count comparison
    feature_change = enhanced_results['num_features'] - original_results['num_features']
    print(f"{'Features':<15} {original_results['num_features']:<12} {enhanced_results['num_features']:<12} {feature_change:>+13}")
    
    print(f"\nBest Enhanced Model: {enhanced_results['best_model']}")
    
    # Feature importance
    if enhanced_results['feature_importance'] is not None:
        print(f"\nTop 5 Most Important Features:")
        for i, row in enhanced_results['feature_importance'].head(5).iterrows():
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Summary
    print(f"\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    
    improvements = []
    if r2_improvement > 0:
        improvements.append(f"R² Score improved by {r2_improvement:.2f}%")
    if rmse_improvement > 0:
        improvements.append(f"RMSE reduced by {rmse_improvement:.2f}%")
    if mae_improvement > 0:
        improvements.append(f"MAE reduced by {mae_improvement:.2f}%")
    
    if improvements:
        print("Key Improvements:")
        for improvement in improvements:
            print(f"✓ {improvement}")
    else:
        print("No significant improvements detected.")
    
    print(f"\nEnhanced model uses {enhanced_results['num_features']} features vs {original_results['num_features']} in original")
    print(f"Model architecture: {enhanced_results['best_model']} (vs Random Forest in original)")

def main():
    """
    Main function to run the comparison.
    """
    print("Laptop Price Prediction Model Comparison")
    print("This script compares the original and enhanced approaches")
    print("")
    
    try:
        # Run original approach
        original_results = original_approach()
        
        # Run enhanced approach  
        enhanced_results = enhanced_approach()
        
        # Compare results
        compare_results(original_results, enhanced_results)
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        print("Make sure the laptop_price.csv file is available in the current directory.")

if __name__ == "__main__":
    main()
