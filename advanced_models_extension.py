#!/usr/bin/env python
# coding: utf-8

"""
Advanced Models Extension for Laptop Price Prediction
==================================================
Adds XGBoost, LightGBM, and advanced ensemble techniques
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Advanced Models Extension for Laptop Price Prediction")
print("=" * 55)

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available (pip install xgboost)")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available (pip install lightgbm)")

def load_preprocessed_data():
    """
    Load the preprocessed data from the main improved model
    This assumes the main script has been run and data is available
    """
    # This is a placeholder - in practice, you'd want to run the main script first
    # or create a separate data loading function
    print("Loading preprocessed data...")
    
    # Load dataset and apply the same preprocessing as the main script
    dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
    
    # Apply the same preprocessing steps as in the main script
    # (This is a simplified version - the full preprocessing is in the main script)
    df = dataset.copy()
    
    # Basic preprocessing (simplified for this extension)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float64')
    
    # Basic feature engineering
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    
    # Simple company grouping
    def add_company(inpt):
        minor_brands = ['Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 
                       'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei']
        return 'Other' if inpt in minor_brands else inpt
    
    df['Company'] = df['Company'].apply(add_company)
    
    # Simple CPU processing
    df['Cpu_name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    def set_processor(name):
        if name in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
            return name
        elif name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
    df['Cpu_name'] = df['Cpu_name'].apply(set_processor)
    
    # Simple GPU processing
    df['Gpu_name'] = df['Gpu'].apply(lambda x: x.split()[0])
    
    # Simple OS processing
    def set_os(inpt):
        if inpt in ['Windows 10', 'Windows 7', 'Windows 10 S']:
            return 'Windows'
        elif inpt in ['macOS', 'Mac OS X']:
            return 'Mac'
        elif inpt == 'Linux':
            return 'Linux'
        else:
            return 'Other'
    df['OpSys'] = df['OpSys'].apply(set_os)
    
    # Drop unnecessary columns
    df = df.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
    
    # One-hot encoding
    df = pd.get_dummies(df)
    
    # Separate features and target
    X = df.drop('Price_euros', axis=1)
    y = df['Price_euros']
    
    return X, y

def evaluate_advanced_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive evaluation for advanced models"""
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean = test_r2
        cv_std = 0.0
    
    results = {
        'Model': model_name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'Test_MAPE': test_mape,
        'Overfitting': train_r2 - test_r2
    }
    
    return results, model

def train_xgboost_models(X_train, X_test, y_train, y_test):
    """Train and tune XGBoost models"""
    print("\n=== XGBoost Model Training ===")
    
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping...")
        return {}
    
    # Basic XGBoost
    xgb_basic = xgb.XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='reg:squarederror'
    )
    
    print("Training basic XGBoost...")
    xgb_basic_results, xgb_basic_model = evaluate_advanced_model(
        xgb_basic, X_train, X_test, y_train, y_test, 'XGBoost_Basic'
    )
    
    # Advanced XGBoost with hyperparameter tuning
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5]
    }
    
    xgb_random_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
        xgb_param_grid,
        n_iter=30,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("Performing XGBoost hyperparameter tuning...")
    xgb_random_search.fit(X_train, y_train)
    best_xgb = xgb_random_search.best_estimator_
    
    print(f"Best XGB parameters: {xgb_random_search.best_params_}")
    print(f"Best XGB CV score: {xgb_random_search.best_score_:.4f}")
    
    xgb_tuned_results, xgb_tuned_model = evaluate_advanced_model(
        best_xgb, X_train, X_test, y_train, y_test, 'XGBoost_Tuned'
    )
    
    return {
        'XGBoost_Basic': (xgb_basic_results, xgb_basic_model),
        'XGBoost_Tuned': (xgb_tuned_results, xgb_tuned_model)
    }

def train_lightgbm_models(X_train, X_test, y_train, y_test):
    """Train and tune LightGBM models"""
    print("\n=== LightGBM Model Training ===")
    
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available, skipping...")
        return {}
    
    # Basic LightGBM
    lgb_basic = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='regression',
        verbosity=-1
    )
    
    print("Training basic LightGBM...")
    lgb_basic_results, lgb_basic_model = evaluate_advanced_model(
        lgb_basic, X_train, X_test, y_train, y_test, 'LightGBM_Basic'
    )
    
    # Advanced LightGBM with hyperparameter tuning
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5],
        'num_leaves': [20, 31, 40]
    }
    
    lgb_random_search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=42, objective='regression', verbosity=-1),
        lgb_param_grid,
        n_iter=30,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("Performing LightGBM hyperparameter tuning...")
    lgb_random_search.fit(X_train, y_train)
    best_lgb = lgb_random_search.best_estimator_
    
    print(f"Best LGB parameters: {lgb_random_search.best_params_}")
    print(f"Best LGB CV score: {lgb_random_search.best_score_:.4f}")
    
    lgb_tuned_results, lgb_tuned_model = evaluate_advanced_model(
        best_lgb, X_train, X_test, y_train, y_test, 'LightGBM_Tuned'
    )
    
    return {
        'LightGBM_Basic': (lgb_basic_results, lgb_basic_model),
        'LightGBM_Tuned': (lgb_tuned_results, lgb_tuned_model)
    }

def create_advanced_ensembles(models, X_train, X_test, y_train, y_test):
    """Create advanced ensemble methods"""
    print("\n=== Advanced Ensemble Methods ===")
    
    ensemble_results = {}
    
    # Extract the actual model objects
    model_objects = {}
    for name, (results, model) in models.items():
        model_objects[name] = model
    
    if len(model_objects) < 2:
        print("Not enough models for ensemble creation")
        return ensemble_results
    
    # Voting Regressor with all available models
    estimators = [(name, model) for name, model in model_objects.items()]
    
    voting_ensemble = VotingRegressor(estimators=estimators)
    
    print("Training Voting Ensemble...")
    voting_results, voting_model = evaluate_advanced_model(
        voting_ensemble, X_train, X_test, y_train, y_test, 'Advanced_Voting_Ensemble'
    )
    ensemble_results['Advanced_Voting_Ensemble'] = (voting_results, voting_model)
    
    # Stacking Regressor
    if len(model_objects) >= 2:
        # Use top 3 models as base estimators
        base_models = list(model_objects.items())[:3]
        
        stacking_ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
        
        print("Training Stacking Ensemble...")
        stacking_results, stacking_model = evaluate_advanced_model(
            stacking_ensemble, X_train, X_test, y_train, y_test, 'Stacking_Ensemble'
        )
        ensemble_results['Stacking_Ensemble'] = (stacking_results, stacking_model)
    
    return ensemble_results

def main():
    """Main function to run advanced model training"""
    print("Starting Advanced Model Training Pipeline...")
    
    # Load data
    X, y = load_preprocessed_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize results collection
    all_results = []
    all_models = {}
    
    # Train XGBoost models
    xgb_models = train_xgboost_models(X_train, X_test, y_train, y_test)
    all_models.update(xgb_models)
    
    for name, (results, model) in xgb_models.items():
        all_results.append(results)
    
    # Train LightGBM models
    lgb_models = train_lightgbm_models(X_train, X_test, y_train, y_test)
    all_models.update(lgb_models)
    
    for name, (results, model) in lgb_models.items():
        all_results.append(results)
    
    # Create advanced ensembles
    ensemble_models = create_advanced_ensembles(all_models, X_train, X_test, y_train, y_test)
    all_models.update(ensemble_models)
    
    for name, (results, model) in ensemble_models.items():
        all_results.append(results)
    
    # Create final results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        print("\n=== ADVANCED MODELS PERFORMANCE COMPARISON ===")
        print(results_df.round(4))
        
        # Save best advanced model
        if not results_df.empty:
            best_model_name = results_df.iloc[0]['Model']
            best_model = all_models[best_model_name][1]
            
            print(f"\nBest Advanced Model: {best_model_name}")
            print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
            print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
            print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.2f}")
            
            # Save the best advanced model
            with open('best_advanced_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            
            print(f"\nBest advanced model saved as 'best_advanced_model.pkl'")
            
            # Save results
            results_df.to_csv('advanced_models_results.csv', index=False)
            print("Results saved as 'advanced_models_results.csv'")
    else:
        print("No advanced models were successfully trained.")
        print("Please install xgboost and lightgbm: pip install xgboost lightgbm")

    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
