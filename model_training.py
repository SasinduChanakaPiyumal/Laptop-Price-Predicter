"""
Model training, evaluation, and hyperparameter tuning module.
Handles training multiple models and selecting the best one.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import config


def split_data(x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Split data into training and testing sets.
    
    Args:
        x (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Data split complete:")
    print(f"  - Training set: {x_train.shape[0]} samples")
    print(f"  - Testing set: {x_test.shape[0]} samples")
    
    return x_train, x_test, y_train, y_test


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name="Model"):
    """
    Evaluate a trained model using multiple metrics.
    
    Args:
        model: Trained scikit-learn model
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        x_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(
        model, x_train, y_train, 
        cv=config.CV_FOLDS, 
        scoring='r2'
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Display results
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} euros")
    print(f"  RMSE: {rmse:.2f} euros")
    print(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return {
        'model_name': model_name,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }


def train_baseline_models(x_train, y_train, x_test, y_test):
    """
    Train and evaluate multiple baseline models.
    
    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        x_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(random_state=config.RANDOM_STATE),
        'Decision Tree': DecisionTreeRegressor(random_state=config.RANDOM_STATE),
        'Random Forest': RandomForestRegressor(random_state=config.RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(random_state=config.RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test, name)
        results[name] = {
            'model': model,
            'metrics': metrics
        }
    
    # Try XGBoost if available
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            random_state=config.RANDOM_STATE, 
            objective='reg:squarederror'
        )
        xgb_model.fit(x_train, y_train)
        metrics = evaluate_model(xgb_model, x_train, y_train, x_test, y_test, "XGBoost")
        results['XGBoost'] = {
            'model': xgb_model,
            'metrics': metrics
        }
    except ImportError:
        print("\n⚠ XGBoost not installed. Install with: pip install xgboost")
    
    return results


def tune_random_forest(x_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest.
    
    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - Random Forest")
    print("="*60)
    
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=config.RANDOM_STATE),
        param_distributions=config.RF_PARAM_GRID,
        n_iter=config.N_ITER_RANDOM_SEARCH,
        cv=config.CV_FOLDS,
        scoring='r2',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n⏳ Training Random Forest with RandomizedSearchCV...")
    rf_search.fit(x_train, y_train)
    
    print(f"\n✓ Best parameters: {rf_search.best_params_}")
    print(f"✓ Best CV score: {rf_search.best_score_:.4f}")
    
    return rf_search.best_estimator_, rf_search.best_params_, rf_search.best_score_


def tune_gradient_boosting(x_train, y_train):
    """
    Perform hyperparameter tuning for Gradient Boosting.
    
    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - Gradient Boosting")
    print("="*60)
    
    gb_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        param_distributions=config.GB_PARAM_GRID,
        n_iter=config.N_ITER_RANDOM_SEARCH,
        cv=config.CV_FOLDS,
        scoring='r2',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n⏳ Training Gradient Boosting with RandomizedSearchCV...")
    gb_search.fit(x_train, y_train)
    
    print(f"\n✓ Best parameters: {gb_search.best_params_}")
    print(f"✓ Best CV score: {gb_search.best_score_:.4f}")
    
    return gb_search.best_estimator_, gb_search.best_params_, gb_search.best_score_


def select_best_model(rf_model, gb_model, x_train, y_train, x_test, y_test):
    """
    Compare tuned models and select the best one.
    
    Args:
        rf_model: Tuned Random Forest model
        gb_model: Tuned Gradient Boosting model
        x_train, y_train: Training data
        x_test, y_test: Testing data
        
    Returns:
        tuple: (best_model, model_name, metrics)
    """
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    rf_metrics = evaluate_model(rf_model, x_train, y_train, x_test, y_test, 
                                 "Best Random Forest")
    gb_metrics = evaluate_model(gb_model, x_train, y_train, x_test, y_test, 
                                 "Best Gradient Boosting")
    
    # Select based on R² score
    if gb_metrics['r2'] > rf_metrics['r2']:
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_metrics = gb_metrics
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_metrics
    
    print(f"\n{'*'*60}")
    print(f"WINNER: {best_name}")
    print(f"{'*'*60}")
    
    return best_model, best_name, best_metrics


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and display feature importance.
    
    Args:
        model: Trained tree-based model with feature_importances_
        feature_names: List of feature names
        top_n (int): Number of top features to display
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance.head(top_n).to_string(index=False))
    
    return feature_importance


def save_model(model, filepath=config.MODEL_OUTPUT_FILE):
    """
    Save the trained model to a pickle file.
    
    Args:
        model: Trained model to save
        filepath (str): Output file path
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"\n✓ Model saved to {filepath}")


def demonstrate_predictions(model, x_test, y_test, n_samples=5):
    """
    Demonstrate model predictions on test samples.
    
    Args:
        model: Trained model
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        n_samples (int): Number of samples to demonstrate
    """
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    predictions = model.predict(x_test[:n_samples])
    actuals = y_test.iloc[:n_samples].values
    
    print(f"\nPredictions for first {n_samples} test samples:")
    for i, (pred, actual) in enumerate(zip(predictions, actuals), 1):
        error = abs(pred - actual)
        error_pct = (error / actual) * 100
        print(f"  Sample {i}: Predicted={pred:.2f}€, Actual={actual:.2f}€, "
              f"Error={error:.2f}€ ({error_pct:.1f}%)")


if __name__ == "__main__":
    # Test the training pipeline
    from data_loader import load_data
    from feature_engineering import preprocess_dataset, prepare_features
    
    print("Loading and preparing data...")
    dataset = load_data()
    dataset = preprocess_dataset(dataset)
    x, y = prepare_features(dataset)
    
    print("\nSplitting data...")
    x_train, x_test, y_train, y_test = split_data(x, y)
    
    print("\nTraining baseline models...")
    baseline_results = train_baseline_models(x_train, y_train, x_test, y_test)
    
    print("\nTuning models...")
    rf_model, _, _ = tune_random_forest(x_train, y_train)
    gb_model, _, _ = tune_gradient_boosting(x_train, y_train)
    
    print("\nSelecting best model...")
    best_model, best_name, metrics = select_best_model(
        rf_model, gb_model, x_train, y_train, x_test, y_test
    )
    
    analyze_feature_importance(best_model, x_train.columns)
    demonstrate_predictions(best_model, x_test, y_test)
    save_model(best_model)
