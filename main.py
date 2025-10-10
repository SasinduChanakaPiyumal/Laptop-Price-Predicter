#!/usr/bin/env python
"""
Main script for laptop price prediction model.
Orchestrates the entire machine learning pipeline from data loading to model training.

Usage:
    python main.py [--skip-tuning] [--verbose]
"""

import argparse
import sys
from data_loader import load_data, explore_data, analyze_correlations
from feature_engineering import preprocess_dataset, prepare_features
from model_training import (
    split_data, 
    train_baseline_models,
    tune_random_forest,
    tune_gradient_boosting,
    select_best_model,
    analyze_feature_importance,
    demonstrate_predictions,
    save_model
)
import config


def main(skip_tuning=False, verbose=True):
    """
    Execute the complete machine learning pipeline.
    
    Args:
        skip_tuning (bool): If True, skip hyperparameter tuning
        verbose (bool): If True, display detailed information
    """
    print("="*60)
    print("LAPTOP PRICE PREDICTION MODEL")
    print("="*60)
    
    # Step 1: Load and explore data
    print("\n[1/6] Loading data...")
    dataset = load_data()
    
    if verbose:
        explore_data(dataset, verbose=True)
        analyze_correlations(dataset)
    
    # Step 2: Preprocess and engineer features
    print("\n[2/6] Preprocessing and feature engineering...")
    dataset = preprocess_dataset(dataset)
    
    # Step 3: Prepare features for modeling
    print("\n[3/6] Preparing features...")
    x, y = prepare_features(dataset)
    
    if verbose:
        print(f"\nFinal dataset shape: {x.shape}")
        print(f"Number of features: {x.shape[1]}")
        print(f"Number of samples: {x.shape[0]}")
    
    # Step 4: Split data
    print("\n[4/6] Splitting data into train and test sets...")
    x_train, x_test, y_train, y_test = split_data(x, y)
    
    # Step 5: Train and compare models
    print("\n[5/6] Training models...")
    baseline_results = train_baseline_models(x_train, y_train, x_test, y_test)
    
    if skip_tuning:
        print("\n⚠ Skipping hyperparameter tuning (use default random forest)")
        from sklearn.ensemble import RandomForestRegressor
        best_model = RandomForestRegressor(
            n_estimators=200,
            random_state=config.RANDOM_STATE
        )
        best_model.fit(x_train, y_train)
        best_name = "Random Forest (default)"
    else:
        print("\n[5a] Tuning Random Forest...")
        rf_model, rf_params, rf_score = tune_random_forest(x_train, y_train)
        
        print("\n[5b] Tuning Gradient Boosting...")
        gb_model, gb_params, gb_score = tune_gradient_boosting(x_train, y_train)
        
        # Step 6: Select best model
        print("\n[6/6] Selecting best model...")
        best_model, best_name, metrics = select_best_model(
            rf_model, gb_model, x_train, y_train, x_test, y_test
        )
    
    # Feature importance analysis
    analyze_feature_importance(best_model, x_train.columns)
    
    # Final model performance
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE")
    print("="*60)
    final_score = best_model.score(x_test, y_test)
    print(f"Test R² Score: {final_score:.4f}")
    print(f"Model: {best_name}")
    
    # Demonstrate predictions
    demonstrate_predictions(best_model, x_test, y_test, n_samples=5)
    
    # Save the model
    save_model(best_model)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n✓ Model saved to: {config.MODEL_OUTPUT_FILE}")
    print(f"✓ Final R² Score: {final_score:.4f}")
    print(f"✓ Model type: {best_name}")
    
    return best_model, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train laptop price prediction model"
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning for faster execution'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    try:
        main(skip_tuning=args.skip_tuning, verbose=not args.no_verbose)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
