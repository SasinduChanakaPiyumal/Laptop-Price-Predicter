# Laptop Price Prediction Model

A machine learning project to predict laptop prices based on their specifications.

## Project Structure

The original monolithic script has been refactored into modular components for better maintainability and readability:

```
.
├── config.py                      # Configuration constants and parameters
├── data_loader.py                 # Data loading and exploration
├── feature_engineering.py         # Preprocessing and feature extraction
├── model_training.py              # Model training and evaluation
├── main.py                        # Main orchestration script
├── laptop_price.csv               # Dataset
├── Laptop Price model(1).py       # Original monolithic script (preserved)
└── predictor.pickle               # Saved trained model (generated)
```

## Module Descriptions

### 1. `config.py`
Contains all configuration constants, file paths, and hyperparameters:
- File paths and encodings
- Data processing constants
- Feature engineering parameters
- Model training parameters
- Hyperparameter tuning grids

### 2. `data_loader.py`
Handles data loading and initial exploration:
- `load_data()`: Load dataset from CSV
- `explore_data()`: Perform exploratory data analysis
- `analyze_correlations()`: Analyze feature correlations with target

### 3. `feature_engineering.py`
Manages all data preprocessing and feature extraction:
- `clean_ram_column()`: Clean and convert RAM to numeric
- `clean_weight_column()`: Clean and convert Weight to numeric
- `group_companies()`: Categorize companies
- `extract_cpu_features()`: Extract CPU information
- `extract_gpu_features()`: Extract GPU information
- `extract_screen_features()`: Extract screen resolution details
- `categorize_operating_system()`: Categorize OS
- `create_interaction_features()`: Create feature interactions
- `preprocess_dataset()`: Complete preprocessing pipeline
- `prepare_features()`: Final feature preparation with encoding

### 4. `model_training.py`
Handles model training, evaluation, and selection:
- `split_data()`: Split into train/test sets
- `evaluate_model()`: Comprehensive model evaluation
- `train_baseline_models()`: Train multiple baseline models
- `tune_random_forest()`: Hyperparameter tuning for Random Forest
- `tune_gradient_boosting()`: Hyperparameter tuning for Gradient Boosting
- `select_best_model()`: Compare and select best model
- `analyze_feature_importance()`: Feature importance analysis
- `save_model()`: Save trained model to disk
- `demonstrate_predictions()`: Show sample predictions

### 5. `main.py`
Orchestrates the entire pipeline:
- Coordinates all modules
- Provides command-line interface
- Executes complete ML pipeline

## Usage

### Basic Usage
```bash
python main.py
```

### Skip Hyperparameter Tuning (faster execution)
```bash
python main.py --skip-tuning
```

### Reduce Verbosity
```bash
python main.py --no-verbose
```

### Use Individual Modules
Each module can be run independently for testing:
```bash
# Test data loading
python data_loader.py

# Test feature engineering
python feature_engineering.py

# Test model training
python model_training.py
```

## Features

### Data Preprocessing
- RAM and Weight column cleaning
- Company categorization
- CPU and GPU brand extraction
- Operating system categorization
- Screen resolution feature extraction (Touchscreen, IPS, PPI)

### Feature Engineering
- Screen quality metrics (Width, Height, Total Pixels, PPI)
- Interaction features (RAM squared, Screen Quality)
- One-hot encoding for categorical variables

### Model Training
- Multiple baseline models:
  - Linear Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost (optional)
- Hyperparameter tuning with RandomizedSearchCV
- Cross-validation for robust evaluation
- Multiple metrics: R², MAE, RMSE

### Model Evaluation
- Train/test split with reproducibility
- 5-fold cross-validation
- Feature importance analysis
- Sample prediction demonstrations

## Requirements

```bash
pip install pandas numpy scikit-learn
# Optional for XGBoost support
pip install xgboost
```

## Benefits of Refactoring

### Improved Readability
- Clear separation of concerns
- Focused modules with single responsibilities
- Better code organization

### Enhanced Maintainability
- Easy to locate and modify specific functionality
- Changes isolated to relevant modules
- Reduced code duplication

### Better Reusability
- Functions can be imported and reused
- Modular design allows mix-and-match
- Easy to extend with new features

### Easier Testing
- Each module can be tested independently
- Mock dependencies for unit testing
- Better error isolation

### Flexibility
- Command-line arguments for different execution modes
- Easy to add new models or features
- Configuration separated from logic

## Output

The pipeline generates:
- Trained model saved as `predictor.pickle`
- Comprehensive evaluation metrics
- Feature importance analysis
- Sample predictions on test data

## Notes

- Original monolithic script (`Laptop Price model(1).py`) is preserved for reference
- All hyperparameters are configurable in `config.py`
- Random state is set for reproducibility
- The best model (Random Forest or Gradient Boosting) is automatically selected based on performance
