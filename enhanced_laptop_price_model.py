#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Laptop Price Prediction Model
=====================================

This module contains improved machine learning implementations for laptop price prediction,
including advanced feature engineering, multiple model architectures, and comprehensive
hyperparameter optimization.

Key Improvements:
- Advanced feature engineering with scaling and feature selection
- Multiple model architectures including XGBoost and LightGBM
- Comprehensive hyperparameter tuning with cross-validation
- Detailed model evaluation with multiple metrics
- Feature importance analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Utility libraries
import pickle
import re
from scipy import stats
from scipy.stats import boxcox
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedLaptopPricePredictor:
    """
    Enhanced laptop price prediction model with comprehensive feature engineering 
    and multiple model architectures.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.label_encoders = {}
        
    def load_and_prepare_data(self, filepath='laptop_price.csv'):
        """
        Load and perform initial data preparation.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Prepared dataset
        """
        logging.info("Loading dataset...")
        
        # Load data with proper encoding
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath, encoding='utf-8')
            
        logging.info(f"Dataset loaded with shape: {df.shape}")
        
        # Initial data exploration
        logging.info("Dataset Info:")
        logging.info(f"Columns: {list(df.columns)}")
        logging.info(f"Missing values: \n{df.isnull().sum()}")
        
        return df
    
    def advanced_feature_engineering(self, df):
        """
        Perform comprehensive feature engineering with improved techniques.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe with engineered features
        """
        logging.info("Starting advanced feature engineering...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # 1. Clean and convert basic numeric features
        df_processed['Ram'] = df_processed['Ram'].str.replace('GB', '').astype('int32')
        df_processed['Weight'] = df_processed['Weight'].str.replace('kg', '').astype('float64')
        
        # 2. Advanced screen resolution feature extraction
        logging.info("Processing screen resolution features...")
        
        # Extract screen resolution dimensions
        def extract_resolution(res_str):
            if pd.isna(res_str):
                return 0, 0, 0
            # Extract width x height resolution
            resolution_match = re.search(r'(\d+)x(\d+)', res_str)
            if resolution_match:
                width, height = int(resolution_match.group(1)), int(resolution_match.group(2))
                total_pixels = width * height
                aspect_ratio = width / height if height > 0 else 1
                return total_pixels, width, height
            return 0, 0, 0
        
        resolution_features = df_processed['ScreenResolution'].apply(extract_resolution)
        df_processed['Screen_Total_Pixels'] = [x[0] for x in resolution_features]
        df_processed['Screen_Width'] = [x[1] for x in resolution_features]
        df_processed['Screen_Height'] = [x[2] for x in resolution_features]
        
        # Calculate PPI (Pixels Per Inch) - more accurate than just resolution
        df_processed['PPI'] = np.sqrt(df_processed['Screen_Width']**2 + df_processed['Screen_Height']**2) / df_processed['Inches']
        df_processed['PPI'].fillna(df_processed['PPI'].median(), inplace=True)
        
        # Screen features
        df_processed['Touchscreen'] = df_processed['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
        df_processed['IPS'] = df_processed['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)
        df_processed['Retina'] = df_processed['ScreenResolution'].apply(lambda x: 1 if 'Retina' in str(x) else 0)
        df_processed['4K'] = df_processed['ScreenResolution'].apply(lambda x: 1 if '4K' in str(x) else 0)
        
        # 3. Enhanced CPU processing
        logging.info("Processing CPU features...")
        
        # Extract CPU brand and generation more intelligently
        def extract_cpu_features(cpu_str):
            if pd.isna(cpu_str):
                return 'Other', 'Unknown', 0, 0
            
            cpu_str = str(cpu_str).lower()
            brand = 'Other'
            model = 'Unknown'
            generation = 0
            cores = 0
            
            if 'intel' in cpu_str:
                brand = 'Intel'
                if 'i9' in cpu_str:
                    model = 'i9'
                elif 'i7' in cpu_str:
                    model = 'i7'
                elif 'i5' in cpu_str:
                    model = 'i5'
                elif 'i3' in cpu_str:
                    model = 'i3'
                
                # Extract generation (e.g., 8th gen, 10th gen)
                gen_match = re.search(r'(\d+)th', cpu_str)
                if gen_match:
                    generation = int(gen_match.group(1))
                    
            elif 'amd' in cpu_str:
                brand = 'AMD'
                if 'ryzen' in cpu_str:
                    model = 'Ryzen'
                
            # Extract number of cores if mentioned
            cores_match = re.search(r'(\d+)\s*core', cpu_str)
            if cores_match:
                cores = int(cores_match.group(1))
            
            return brand, model, generation, cores
        
        cpu_features = df_processed['Cpu'].apply(extract_cpu_features)
        df_processed['CPU_Brand'] = [x[0] for x in cpu_features]
        df_processed['CPU_Model'] = [x[1] for x in cpu_features]
        df_processed['CPU_Generation'] = [x[2] for x in cpu_features]
        df_processed['CPU_Cores'] = [x[3] for x in cpu_features]
        
        # 4. Enhanced GPU processing
        logging.info("Processing GPU features...")
        
        def categorize_gpu(gpu_str):
            if pd.isna(gpu_str):
                return 'Other', 'Unknown', 0
            
            gpu_str = str(gpu_str).lower()
            brand = 'Other'
            series = 'Unknown'
            memory = 0
            
            if 'nvidia' in gpu_str or 'geforce' in gpu_str:
                brand = 'NVIDIA'
                if 'rtx' in gpu_str:
                    series = 'RTX'
                elif 'gtx' in gpu_str:
                    series = 'GTX'
                elif 'quadro' in gpu_str:
                    series = 'Quadro'
            elif 'amd' in gpu_str or 'radeon' in gpu_str:
                brand = 'AMD'
                series = 'Radeon'
            elif 'intel' in gpu_str:
                brand = 'Intel'
                series = 'Integrated'
            
            # Extract GPU memory
            memory_match = re.search(r'(\d+)\s*gb', gpu_str)
            if memory_match:
                memory = int(memory_match.group(1))
            
            return brand, series, memory
        
        gpu_features = df_processed['Gpu'].apply(categorize_gpu)
        df_processed['GPU_Brand'] = [x[0] for x in gpu_features]
        df_processed['GPU_Series'] = [x[1] for x in gpu_features]
        df_processed['GPU_Memory'] = [x[2] for x in gpu_features]
        
        # 5. Enhanced company grouping based on market position
        def categorize_company(company):
            premium_brands = ['Apple', 'Microsoft', 'Razer', 'Alienware']
            mainstream_brands = ['HP', 'Dell', 'Lenovo', 'Asus', 'Acer']
            budget_brands = ['Toshiba', 'Samsung', 'Mediacom', 'Chuwi']
            
            if company in premium_brands:
                return 'Premium'
            elif company in mainstream_brands:
                return 'Mainstream'
            elif company in budget_brands:
                return 'Budget'
            else:
                return 'Other'
        
        df_processed['Company_Category'] = df_processed['Company'].apply(categorize_company)
        
        # 6. Operating System enhancement
        def categorize_os(os_str):
            if pd.isna(os_str):
                return 'Other'
            
            os_str = str(os_str).lower()
            if 'windows' in os_str:
                return 'Windows'
            elif 'mac' in os_str:
                return 'macOS'
            elif 'linux' in os_str:
                return 'Linux'
            elif 'chrome' in os_str:
                return 'Chrome OS'
            else:
                return 'Other'
        
        df_processed['OS_Category'] = df_processed['OpSys'].apply(categorize_os)
        
        # 7. Create interaction features
        logging.info("Creating interaction features...")
        
        # Performance ratios and combinations
        df_processed['RAM_per_Weight'] = df_processed['Ram'] / df_processed['Weight']
        df_processed['PPI_per_Inch'] = df_processed['PPI'] / df_processed['Inches']
        df_processed['Screen_Area'] = df_processed['Inches'] ** 2 * 0.785  # Approximate screen area
        df_processed['Pixel_Density_Score'] = df_processed['Screen_Total_Pixels'] / df_processed['Screen_Area']
        
        # Performance indicators
        df_processed['High_End_CPU'] = ((df_processed['CPU_Model'] == 'i7') | 
                                       (df_processed['CPU_Model'] == 'i9') |
                                       (df_processed['CPU_Model'] == 'Ryzen')).astype(int)
        
        df_processed['Gaming_GPU'] = ((df_processed['GPU_Series'] == 'RTX') | 
                                     (df_processed['GPU_Series'] == 'GTX')).astype(int)
        
        df_processed['High_RAM'] = (df_processed['Ram'] >= 16).astype(int)
        df_processed['Large_Screen'] = (df_processed['Inches'] >= 15).astype(int)
        df_processed['Lightweight'] = (df_processed['Weight'] <= 2.0).astype(int)
        
        # 8. Clean up and prepare for modeling
        # Remove rows with ARM GPUs (outliers)
        df_processed = df_processed[~df_processed['Gpu'].str.contains('ARM', na=False)]
        
        # Drop original columns that have been processed
        columns_to_drop = ['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        
        logging.info(f"Feature engineering complete. New shape: {df_processed.shape}")
        
        return df_processed
    
    def prepare_features_for_modeling(self, df):
        """
        Prepare features for modeling with proper encoding and scaling.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        logging.info("Preparing features for modeling...")
        
        # Separate target variable
        if 'Price_euros' in df.columns:
            X = df.drop('Price_euros', axis=1)
            y = df['Price_euros'].copy()
        else:
            raise ValueError("Target variable 'Price_euros' not found in dataset")
        
        # Handle categorical variables with label encoding for tree-based models
        # and one-hot encoding for linear models
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        X_processed = X.copy()
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Handle any remaining missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Store feature names
        self.feature_names = list(X_processed.columns)
        
        # Apply Box-Cox transformation to target if it improves normality
        y_transformed = y.copy()
        
        # Check if target needs transformation
        _, p_value = stats.normaltest(y)
        if p_value < 0.05:  # Not normally distributed
            y_positive = y + 1  # Ensure positive values for Box-Cox
            try:
                y_transformed, _ = boxcox(y_positive)
                logging.info("Applied Box-Cox transformation to target variable")
            except:
                logging.info("Box-Cox transformation failed, using original target")
        
        return X_processed, y_transformed
    
    def feature_selection(self, X, y, method='rfe', k=20):
        """
        Perform feature selection to identify most important features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Feature selection method ('rfe', 'kbest', 'both')
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        logging.info(f"Performing feature selection using {method}...")
        
        if method == 'kbest' or method == 'both':
            selector_kbest = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            X_kbest = selector_kbest.fit_transform(X, y)
            selected_features_kbest = X.columns[selector_kbest.get_support()].tolist()
            logging.info(f"K-best selected features: {selected_features_kbest}")
        
        if method == 'rfe' or method == 'both':
            # Use Random Forest for feature selection
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            selector_rfe = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            selector_rfe.fit(X, y)
            selected_features_rfe = X.columns[selector_rfe.support_].tolist()
            logging.info(f"RFE selected features: {selected_features_rfe}")
        
        if method == 'both':
            # Take intersection of both methods
            selected_features = list(set(selected_features_kbest) & set(selected_features_rfe))
            if len(selected_features) < k//2:  # If intersection too small, take union
                selected_features = list(set(selected_features_kbest) | set(selected_features_rfe))
            logging.info(f"Combined feature selection resulted in {len(selected_features)} features")
        elif method == 'kbest':
            selected_features = selected_features_kbest
        else:  # rfe
            selected_features = selected_features_rfe
        
        self.feature_selector = selected_features
        return X[selected_features]
    
    def initialize_models(self):
        """
        Initialize all models with base parameters.
        """
        logging.info("Initializing models...")
        
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'ada_boost': AdaBoostRegressor(random_state=self.random_state),
            'svr': SVR()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
            logging.info("XGBoost model added")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            logging.info("LightGBM model added")
        
        self.models = models
        return models
    
    def evaluate_models(self, X_train, X_test, y_train, y_test, cv_folds=5):
        """
        Evaluate all models using cross-validation and test set performance.
        
        Args:
            X_train, X_test: Training and test feature matrices
            y_train, y_test: Training and test target vectors
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Model evaluation results
        """
        logging.info("Evaluating models with cross-validation...")
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Create pipeline with scaling for models that need it
                if name in ['linear_regression', 'ridge', 'lasso', 'elastic_net', 'svr']:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                else:
                    pipeline = Pipeline([
                        ('model', model)
                    ])
                
                # Cross-validation scores
                cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                          cv=cv_folds, scoring='r2', n_jobs=-1)
                
                # Fit on training data and predict on test set
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_r2': r2,
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'pipeline': pipeline
                }
                
                logging.info(f"{name}: CV R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), "
                           f"Test R² = {r2:.4f}, RMSE = {rmse:.2f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def hyperparameter_optimization(self, X_train, y_train, top_models=3):
        """
        Perform comprehensive hyperparameter optimization for top performing models.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            top_models (int): Number of top models to optimize
            
        Returns:
            dict: Optimized model results
        """
        logging.info("Starting hyperparameter optimization...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['auto', 'sqrt', 0.8]
            },
            'gradient_boosting': {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
                'model__solver': ['auto', 'svd', 'cholesky']
            },
            'lasso': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'model__max_iter': [1000, 2000, 5000]
            }
        }
        
        # Add advanced model parameter grids if available
        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 6, 9],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['lightgbm'] = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [5, 10, 15],
                'model__num_leaves': [31, 50, 100],
                'model__subsample': [0.8, 0.9, 1.0]
            }
        
        optimized_results = {}
        
        # Get initial model evaluation to identify top performers
        initial_results = self.evaluate_models(X_train, X_train, y_train, y_train, cv_folds=3)
        
        # Sort models by CV performance
        sorted_models = sorted(
            [(name, result['cv_mean']) for name, result in initial_results.items() 
             if 'cv_mean' in result],
            key=lambda x: x[1], reverse=True
        )
        
        top_model_names = [name for name, _ in sorted_models[:top_models]]
        logging.info(f"Top {top_models} models for optimization: {top_model_names}")
        
        for model_name in top_model_names:
            if model_name in param_grids:
                logging.info(f"Optimizing hyperparameters for {model_name}...")
                
                # Create pipeline
                if model_name in ['ridge', 'lasso', 'elastic_net', 'svr']:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', self.models[model_name])
                    ])
                else:
                    pipeline = Pipeline([
                        ('model', self.models[model_name])
                    ])
                
                # Use RandomizedSearchCV for efficiency with large parameter spaces
                search = RandomizedSearchCV(
                    pipeline,
                    param_grids[model_name],
                    n_iter=50,  # Number of parameter settings sampled
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                
                try:
                    search.fit(X_train, y_train)
                    
                    optimized_results[model_name] = {
                        'best_score': search.best_score_,
                        'best_params': search.best_params_,
                        'best_estimator': search.best_estimator_
                    }
                    
                    logging.info(f"{model_name} optimized - Best CV Score: {search.best_score_:.4f}")
                    logging.info(f"Best parameters: {search.best_params_}")
                    
                except Exception as e:
                    logging.error(f"Error optimizing {model_name}: {str(e)}")
        
        return optimized_results
    
    def get_feature_importance(self, model, feature_names):
        """
        Extract and return feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        try:
            # Extract model from pipeline if necessary
            if hasattr(model, 'named_steps'):
                actual_model = model.named_steps['model']
            else:
                actual_model = model
            
            # Get feature importance
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
            elif hasattr(actual_model, 'coef_'):
                importances = np.abs(actual_model.coef_)
            else:
                return None
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            logging.error(f"Error extracting feature importance: {str(e)}")
            return None
    
    def plot_results(self, results, feature_importance=None):
        """
        Create visualizations for model results and feature importance.
        
        Args:
            results (dict): Model evaluation results
            feature_importance (pd.DataFrame): Feature importance data
        """
        try:
            # Model performance comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract performance metrics
            model_names = []
            cv_scores = []
            test_scores = []
            rmse_scores = []
            
            for name, result in results.items():
                if 'cv_mean' in result:
                    model_names.append(name)
                    cv_scores.append(result['cv_mean'])
                    test_scores.append(result['test_r2'])
                    rmse_scores.append(result['test_rmse'])
            
            # CV vs Test R² scores
            axes[0, 0].scatter(cv_scores, test_scores)
            for i, name in enumerate(model_names):
                axes[0, 0].annotate(name, (cv_scores[i], test_scores[i]), 
                                   rotation=45, fontsize=8)
            axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[0, 0].set_xlabel('CV R² Score')
            axes[0, 0].set_ylabel('Test R² Score')
            axes[0, 0].set_title('CV vs Test Performance')
            
            # Model comparison bar plot
            axes[0, 1].barh(model_names, test_scores)
            axes[0, 1].set_xlabel('Test R² Score')
            axes[0, 1].set_title('Model Performance Comparison')
            
            # RMSE comparison
            axes[1, 0].barh(model_names, rmse_scores)
            axes[1, 0].set_xlabel('Test RMSE')
            axes[1, 0].set_title('Model RMSE Comparison')
            
            # Feature importance plot
            if feature_importance is not None:
                top_features = feature_importance.head(15)
                axes[1, 1].barh(top_features['feature'], top_features['importance'])
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 15 Feature Importances')
                plt.setp(axes[1, 1].get_yticklabels(), fontsize=8)
            
            plt.tight_layout()
            plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
            logging.info("Results visualization saved as 'model_evaluation_results.png'")
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
    
    def fit(self, filepath='laptop_price.csv', test_size=0.2, optimize_hyperparams=True):
        """
        Complete model training pipeline.
        
        Args:
            filepath (str): Path to dataset
            test_size (float): Proportion of data for testing
            optimize_hyperparams (bool): Whether to perform hyperparameter optimization
            
        Returns:
            dict: Complete training results
        """
        logging.info("Starting complete model training pipeline...")
        
        # 1. Load and prepare data
        df = self.load_and_prepare_data(filepath)
        
        # 2. Feature engineering
        df_processed = self.advanced_feature_engineering(df)
        
        # 3. Prepare features for modeling
        X, y = self.prepare_features_for_modeling(df_processed)
        
        # 4. Feature selection
        X_selected = self.feature_selection(X, y, method='both', k=25)
        
        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=self.random_state
        )
        
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # 6. Initialize and evaluate models
        self.initialize_models()
        results = self.evaluate_models(X_train, X_test, y_train, y_test)
        
        # 7. Hyperparameter optimization
        optimized_results = {}
        if optimize_hyperparams:
            optimized_results = self.hyperparameter_optimization(X_train, y_train, top_models=3)
        
        # 8. Select best model
        best_model_name = max(results.keys(), 
                             key=lambda k: results[k].get('test_r2', -1) 
                             if 'test_r2' in results[k] else -1)
        
        # Use optimized version if available
        if best_model_name in optimized_results:
            self.best_model = optimized_results[best_model_name]['best_estimator']
            best_score = optimized_results[best_model_name]['best_score']
        else:
            self.best_model = results[best_model_name]['pipeline']
            best_score = results[best_model_name]['test_r2']
        
        logging.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
        
        # 9. Feature importance analysis
        feature_importance = self.get_feature_importance(
            self.best_model, X_selected.columns.tolist()
        )
        
        # 10. Create visualizations
        self.plot_results(results, feature_importance)
        
        # 11. Final model evaluation on test set
        final_predictions = self.best_model.predict(X_test)
        final_r2 = r2_score(y_test, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        final_mae = mean_absolute_error(y_test, final_predictions)
        
        logging.info(f"Final model performance - R²: {final_r2:.4f}, "
                    f"RMSE: {final_rmse:.2f}, MAE: {final_mae:.2f}")
        
        # Return comprehensive results
        training_results = {
            'best_model_name': best_model_name,
            'best_model': self.best_model,
            'model_results': results,
            'optimized_results': optimized_results,
            'feature_importance': feature_importance,
            'final_metrics': {
                'r2': final_r2,
                'rmse': final_rmse,
                'mae': final_mae
            },
            'feature_names': X_selected.columns.tolist(),
            'data_info': {
                'original_shape': df.shape,
                'processed_shape': df_processed.shape,
                'final_features': X_selected.shape[1],
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
        }
        
        return training_results
    
    def predict(self, X):
        """
        Make predictions using the best trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            np.array: Predictions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        return self.best_model.predict(X)
    
    def save_model(self, filepath='enhanced_laptop_price_predictor.pkl'):
        """
        Save the trained model and preprocessors.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'best_model': self.best_model,
            'feature_selector': self.feature_selector,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)
        
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_laptop_price_predictor.pkl'):
        """
        Load a trained model and preprocessors.
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as file:
            model_data = pickle.load(file)
        
        self.best_model = model_data['best_model']
        self.feature_selector = model_data['feature_selector']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        logging.info(f"Model loaded from {filepath}")


def main():
    """
    Main function to run the enhanced laptop price prediction model.
    """
    logging.info("Starting Enhanced Laptop Price Prediction Model")
    
    # Initialize the predictor
    predictor = EnhancedLaptopPricePredictor(random_state=42)
    
    # Train the model with comprehensive improvements
    results = predictor.fit(
        filepath='laptop_price.csv',
        test_size=0.2,
        optimize_hyperparams=True
    )
    
    # Print summary results
    print("\n" + "="*60)
    print("ENHANCED LAPTOP PRICE PREDICTION MODEL RESULTS")
    print("="*60)
    
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"Final R² Score: {results['final_metrics']['r2']:.4f}")
    print(f"Final RMSE: {results['final_metrics']['rmse']:.2f} euros")
    print(f"Final MAE: {results['final_metrics']['mae']:.2f} euros")
    
    print(f"\nDataset Information:")
    print(f"- Original features: {results['data_info']['original_shape'][1]}")
    print(f"- Final features after engineering: {results['data_info']['final_features']}")
    print(f"- Training samples: {results['data_info']['training_samples']}")
    print(f"- Test samples: {results['data_info']['test_samples']}")
    
    if results['feature_importance'] is not None:
        print(f"\nTop 10 Most Important Features:")
        for i, row in results['feature_importance'].head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nAll Models Performance Comparison:")
    for name, result in results['model_results'].items():
        if 'test_r2' in result:
            print(f"- {name}: R² = {result['test_r2']:.4f}, "
                  f"RMSE = {result['test_rmse']:.2f}")
    
    # Save the best model
    predictor.save_model('best_laptop_price_model.pkl')
    print(f"\nBest model saved as 'best_laptop_price_model.pkl'")
    print("Model evaluation plots saved as 'model_evaluation_results.png'")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
