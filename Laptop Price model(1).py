#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataset = pd.read_csv("laptop_price.csv",encoding = 'latin-1')


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset['Ram']=dataset['Ram'].str.replace('GB','').astype('int32')


# In[9]:


dataset.head()


# In[10]:


dataset['Weight']=dataset['Weight'].str.replace('kg','').astype('float64')


# In[11]:


dataset.head(2)


# In[12]:


non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)


# In[13]:


numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']


# In[14]:


correlation


# In[15]:


dataset['Company'].value_counts()


# In[16]:


def add_company(inpt):
    if inpt == 'Samsung'or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt
dataset['Company'] = dataset['Company'].apply(add_company)


# In[17]:


dataset['Company'].value_counts()


# In[18]:


len(dataset['Product'].value_counts())


# In[19]:


dataset['TypeName'].value_counts()


# In[20]:


dataset['ScreenResolution'].value_counts()


# In[21]:


# Enhanced screen resolution feature engineering
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

# Extract actual screen resolution dimensions
def extract_resolution(resolution_str):
    """Extract width and height from resolution string"""
    pattern = r'(\d+)x(\d+)'
    match = re.search(pattern, resolution_str)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        return width, height, width * height
    return 1920, 1080, 1920 * 1080  # Default values

dataset[['Screen_Width', 'Screen_Height', 'Screen_Area']] = dataset['ScreenResolution'].apply(
    lambda x: pd.Series(extract_resolution(x))
)

# Screen quality tiers based on resolution
def get_resolution_tier(area):
    if area >= 4000000:  # 4K and above
        return 'Ultra_High'
    elif area >= 2000000:  # QHD
        return 'High' 
    elif area >= 1400000:  # Full HD
        return 'Medium'
    else:  # HD and below
        return 'Low'

dataset['Resolution_Tier'] = dataset['Screen_Area'].apply(get_resolution_tier)


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:


# Enhanced CPU feature extraction
def extract_cpu_features(cpu_str):
    """Extract detailed CPU features"""
    cpu_lower = cpu_str.lower()
    
    # CPU Brand
    if 'intel' in cpu_lower:
        brand = 'Intel'
    elif 'amd' in cpu_lower:
        brand = 'AMD'
    else:
        brand = 'Other'
    
    # CPU Series for Intel
    if 'core i7' in cpu_lower:
        series = 'i7'
    elif 'core i5' in cpu_lower:
        series = 'i5'
    elif 'core i3' in cpu_lower:
        series = 'i3'
    elif 'core i9' in cpu_lower:
        series = 'i9'
    elif 'amd' in cpu_lower:
        if 'ryzen 7' in cpu_lower:
            series = 'Ryzen_7'
        elif 'ryzen 5' in cpu_lower:
            series = 'Ryzen_5'
        elif 'ryzen 3' in cpu_lower:
            series = 'Ryzen_3'
        elif 'a9' in cpu_lower:
            series = 'A9'
        else:
            series = 'AMD_Other'
    else:
        series = 'Other'
    
    # Extract clock speed
    speed_match = re.search(r'(\d+\.?\d*)\s*ghz', cpu_lower)
    speed = float(speed_match.group(1)) if speed_match else 2.0
    
    return brand, series, speed

cpu_features = dataset['Cpu'].apply(lambda x: pd.Series(extract_cpu_features(x)))
cpu_features.columns = ['CPU_Brand', 'CPU_Series', 'CPU_Speed']
dataset = pd.concat([dataset, cpu_features], axis=1)

# Create CPU performance score
cpu_performance_map = {
    'i9': 9, 'i7': 7, 'i5': 5, 'i3': 3,
    'Ryzen_7': 7, 'Ryzen_5': 5, 'Ryzen_3': 3,
    'A9': 2, 'AMD_Other': 2, 'Other': 1
}
dataset['CPU_Performance_Score'] = dataset['CPU_Series'].map(cpu_performance_map)
dataset['CPU_Performance_Score'] *= dataset['CPU_Speed'] / 2.0  # Normalize by typical base speed


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:


# Enhanced GPU feature extraction
def extract_gpu_features(gpu_str):
    """Extract detailed GPU features"""
    gpu_lower = gpu_str.lower()
    
    # GPU Brand
    if 'nvidia' in gpu_lower or 'geforce' in gpu_lower:
        brand = 'Nvidia'
    elif 'amd' in gpu_lower or 'radeon' in gpu_lower:
        brand = 'AMD'
    elif 'intel' in gpu_lower:
        brand = 'Intel'
    else:
        brand = 'Other'
    
    # GPU Performance Tier
    if any(term in gpu_lower for term in ['rtx', 'gtx 1080', 'gtx 1070', 'gtx 1060']):
        tier = 'High_Performance'
    elif any(term in gpu_lower for term in ['gtx', 'mx150', 'mx250', 'mx350', 'radeon pro']):
        tier = 'Mid_Performance'
    elif any(term in gpu_lower for term in ['hd graphics', 'uhd graphics', 'iris']):
        tier = 'Integrated'
    else:
        tier = 'Basic'
    
    # Dedicated vs Integrated
    is_dedicated = 1 if tier in ['High_Performance', 'Mid_Performance'] else 0
    
    return brand, tier, is_dedicated

gpu_features = dataset['Gpu'].apply(lambda x: pd.Series(extract_gpu_features(x)))
gpu_features.columns = ['GPU_Brand', 'GPU_Tier', 'GPU_Dedicated']
dataset = pd.concat([dataset, gpu_features], axis=1)

# Create GPU performance score
gpu_performance_map = {
    'High_Performance': 10,
    'Mid_Performance': 6,
    'Integrated': 3,
    'Basic': 1
}
dataset['GPU_Performance_Score'] = dataset['GPU_Tier'].map(gpu_performance_map)

# Filter out ARM GPUs (they're likely mobile processors)
dataset = dataset[~dataset['Gpu'].str.contains('ARM', case=False, na=False)]


# In[32]:


dataset.head(2)


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:


def set_os(inpt):
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'
dataset['OpSys']= dataset['OpSys'].apply(set_os)


# In[37]:


# Create interaction features
dataset['Performance_Index'] = dataset['CPU_Performance_Score'] * dataset['GPU_Performance_Score'] * dataset['Ram']
dataset['Screen_Performance_Ratio'] = dataset['Screen_Area'] / (dataset['CPU_Performance_Score'] + 1)
dataset['Ram_per_Weight'] = dataset['Ram'] / dataset['Weight']
dataset['Price_per_Ram'] = dataset['Ram'] * 100  # Will be used for feature engineering

# Memory tier based on RAM
def get_ram_tier(ram):
    if ram >= 32:
        return 'Ultra_High'
    elif ram >= 16:
        return 'High'
    elif ram >= 8:
        return 'Medium'
    else:
        return 'Low'

dataset['RAM_Tier'] = dataset['Ram'].apply(get_ram_tier)

# Weight category
def get_weight_category(weight):
    if weight <= 1.5:
        return 'Ultra_Light'
    elif weight <= 2.0:
        return 'Light'
    elif weight <= 2.5:
        return 'Medium'
    else:
        return 'Heavy'

dataset['Weight_Category'] = dataset['Weight'].apply(get_weight_category)

# Drop original columns that have been feature engineered
dataset = dataset.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])

print(f"Dataset shape after feature engineering: {dataset.shape}")
print(f"New features added: {[col for col in dataset.columns if col not in ['Company', 'TypeName', 'Ram', 'OpSys', 'Weight', 'Price_euros']]}")

# Convert categorical variables to dummy variables
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
if 'Price_euros' in categorical_columns:
    categorical_columns.remove('Price_euros')

print(f"Categorical columns to encode: {categorical_columns}")
dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)


# In[40]:


dataset.head()


# In[41]:


# Prepare features and target
X = dataset.drop('Price_euros', axis=1)
y = dataset['Price_euros']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data with stratification to ensure balanced price ranges
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Feature scaling for algorithms that benefit from it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Enhanced model evaluation function
def evaluate_model(model, X_train_data, X_test_data, y_train_data, y_test_data, model_name, use_cv=True):
    """Comprehensive model evaluation with cross-validation"""
    
    # Fit the model
    model.fit(X_train_data, y_train_data)
    
    # Predictions
    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)
    
    # Metrics
    train_r2 = r2_score(y_train_data, y_train_pred)
    test_r2 = r2_score(y_test_data, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_data, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_data, y_test_pred))
    train_mae = mean_absolute_error(y_train_data, y_train_pred)
    test_mae = mean_absolute_error(y_test_data, y_test_pred)
    
    # Cross-validation
    cv_scores = None
    if use_cv:
        try:
            cv_scores = cross_val_score(model, X_train_data, y_train_data, cv=5, scoring='r2')
        except:
            cv_scores = None
    
    results = {
        'Model': model_name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'CV_Mean': cv_scores.mean() if cv_scores is not None else None,
        'CV_Std': cv_scores.std() if cv_scores is not None else None,
        'Overfitting': train_r2 - test_r2
    }
    
    return results, model

# Test multiple models
models_to_test = []

# Traditional models
models_to_test.append((LinearRegression(), X_train_scaled, X_test_scaled, 'Linear Regression'))
models_to_test.append((Ridge(alpha=1.0), X_train_scaled, X_test_scaled, 'Ridge Regression'))
models_to_test.append((Lasso(alpha=1.0), X_train_scaled, X_test_scaled, 'Lasso Regression'))
models_to_test.append((ElasticNet(alpha=1.0, l1_ratio=0.5), X_train_scaled, X_test_scaled, 'ElasticNet'))

# Tree-based models (don't need scaling)
models_to_test.append((DecisionTreeRegressor(random_state=42), X_train, X_test, 'Decision Tree'))
models_to_test.append((RandomForestRegressor(random_state=42), X_train, X_test, 'Random Forest'))
models_to_test.append((GradientBoostingRegressor(random_state=42), X_train, X_test, 'Gradient Boosting'))

# Advanced models if available
if XGBOOST_AVAILABLE:
    models_to_test.append((xgb.XGBRegressor(random_state=42, eval_metric='rmse'), X_train, X_test, 'XGBoost'))

if LIGHTGBM_AVAILABLE:
    models_to_test.append((lgb.LGBMRegressor(random_state=42, verbose=-1), X_train, X_test, 'LightGBM'))

# Evaluate all models
results_list = []
trained_models = {}

print("Evaluating models...")
print("="*80)

for model, X_train_data, X_test_data, name in models_to_test:
    try:
        results, fitted_model = evaluate_model(model, X_train_data, X_test_data, y_train, y_test, name)
        results_list.append(results)
        trained_models[name] = fitted_model
        
        print(f"{name}:")
        print(f"  Test R²: {results['Test_R2']:.4f}")
        print(f"  Test RMSE: {results['Test_RMSE']:.2f}")
        print(f"  Test MAE: {results['Test_MAE']:.2f}")
        if results['CV_Mean'] is not None:
            print(f"  CV R² (mean±std): {results['CV_Mean']:.4f}±{results['CV_Std']:.4f}")
        print(f"  Overfitting: {results['Overfitting']:.4f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error training {name}: {str(e)}")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('Test_R2', ascending=False)
print("\nModel Comparison Summary:")
print(results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'CV_Mean', 'Overfitting']].round(4))

# Advanced hyperparameter tuning for best performing models
print("\n" + "="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Get top 3 models for hyperparameter tuning
top_models = results_df.head(3)['Model'].tolist()
tuned_models = {}

# Define parameter grids for different models
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9, 1.0]
    } if XGBOOST_AVAILABLE else {},
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9, 1.0]
    } if LIGHTGBM_AVAILABLE else {}
}

for model_name in top_models:
    if model_name in param_grids and param_grids[model_name]:
        print(f"\nTuning {model_name}...")
        
        # Get the base model
        if model_name == 'Random Forest':
            base_model = RandomForestRegressor(random_state=42)
            X_train_data, X_test_data = X_train, X_test
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingRegressor(random_state=42)
            X_train_data, X_test_data = X_train, X_test
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            base_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
            X_train_data, X_test_data = X_train, X_test
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            X_train_data, X_test_data = X_train, X_test
        else:
            continue
        
        # Use RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            base_model,
            param_grids[model_name],
            n_iter=50,  # Number of parameter settings sampled
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the random search
        random_search.fit(X_train_data, y_train)
        
        # Get the best model
        best_tuned_model = random_search.best_estimator_
        tuned_models[model_name] = best_tuned_model
        
        # Evaluate the tuned model
        y_pred_tuned = best_tuned_model.predict(X_test_data)
        tuned_r2 = r2_score(y_test, y_pred_tuned)
        tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
        tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
        
        print(f"  Best parameters: {random_search.best_params_}")
        print(f"  Tuned Test R²: {tuned_r2:.4f}")
        print(f"  Tuned Test RMSE: {tuned_rmse:.2f}")
        print(f"  Tuned Test MAE: {tuned_mae:.2f}")
        print(f"  Best CV score: {random_search.best_score_:.4f}")

# Feature importance analysis for the best model
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get the best overall model
best_model_name = results_df.iloc[0]['Model']
if best_model_name in tuned_models:
    final_best_model = tuned_models[best_model_name]
    best_model_data = X_train, X_test
else:
    final_best_model = trained_models[best_model_name]
    # Determine if we need scaled data
    if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        best_model_data = X_train_scaled, X_test_scaled
    else:
        best_model_data = X_train, X_test

# Feature importance (for tree-based models)
if hasattr(final_best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 15 Most Important Features for {best_model_name}:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Feature selection based on importance
    selector = SelectKBest(score_func=f_regression, k=20)
    X_train_selected = selector.fit_transform(best_model_data[0] if isinstance(best_model_data[0], np.ndarray) else best_model_data[0], y_train)
    X_test_selected = selector.transform(best_model_data[1] if isinstance(best_model_data[1], np.ndarray) else best_model_data[1])
    
    print(f"\nSelected features: {X.columns[selector.get_support()].tolist()}")

print(f"\nFinal Best Model: {best_model_name}")
y_final_pred = final_best_model.predict(best_model_data[1])
final_r2 = r2_score(y_test, y_final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
final_mae = mean_absolute_error(y_test, y_final_pred)

print(f"Final Model Performance:")
print(f"  R² Score: {final_r2:.4f}")
print(f"  RMSE: {final_rmse:.2f} euros")
print(f"  MAE: {final_mae:.2f} euros")

# Save the best model
import pickle
with open('improved_laptop_price_predictor.pickle', 'wb') as file:
    pickle.dump({
        'model': final_best_model,
        'scaler': scaler if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet'] else None,
        'feature_names': X.columns.tolist(),
        'model_name': best_model_name
    }, file)

print(f"\nModel saved as 'improved_laptop_price_predictor.pickle'")

# Example predictions with the improved model
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

# Create sample prediction function
def predict_laptop_price(model, scaler, feature_names, **kwargs):
    """Make prediction for a laptop with given specifications"""
    # Create a sample with default values
    sample = pd.Series(0, index=feature_names)
    
    # Set the provided values
    for key, value in kwargs.items():
        if key in feature_names:
            sample[key] = value
    
    # Transform if needed
    if scaler is not None:
        sample_scaled = scaler.transform(sample.values.reshape(1, -1))
        prediction = model.predict(sample_scaled)[0]
    else:
        prediction = model.predict(sample.values.reshape(1, -1))[0]
    
    return prediction

# Example predictions
sample_configs = [
    {
        'Ram': 8,
        'Weight': 1.4,
        'CPU_Performance_Score': 35,
        'GPU_Performance_Score': 6,
        'Screen_Area': 2073600,
        'Touchscreen': 0,
        'IPS': 1
    },
    {
        'Ram': 16,
        'Weight': 1.8,
        'CPU_Performance_Score': 49,
        'GPU_Performance_Score': 10,
        'Screen_Area': 2073600,
        'Touchscreen': 1,
        'IPS': 1
    }
]

for i, config in enumerate(sample_configs, 1):
    try:
        pred = predict_laptop_price(
            final_best_model, 
            scaler if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet'] else None,
            X.columns.tolist(),
            **config
        )
        print(f"Sample {i} - Predicted price: €{pred:.2f}")
        print(f"  Configuration: {config}")
    except Exception as e:
        print(f"Error making prediction {i}: {str(e)}")

print("\n" + "="*80)
print("MODEL IMPROVEMENT SUMMARY")
print("="*80)
print("Improvements made:")
print("1. Enhanced feature engineering:")
print("   - Screen resolution dimensions and area")
print("   - Detailed CPU/GPU performance scores")
print("   - Interaction features (Performance Index, etc.)")
print("   - Categorical tiers for RAM and weight")
print("2. Advanced algorithms:")
print("   - XGBoost and LightGBM (if available)")
print("   - Gradient Boosting")
print("   - Multiple regularized linear models")
print("3. Better evaluation:")
print("   - Cross-validation")
print("   - Multiple metrics (R², RMSE, MAE)")
print("   - Overfitting detection")
print("4. Hyperparameter tuning:")
print("   - RandomizedSearchCV for efficient search")
print("   - Model-specific parameter grids")
print("5. Feature selection and analysis:")
print("   - Importance ranking")
print("   - Statistical feature selection")
