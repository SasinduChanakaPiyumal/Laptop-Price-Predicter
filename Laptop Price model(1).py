#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


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


# Enhanced feature engineering from ScreenResolution
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

# Extract actual screen resolution (width x height)
def extract_resolution(res_string):
    import re
    # Find pattern like "1920x1080" or "3840x2160"
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height, width * height  # width, height, total pixels
    return 1366, 768, 1366*768  # default resolution if not found

dataset['Screen_Width'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[0])
dataset['Screen_Height'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[1])
dataset['Total_Pixels'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[2])

# Calculate PPI (Pixels Per Inch) - important quality metric
# Fix: Prevent division by zero or invalid values
dataset['PPI'] = np.where(dataset['Inches'] > 0, 
                          np.sqrt(dataset['Total_Pixels']) / dataset['Inches'],
                          0)  # Default to 0 for invalid screen sizes


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:


dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[24]:


dataset['Cpu_name'].value_counts()


# In[25]:


def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:


dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))


# In[30]:


dataset['Gpu_name'].value_counts()


# In[29]:


dataset = dataset[dataset['Gpu_name'] != 'ARM']


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


# In[34a]:


# IMPROVEMENT: Enhanced Memory/Storage Feature Engineering
# Extract storage type and capacity from Memory column
print("Memory/Storage Feature Engineering...")

def extract_storage_features(memory_string):
    """
    Extract storage type and total capacity from memory string.
    Examples: "256GB SSD", "1TB HDD", "128GB SSD +  1TB HDD", "256GB Flash Storage"
    """
    memory_string = str(memory_string)
    
    # Initialize features
    has_ssd = 0
    has_hdd = 0
    has_flash = 0
    has_hybrid = 0
    total_capacity_gb = 0
    
    # Check for storage types
    if 'SSD' in memory_string:
        has_ssd = 1
    if 'HDD' in memory_string:
        has_hdd = 1
    if 'Flash' in memory_string:
        has_flash = 1
    if 'Hybrid' in memory_string:
        has_hybrid = 1
    
    # Extract capacities
    import re
    
    # Find all capacity values with TB or GB
    tb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_string)
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_string)
    
    # Convert to GB and sum
    for tb in tb_matches:
        total_capacity_gb += float(tb) * 1024
    for gb in gb_matches:
        total_capacity_gb += float(gb)
    
    return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb

# Apply storage feature extraction
storage_features = dataset['Memory'].apply(extract_storage_features)
dataset['Has_SSD'] = storage_features.apply(lambda x: x[0])
dataset['Has_HDD'] = storage_features.apply(lambda x: x[1])
dataset['Has_Flash'] = storage_features.apply(lambda x: x[2])
dataset['Has_Hybrid'] = storage_features.apply(lambda x: x[3])
dataset['Storage_Capacity_GB'] = storage_features.apply(lambda x: x[4])

# Create derived storage features
dataset['Storage_Type_Score'] = (
    dataset['Has_SSD'] * 3 +      # SSD is premium
    dataset['Has_Flash'] * 2.5 +  # Flash is also premium
    dataset['Has_Hybrid'] * 2 +   # Hybrid is mid-range
    dataset['Has_HDD'] * 1        # HDD is budget
)

print(f"Storage feature engineering complete.")
print(f"Sample storage features:")
print(dataset[['Memory', 'Has_SSD', 'Has_HDD', 'Storage_Capacity_GB', 'Storage_Type_Score']].head())


# In[37]:


# Keep screen size and drop only redundant columns (now also drop Memory after feature extraction)
dataset=dataset.drop(columns=['laptop_ID','Product','ScreenResolution','Cpu','Gpu','Memory'])


# In[38]:


dataset.head()


# In[39]:


dataset = pd.get_dummies(dataset)


# In[40]:


dataset.head()


# In[41]:


x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']


# In[42]:


# Create interaction features for better predictions
# RAM and CPU quality interaction (high RAM with good CPU = premium laptop)
# These will be added after dummy encoding, so we need to calculate them from the encoded features
# Store numeric columns before creating interactions
numeric_cols = ['Ram', 'Weight', 'Inches', 'Touchscreen', 'IPS', 'Screen_Width', 'Screen_Height', 'Total_Pixels', 'PPI']

# Create polynomial and interaction features for key numeric features
from sklearn.preprocessing import PolynomialFeatures
key_features = ['Ram', 'Weight', 'Total_Pixels', 'PPI']
poly_feature_names = [col for col in x.columns if any(feat in str(col) for feat in key_features)]

# Add RAM squared (premium laptops may have non-linear pricing with RAM)
if 'Ram' in x.columns:
    x['Ram_squared'] = x['Ram'] ** 2
    
# Add interaction between screen quality and size
if 'Total_Pixels' in x.columns and 'Inches' in x.columns:
    x['Screen_Quality'] = x['Total_Pixels'] / 1000000 * x['Inches']  # Normalized quality metric

# IMPROVEMENT: Additional advanced interaction features
print("\nCreating advanced interaction features...")

# Storage capacity * SSD indicator (SSD with high capacity is premium)
if 'Storage_Capacity_GB' in x.columns and 'Has_SSD' in x.columns:
    x['Premium_Storage'] = x['Storage_Capacity_GB'] * (x['Has_SSD'] + 1) / 1000  # Normalized

# RAM * Storage Type Score (high RAM + fast storage = workstation/gaming)
if 'Ram' in x.columns and 'Storage_Type_Score' in x.columns:
    x['RAM_Storage_Quality'] = x['Ram'] * x['Storage_Type_Score']

# Screen quality * Storage quality (premium display + premium storage)
if 'PPI' in x.columns and 'Storage_Type_Score' in x.columns:
    x['Display_Storage_Premium'] = x['PPI'] * x['Storage_Type_Score']

# Weight to size ratio (portability factor)
if 'Weight' in x.columns and 'Inches' in x.columns:
    # Fix: Prevent division by zero
    x['Weight_Size_Ratio'] = np.where(x['Inches'] > 0, 
                                       x['Weight'] / x['Inches'],
                                       0)

# Total pixels per RAM (graphics capability estimation)
if 'Total_Pixels' in x.columns and 'Ram' in x.columns:
    # Fix: Prevent division by zero
    x['Pixels_Per_RAM'] = np.where(x['Ram'] > 0,
                                    x['Total_Pixels'] / (x['Ram'] * 1000000),
                                    0)

# Storage per inch (how much storage per screen size)
if 'Storage_Capacity_GB' in x.columns and 'Inches' in x.columns:
    # Fix: Prevent division by zero
    x['Storage_Per_Inch'] = np.where(x['Inches'] > 0,
                                      x['Storage_Capacity_GB'] / x['Inches'],
                                      0)

print(f"Advanced interaction features created. Total features: {x.shape[1]}")


# In[50]:


pip install scikit-learn


# In[51]:


from sklearn.model_selection import train_test_split
# Add random_state for reproducibility
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[52]:


# IMPROVEMENT: Add feature scaling for linear models
from sklearn.preprocessing import StandardScaler

# Create scaled versions for linear models
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert back to DataFrame for consistency
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

print(f"\nFeature scaling complete. Shape: {x_train_scaled_df.shape}")


# In[53]:


x_train.shape,x_test.shape


# In[53a]:


# IMPROVEMENT: Basic outlier detection and reporting
print("\n" + "="*60)
print("OUTLIER DETECTION")
print("="*60)

from scipy import stats

# Detect outliers in target variable using Z-score
z_scores_target = np.abs(stats.zscore(y_train))
outliers_target = np.where(z_scores_target > 3)[0]

print(f"\nTarget variable (Price) outliers (Z-score > 3): {len(outliers_target)}")
if len(outliers_target) > 0:
    print(f"Outlier prices: {y_train.iloc[outliers_target].values[:5]} (showing first 5)")

# Detect outliers in key features
key_numeric_features = ['Ram', 'Weight', 'Inches', 'Total_Pixels', 'Storage_Capacity_GB']
outlier_counts = {}

for feature in key_numeric_features:
    if feature in x_train.columns:
        z_scores = np.abs(stats.zscore(x_train[feature]))
        outliers = np.where(z_scores > 3)[0]
        outlier_counts[feature] = len(outliers)
        if len(outliers) > 0:
            print(f"{feature} outliers: {len(outliers)}")

print(f"\nNote: Outliers are kept in the dataset as they may represent legitimate premium/budget laptops.")
print("Tree-based models handle outliers well without removal.")


# In[54]:


# Improved evaluation function with multiple metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def model_acc(model, model_name="Model", use_scaled=False):
    """
    Improved model evaluation with multiple metrics.
    
    Args:
        model: sklearn model to evaluate
        model_name: Name for display
        use_scaled: If True, use scaled data (for linear models)
    """
    # Choose data based on scaling requirement
    if use_scaled:
        X_train = x_train_scaled_df
        X_test = x_test_scaled_df
    else:
        X_train = x_train
        X_test = x_test
    
    model.fit(X_train, y_train)
    # Test set predictions
    y_pred = model.predict(X_test)
    
    # Calculate multiple metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} euros")
    print(f"  RMSE: {rmse:.2f} euros")
    print(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return r2, mae, rmse, cv_mean


# In[57]:


print("\n" + "="*60)
print("MODEL COMPARISON - BASELINE MODELS")
print("="*60)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Linear models (use scaled data)
print("\n--- LINEAR MODELS (with scaling) ---")
lr = LinearRegression()
model_acc(lr, "Linear Regression", use_scaled=True)

# IMPROVEMENT: Add Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0, random_state=42)
model_acc(ridge, "Ridge Regression", use_scaled=True)

lasso = Lasso(alpha=1.0, random_state=42)
model_acc(lasso, "Lasso Regression (L1)", use_scaled=True)

# IMPROVEMENT: Add ElasticNet (L1 + L2 regularization)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
model_acc(elastic, "ElasticNet (L1+L2)", use_scaled=True)

# Tree-based models (don't need scaling)
print("\n--- TREE-BASED MODELS (no scaling needed) ---")
dt = DecisionTreeRegressor(random_state=42)
model_acc(dt, "Decision Tree")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_acc(rf, "Random Forest (default)")

# Gradient Boosting models
print("\n--- GRADIENT BOOSTING MODELS ---")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_acc(gb, "Gradient Boosting")

# IMPROVEMENT: Add LightGBM (faster gradient boosting)
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_acc(lgb_model, "LightGBM")
except ImportError:
    print("\nLightGBM not installed. Install with: pip install lightgbm")

# Add XGBoost if available
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    model_acc(xgb_model, "XGBoost")
except ImportError:
    print("\nXGBoost not installed. Install with: pip install xgboost")


# In[58]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# IMPROVEMENT: Enhanced hyperparameter tuning for Random Forest
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - Random Forest")
print("="*60)

# Expanded parameter space with continuous distributions
rf_parameters = {
    'n_estimators': [150, 200, 300, 400],  # More estimators for better performance
    'max_depth': [15, 20, 25, 30, None],   # Better depth range
    'min_samples_split': [2, 4, 6, 8],     # Finer granularity
    'min_samples_leaf': [1, 2, 3, 4],      # More options
    'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Include float values
    'bootstrap': [True, False],
    'min_impurity_decrease': [0.0, 0.001, 0.01]  # NEW: Regularization parameter
}

grid_obj = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_parameters,
    n_iter=60,  # Increased from 50 to 60 iterations
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

print("\nTraining Random Forest with RandomizedSearchCV (60 iterations)...")
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
print(f"\nBest parameters: {grid_fit.best_params_}")
print(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:


# IMPROVEMENT: Enhanced Gradient Boosting tuning
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - Gradient Boosting")
print("="*60)

# Improved parameter space based on best practices
gb_parameters = {
    'n_estimators': [150, 200, 300, 400],     # More estimators
    'learning_rate': [0.01, 0.03, 0.05, 0.075, 0.1, 0.15],  # Finer learning rate grid
    'max_depth': [3, 4, 5, 6, 7],             # Optimal range for GB
    'min_samples_split': [2, 4, 6, 8, 10],    # More options
    'min_samples_leaf': [1, 2, 3, 4],         # Finer granularity
    'subsample': [0.7, 0.8, 0.85, 0.9, 1.0],  # More subsample options
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],  # Include float values
    'min_impurity_decrease': [0.0, 0.001, 0.01]  # NEW: Regularization
}

gb_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=gb_parameters,
    n_iter=60,  # Increased from 50 to 60
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Gradient Boosting with RandomizedSearchCV (60 iterations)...")
gb_fit = gb_search.fit(x_train, y_train)

best_gb_model = gb_fit.best_estimator_
print(f"\nBest parameters: {gb_fit.best_params_}")
print(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:


# IMPROVEMENT: Add LightGBM hyperparameter tuning
best_lgb_model = None
try:
    import lightgbm as lgb
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - LightGBM")
    print("="*60)
    
    lgb_parameters = {
        'n_estimators': [150, 200, 300, 400],
        'learning_rate': [0.01, 0.03, 0.05, 0.075, 0.1],
        'max_depth': [3, 5, 7, 10, -1],  # -1 means no limit
        'num_leaves': [15, 31, 50, 70, 100],  # Important for LightGBM
        'min_child_samples': [5, 10, 20, 30],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1.0],  # L1 regularization
        'reg_lambda': [0, 0.01, 0.1, 1.0]  # L2 regularization
    }
    
    lgb_search = RandomizedSearchCV(
        estimator=lgb.LGBMRegressor(random_state=42, verbose=-1),
        param_distributions=lgb_parameters,
        n_iter=60,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\nTraining LightGBM with RandomizedSearchCV (60 iterations)...")
    lgb_fit = lgb_search.fit(x_train, y_train)
    
    best_lgb_model = lgb_fit.best_estimator_
    print(f"\nBest parameters: {lgb_fit.best_params_}")
    print(f"Best CV score: {lgb_fit.best_score_:.4f}")
    
except ImportError:
    print("\nLightGBM not installed. Skipping LightGBM tuning.")


# In[58c]:


# Compare all best models
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

rf_r2, rf_mae, rf_rmse, rf_cv = model_acc(best_model, "Best Random Forest")
gb_r2, gb_mae, gb_rmse, gb_cv = model_acc(best_gb_model, "Best Gradient Boosting")

# Compare with LightGBM if available
models_dict = {
    'Random Forest': (best_model, rf_r2),
    'Gradient Boosting': (best_gb_model, gb_r2)
}

if best_lgb_model is not None:
    lgb_r2, lgb_mae, lgb_rmse, lgb_cv = model_acc(best_lgb_model, "Best LightGBM")
    models_dict['LightGBM'] = (best_lgb_model, lgb_r2)

# Select the best overall model
best_model_name = max(models_dict, key=lambda k: models_dict[k][1])
best_overall_model = models_dict[best_model_name][0]

print(f"\n{'*'*60}")
print(f"WINNER: {best_model_name}")
print(f"R² Score: {models_dict[best_model_name][1]:.4f}")
print(f"{'*'*60}")


# In[59]:


# IMPROVEMENT SUMMARY:
# This enhanced version includes the following improvements:
# 1. Storage/Memory feature engineering (SSD/HDD detection, capacity extraction)
# 2. Advanced interaction features (RAM-Storage, Display-Storage, Weight-Size ratio, etc.)
# 3. Feature scaling for linear models
# 4. Additional models: Ridge, ElasticNet, LightGBM
# 5. Enhanced hyperparameter tuning with broader parameter ranges (60 iterations)
# 6. Outlier detection and reporting
# 7. Comprehensive model comparison across all tuned models

# Feature importance analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': best_overall_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Final model performance
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE")
print("="*60)
final_score = best_overall_model.score(x_test, y_test)
print(f"Test R² Score: {final_score:.4f}")


# In[60]:


x_train.columns


# In[68]:


import pickle
# Save the best overall model (could be RF or GB)
with open('predictor.pickle','wb') as file:
    pickle.dump(best_overall_model,file)
    
print("\nModel saved to predictor.pickle")


# In[66]:


# Note: Predictions need to be updated with the new feature set
# The feature count has changed due to keeping 'Inches' and adding new features
print(f"\nNumber of features: {len(x_train.columns)}")
print("Sample prediction with best model:")
# Use actual test data for demonstration
sample_predictions = best_overall_model.predict(x_test[:5])
print(f"Predictions for first 5 test samples: {sample_predictions}")
print(f"Actual prices: {y_test.iloc[:5].values}")


# In[ ]:




