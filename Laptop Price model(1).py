#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("LAPTOP PRICE PREDICTION MODEL - EXECUTION STARTED")
logger.info("="*70)


# In[2]:


logger.info("Loading dataset from 'laptop_price.csv'...")
dataset = pd.read_csv("laptop_price.csv",encoding = 'latin-1')
logger.info(f"Dataset loaded successfully. Shape: {dataset.shape} (rows: {dataset.shape[0]}, columns: {dataset.shape[1]})")


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


logger.info("Processing RAM column: removing 'GB' suffix and converting to integer...")
dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
logger.debug(f"RAM column processed. Range: {dataset['Ram'].min()}-{dataset['Ram'].max()} GB")


# In[9]:


dataset.head()


# In[10]:


logger.info("Processing Weight column: removing 'kg' suffix and converting to float...")
dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float64')
logger.debug(f"Weight column processed. Range: {dataset['Weight'].min():.2f}-{dataset['Weight'].max():.2f} kg")


# In[11]:


dataset.head(2)


# In[12]:


logger.info("Identifying non-numeric columns...")
non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
logger.info(f"Non-numeric columns identified: {list(non_numeric_columns)}")


# In[13]:


numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']


# In[14]:


correlation


# In[15]:


dataset['Company'].value_counts()


# In[16]:


logger.info("Consolidating low-frequency companies into 'Other' category...")
OTHER_COMPANIES = {'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'}
def add_company(inpt):
    return 'Other' if inpt in OTHER_COMPANIES else inpt
dataset['Company'] = dataset['Company'].apply(add_company)
logger.info(f"Company consolidation complete. Unique companies: {dataset['Company'].nunique()}")


# In[17]:


dataset['Company'].value_counts()


# In[18]:


len(dataset['Product'].value_counts())


# In[19]:


dataset['TypeName'].value_counts()


# In[20]:


dataset['ScreenResolution'].value_counts()


# In[21]:


logger.info("Extracting touchscreen and IPS features from ScreenResolution...")
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
logger.info(f"Features extracted - Touchscreen: {dataset['Touchscreen'].sum()} laptops, IPS: {dataset['IPS'].sum()} laptops")


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:


logger.info("Extracting CPU name from CPU specification...")
dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
logger.debug(f"CPU name extraction complete. Unique CPUs: {dataset['Cpu_name'].nunique()}")


# In[24]:


dataset['Cpu_name'].value_counts()


# In[25]:


logger.info("Standardizing CPU names (Intel Core i3/i5/i7, AMD, Other)...")
def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)
logger.info(f"CPU standardization complete. Categories: {dataset['Cpu_name'].value_counts().to_dict()}")


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:


logger.info("Extracting GPU manufacturer from GPU specification...")
dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))
logger.debug(f"GPU extraction complete. Unique manufacturers: {dataset['Gpu_name'].nunique()}")


# In[30]:


dataset['Gpu_name'].value_counts()


# In[29]:


logger.info("Filtering out ARM GPU entries...")
rows_before = len(dataset)
dataset = dataset[dataset['Gpu_name'] != 'ARM']
logger.info(f"Removed {rows_before - len(dataset)} ARM GPU entries. Remaining rows: {len(dataset)}")


# In[32]:


dataset.head(2)


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:


logger.info("Standardizing Operating System categories...")
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
logger.info(f"OS standardization complete. Categories: {dataset['OpSys'].value_counts().to_dict()}")


# In[37]:


logger.info("Dropping redundant columns...")
columns_to_drop = ['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu']
logger.debug(f"Columns to drop: {columns_to_drop}")
dataset=dataset.drop(columns=columns_to_drop)
logger.info(f"Columns dropped. Remaining columns: {dataset.shape[1]}")


# In[38]:


dataset.head()


# In[39]:


logger.info("Applying one-hot encoding to categorical variables...")
dataset = pd.get_dummies(dataset)
logger.info(f"One-hot encoding complete. Final shape: {dataset.shape} ({dataset.shape[1]} features)")


# In[40]:


dataset.head()


# In[41]:


logger.info("Splitting dataset into features (X) and target (y)...")
x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']
logger.info(f"Features shape: {x.shape}, Target shape: {y.shape}")


# In[50]:


logger.info("Checking scikit-learn installation...")
try:
    import sklearn  # noqa: F401
    logger.info("scikit-learn is installed and ready")
except ImportError as e:
    logger.error("scikit-learn is not installed!")
    raise ImportError("scikit-learn is required. Please install it via 'pip install scikit-learn'.") from e


# In[51]:


logger.info("Splitting data into train and test sets (75%-25% split)...")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
logger.info(f"Train set: {x_train.shape[0]} samples, Test set: {x_test.shape[0]} samples")


# In[53]:


x_train.shape,x_test.shape


# In[54]:


def model_acc(model):
    logger.info(f"Training model: {model.__class__.__name__}")
    model.fit(x_train,y_train)
    acc = model.score(x_test, y_test)
    logger.info(f"{model.__class__.__name__} --> R² Score: {acc:.4f}")


# In[57]:


logger.info("="*70)
logger.info("BASELINE MODEL EVALUATION")
logger.info("="*70)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
model_acc(rf)


# In[58]:


logger.info("="*70)
logger.info("HYPERPARAMETER TUNING - Random Forest")
logger.info("="*70)

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10,50,100],'criterion':['squared_error','absolute_error','poisson']}
logger.info(f"Grid search parameters: {parameters}")

logger.info("Starting GridSearchCV with 5-fold cross-validation...")
grid_obj = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, n_jobs=-1, scoring='r2')

grid_fit = grid_obj.fit(x_train,y_train)

best_model = grid_fit.best_estimator_
logger.info(f"Best parameters found: {grid_fit.best_params_}")
logger.info(f"Best cross-validation score: {grid_fit.best_score_:.4f}")
best_model


# In[59]:


test_score = best_model.score(x_test,y_test)
logger.info(f"Best model test set R² score: {test_score:.4f}")
test_score


# In[60]:


x_train.columns


# In[68]:


logger.info("Saving trained model to 'predictor.pickle'...")
import pickle
try:
    with open('predictor.pickle','wb') as file:
        pickle.dump(best_model,file)
    logger.info("Model saved successfully")
except Exception as e:
    logger.error(f"Failed to save model: {e}")
    raise


# In[66]:


best_model.predict([[8,1.4,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])


# In[69]:


best_model.predict([[8,0.9,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])


# In[70]:


best_model.predict([[8,1.2,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])


# In[71]:


best_model.predict([[8,0.9,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])


# In[ ]:




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

logger.info("Storage feature engineering complete")
logger.info("Sample storage features:")
logger.info(f"\n{dataset[['Memory', 'Has_SSD', 'Has_HDD', 'Storage_Capacity_GB', 'Storage_Type_Score']].head().to_string()}")


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
logger.info("Creating advanced interaction features...")

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
    x['Weight_Size_Ratio'] = x['Weight'] / x['Inches']

# Total pixels per RAM (graphics capability estimation)
if 'Total_Pixels' in x.columns and 'Ram' in x.columns:
    x['Pixels_Per_RAM'] = x['Total_Pixels'] / (x['Ram'] * 1000000)

# Storage per inch (how much storage per screen size)
if 'Storage_Capacity_GB' in x.columns and 'Inches' in x.columns:
    x['Storage_Per_Inch'] = x['Storage_Capacity_GB'] / x['Inches']

logger.info(f"Advanced interaction features created. Total features: {x.shape[1]}")


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

logger.info(f"Feature scaling complete. Shape: {x_train_scaled_df.shape}")


# In[53]:


x_train.shape,x_test.shape


# In[53a]:


# IMPROVEMENT: Basic outlier detection and reporting
logger.info("="*70)
logger.info("OUTLIER DETECTION")
logger.info("="*70)

from scipy import stats

# Detect outliers in target variable using Z-score
z_scores_target = np.abs(stats.zscore(y_train))
outliers_target = np.where(z_scores_target > 3)[0]

logger.info(f"Target variable (Price) outliers (Z-score > 3): {len(outliers_target)}")
if len(outliers_target) > 0:
    logger.debug(f"Outlier prices: {y_train.iloc[outliers_target].values[:5]} (showing first 5)")

# Detect outliers in key features
key_numeric_features = ['Ram', 'Weight', 'Inches', 'Total_Pixels', 'Storage_Capacity_GB']
outlier_counts = {}

for feature in key_numeric_features:
    if feature in x_train.columns:
        z_scores = np.abs(stats.zscore(x_train[feature]))
        outliers = np.where(z_scores > 3)[0]
        outlier_counts[feature] = len(outliers)
        if len(outliers) > 0:
            logger.info(f"{feature} outliers: {len(outliers)}")

logger.info("Note: Outliers are kept in the dataset as they may represent legitimate premium/budget laptops")
logger.info("Tree-based models handle outliers well without removal")


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
    
    logger.info(f"\n{model_name}:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  MAE: {mae:.2f} euros")
    logger.info(f"  RMSE: {rmse:.2f} euros")
    logger.info(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return r2, mae, rmse, cv_mean


# In[57]:


logger.info("="*70)
logger.info("MODEL COMPARISON - BASELINE MODELS")
logger.info("="*70)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Linear models (use scaled data)
logger.info("--- LINEAR MODELS (with scaling) ---")
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
logger.info("--- TREE-BASED MODELS (no scaling needed) ---")
dt = DecisionTreeRegressor(random_state=42)
model_acc(dt, "Decision Tree")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_acc(rf, "Random Forest (default)")

# Gradient Boosting models
logger.info("--- GRADIENT BOOSTING MODELS ---")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_acc(gb, "Gradient Boosting")

# IMPROVEMENT: Add LightGBM (faster gradient boosting)
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model_acc(lgb_model, "LightGBM")
except ImportError:
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

# Add XGBoost if available
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    model_acc(xgb_model, "XGBoost")
except ImportError:
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


# In[58]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# IMPROVEMENT: Enhanced hyperparameter tuning for Random Forest
logger.info("="*70)
logger.info("HYPERPARAMETER TUNING - Random Forest")
logger.info("="*70)

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

logger.info("Training Random Forest with RandomizedSearchCV (60 iterations)...")
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
logger.info(f"Best parameters: {grid_fit.best_params_}")
logger.info(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:


# IMPROVEMENT: Enhanced Gradient Boosting tuning
logger.info("="*70)
logger.info("HYPERPARAMETER TUNING - Gradient Boosting")
logger.info("="*70)

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

logger.info("Training Gradient Boosting with RandomizedSearchCV (60 iterations)...")
gb_fit = gb_search.fit(x_train, y_train)

best_gb_model = gb_fit.best_estimator_
logger.info(f"Best parameters: {gb_fit.best_params_}")
logger.info(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:


# IMPROVEMENT: Add LightGBM hyperparameter tuning
best_lgb_model = None
try:
    import lightgbm as lgb
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER TUNING - LightGBM")
    logger.info("="*70)
    
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
    
    logger.info("Training LightGBM with RandomizedSearchCV (60 iterations)...")
    lgb_fit = lgb_search.fit(x_train, y_train)
    
    best_lgb_model = lgb_fit.best_estimator_
    logger.info(f"Best parameters: {lgb_fit.best_params_}")
    logger.info(f"Best CV score: {lgb_fit.best_score_:.4f}")
    
except ImportError:
    logger.warning("LightGBM not installed. Skipping LightGBM tuning.")


# In[58c]:


# Compare all best models
logger.info("="*70)
logger.info("FINAL MODEL COMPARISON")
logger.info("="*70)

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

logger.info(f"{'*'*70}")
logger.info(f"WINNER: {best_model_name}")
logger.info(f"R² Score: {models_dict[best_model_name][1]:.4f}")
logger.info(f"{'*'*70}")


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
logger.info("="*70)
logger.info("FEATURE IMPORTANCE ANALYSIS")
logger.info("="*70)

feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': best_overall_model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("Top 15 Most Important Features:")
logger.info(f"\n{feature_importance.head(15).to_string(index=False)}")

# Final model performance
logger.info("="*70)
logger.info("FINAL MODEL PERFORMANCE")
logger.info("="*70)
final_score = best_overall_model.score(x_test, y_test)
logger.info(f"Test R² Score: {final_score:.4f}")


# In[60]:


x_train.columns


# In[68]:


import pickle
import os
import tempfile

# Save the best overall model with comprehensive error handling
logger.info("Saving best overall model with error handling...")
pickle_filename = 'predictor.pickle'
temp_filename = None

try:
    # Create a temporary file first to avoid corrupting existing file
    temp_fd, temp_filename = tempfile.mkstemp(suffix='.pickle', dir='.')
    
    try:
        # Write to temporary file
        with os.fdopen(temp_fd, 'wb') as temp_file:
            pickle.dump(best_overall_model, temp_file)
        
        # If successful, replace the target file
        # Remove existing file if it exists
        if os.path.exists(pickle_filename):
            try:
                os.remove(pickle_filename)
            except PermissionError:
                logger.error(f"Cannot remove existing file '{pickle_filename}'. Permission denied.")
                raise
        
        # Rename temp file to target file
        os.rename(temp_filename, pickle_filename)
        temp_filename = None  # Mark as successfully moved
        
        # Verify the file was written correctly
        file_size = os.path.getsize(pickle_filename)
        logger.info(f"Model successfully saved to '{pickle_filename}' ({file_size:,} bytes)")
        
    except Exception as e:
        # If anything fails, clean up temp file
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.close(temp_fd)
            except:
                pass
            try:
                os.remove(temp_filename)
            except:
                pass
        raise

except PermissionError:
    logger.error(f"Permission denied when trying to write to '{pickle_filename}'.")
    logger.error("Please check that you have write permissions in the current directory.")
    raise
except IOError as e:
    logger.error(f"I/O error while writing model file: {e}")
    logger.error("This could be due to disk space issues or file system problems.")
    raise
except OSError as e:
    logger.error(f"OS error while saving model: {e}")
    if e.errno == 28:  # ENOSPC - No space left on device
        logger.error("Disk is full. Please free up space and try again.")
    raise
except pickle.PicklingError as e:
    logger.error(f"Failed to pickle the model: {e}")
    logger.error("The model may contain unpicklable objects.")
    raise
except Exception as e:
    logger.error(f"Unexpected error while saving model: {e}")
    raise
finally:
    # Clean up temp file if it still exists
    if temp_filename and os.path.exists(temp_filename):
        try:
            os.remove(temp_filename)
            logger.debug(f"Cleaned up temporary file: {temp_filename}")
        except Exception as cleanup_error:
            logger.warning(f"Could not clean up temporary file '{temp_filename}': {cleanup_error}")


# In[66]:


# Note: Predictions need to be updated with the new feature set
# The feature count has changed due to keeping 'Inches' and adding new features
logger.info(f"Number of features: {len(x_train.columns)}")
logger.info("Sample prediction with best model:")
# Use actual test data for demonstration
sample_predictions = best_overall_model.predict(x_test[:5])
logger.info(f"Predictions for first 5 test samples: {sample_predictions}")
logger.info(f"Actual prices: {y_test.iloc[:5].values}")

logger.info("="*70)
logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
logger.info("="*70)


# In[ ]: