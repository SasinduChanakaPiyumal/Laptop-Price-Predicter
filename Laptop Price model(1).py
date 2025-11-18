#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'laptop_price_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("LAPTOP PRICE PREDICTION MODEL - Starting Execution")
logger.info("="*60)


# In[2]:

logger.info("Loading dataset from 'laptop_price.csv'...")
dataset = pd.read_csv("laptop_price.csv",encoding = 'latin-1')
logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# In[7]:

logger.info("Checking for null values...")
null_counts = dataset.isnull().sum()
logger.debug(f"Null value counts:\n{null_counts}")


# In[8]:

logger.info("Preprocessing: Converting RAM column to numeric (removing 'GB' suffix)...")
dataset['Ram'] = dataset['Ram'].str.replace('GB', '', regex=False).astype('int32')
logger.info("RAM column converted successfully")


# In[9]:


dataset.head()


# In[10]:

logger.info("Preprocessing: Converting Weight column to numeric (removing 'kg' suffix)...")
dataset['Weight'] = dataset['Weight'].str.replace('kg', '', regex=False).astype('float64')
logger.info("Weight column converted successfully")


# In[11]:


dataset.head(2)


# In[12]:

logger.info("Identifying non-numeric columns for feature engineering...")
non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
logger.info(f"Non-numeric columns found: {list(non_numeric_columns)}")


# In[13]:

logger.info("Computing correlation matrix for numeric features...")
numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']
logger.debug("Correlation analysis completed")


# In[14]:


correlation


# In[15]:


dataset['Company'].value_counts()


# In[16]:

logger.info("Feature engineering: Grouping rare companies into 'Other' category...")
OTHER_COMPANIES = {'Samsung', 'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'}
def add_company(inpt):
    return 'Other' if inpt in OTHER_COMPANIES else inpt
dataset['Company'] = dataset['Company'].apply(add_company)
logger.info(f"Company grouping completed. Unique companies: {dataset['Company'].nunique()}")


# In[17]:


dataset['Company'].value_counts()


# In[18]:


len(dataset['Product'].value_counts())


# In[19]:


dataset['TypeName'].value_counts()


# In[20]:


dataset['ScreenResolution'].value_counts()


# In[21]:

logger.info("Feature engineering: Extracting Touchscreen and IPS indicators from screen resolution...")
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
logger.info(f"Touchscreen devices: {dataset['Touchscreen'].sum()}, IPS displays: {dataset['IPS'].sum()}")


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:

logger.info("Feature engineering: Extracting CPU name from full CPU specification...")
dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
logger.debug(f"CPU names extracted. Unique CPUs: {dataset['Cpu_name'].nunique()}")


# In[24]:


dataset['Cpu_name'].value_counts()


# In[25]:

logger.info("Feature engineering: Categorizing CPU names into major groups...")
def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)
logger.info(f"CPU categorization completed. Categories: {dataset['Cpu_name'].value_counts().to_dict()}")


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:

logger.info("Feature engineering: Extracting GPU brand from GPU specification...")
dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))
logger.debug(f"GPU brands extracted. Unique GPUs: {dataset['Gpu_name'].nunique()}")


# In[30]:


dataset['Gpu_name'].value_counts()


# In[29]:

logger.info("Data cleaning: Removing ARM GPU entries (edge cases)...")
original_count = len(dataset)
dataset = dataset[dataset['Gpu_name'] != 'ARM']
logger.info(f"Removed {original_count - len(dataset)} ARM GPU entries. Remaining: {len(dataset)}")


# In[32]:


dataset.head(2)


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:

logger.info("Feature engineering: Categorizing operating systems...")
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
logger.info(f"OS categorization completed. Distribution: {dataset['OpSys'].value_counts().to_dict()}")


# In[37]:

logger.info("Dropping redundant columns after feature extraction...")
columns_to_drop = ['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu']
logger.debug(f"Dropping columns: {columns_to_drop}")
dataset=dataset.drop(columns=columns_to_drop)
logger.info(f"Columns dropped. Remaining shape: {dataset.shape}")


# In[38]:


dataset.head()


# In[39]:

logger.info("Applying one-hot encoding to categorical variables...")
original_columns = dataset.shape[1]
dataset = pd.get_dummies(dataset)
logger.info(f"One-hot encoding completed. Columns before: {original_columns}, after: {dataset.shape[1]}")


# In[40]:


dataset.head()


# In[41]:

logger.info("Splitting dataset into features (X) and target (y)...")
x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']
logger.info(f"Feature matrix shape: {x.shape}, Target shape: {y.shape}")


# In[50]:


try:
    import sklearn  # noqa: F401
except ImportError as e:
    raise ImportError("scikit-learn is required. Please install it via 'pip install scikit-learn'.") from e


# In[51]:

logger.info("Splitting data into train and test sets (75-25 split)...")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
logger.info(f"Train set: {x_train.shape[0]} samples, Test set: {x_test.shape[0]} samples")


# In[53]:


x_train.shape,x_test.shape


# In[54]:

logger.info("Defining model evaluation function...")
def model_acc(model):
    logger.info(f"Training model: {type(model).__name__}...")
    model.fit(x_train,y_train)
    acc = model.score(x_test, y_test)
    logger.info(f"{type(model).__name__} --> R² Score: {acc:.4f}")


# In[57]:

logger.info("="*60)
logger.info("MODEL TRAINING - Baseline Models")
logger.info("="*60)

logger.info("Training Linear Regression model...")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

logger.info("Training Lasso Regression model...")
from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

logger.info("Training Decision Tree Regressor...")
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

logger.info("Training Random Forest Regressor...")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
model_acc(rf)


# In[58]:

logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - Random Forest with GridSearchCV")
logger.info("="*60)

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10,50,100],'criterion':['squared_error','absolute_error','poisson']}
logger.info(f"Parameter grid: {parameters}")

logger.info("Starting GridSearchCV (5-fold cross-validation)...")
grid_obj = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, n_jobs=-1, scoring='r2')

grid_fit = grid_obj.fit(x_train,y_train)
logger.info("GridSearchCV completed")

best_model = grid_fit.best_estimator_
logger.info(f"Best parameters: {grid_fit.best_params_}")
logger.info(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[59]:

logger.info("Evaluating best model on test set...")
test_score = best_model.score(x_test,y_test)
logger.info(f"Best model test R² score: {test_score:.4f}")
test_score


# In[60]:


x_train.columns


# In[68]:


import pickle
with open('predictor.pickle','wb') as file:
    pickle.dump(best_model,file)


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
logger.info("Advanced Feature Engineering: Extracting storage features from Memory column...")
storage_features = dataset['Memory'].apply(extract_storage_features)
dataset['Has_SSD'] = storage_features.apply(lambda x: x[0])
dataset['Has_HDD'] = storage_features.apply(lambda x: x[1])
dataset['Has_Flash'] = storage_features.apply(lambda x: x[2])
dataset['Has_Hybrid'] = storage_features.apply(lambda x: x[3])
dataset['Storage_Capacity_GB'] = storage_features.apply(lambda x: x[4])
logger.info(f"Storage features extracted - SSD: {dataset['Has_SSD'].sum()}, HDD: {dataset['Has_HDD'].sum()}, Flash: {dataset['Has_Flash'].sum()}, Hybrid: {dataset['Has_Hybrid'].sum()}")

# Create derived storage features
logger.info("Creating derived storage type score (weighted by premium level)...")
dataset['Storage_Type_Score'] = (
    dataset['Has_SSD'] * 3 +      # SSD is premium
    dataset['Has_Flash'] * 2.5 +  # Flash is also premium
    dataset['Has_Hybrid'] * 2 +   # Hybrid is mid-range
    dataset['Has_HDD'] * 1        # HDD is budget
)

logger.info("Storage feature engineering complete.")
logger.debug(f"Sample storage features:\n{dataset[['Memory', 'Has_SSD', 'Has_HDD', 'Storage_Capacity_GB', 'Storage_Type_Score']].head()}")


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
logger.info("="*60)
logger.info("Creating advanced interaction features...")
logger.info("="*60)
interaction_count = 0

# Storage capacity * SSD indicator (SSD with high capacity is premium)
if 'Storage_Capacity_GB' in x.columns and 'Has_SSD' in x.columns:
    x['Premium_Storage'] = x['Storage_Capacity_GB'] * (x['Has_SSD'] + 1) / 1000  # Normalized
    logger.info("✓ Created Premium_Storage interaction feature")
    interaction_count += 1

# RAM * Storage Type Score (high RAM + fast storage = workstation/gaming)
if 'Ram' in x.columns and 'Storage_Type_Score' in x.columns:
    x['RAM_Storage_Quality'] = x['Ram'] * x['Storage_Type_Score']
    logger.info("✓ Created RAM_Storage_Quality interaction feature")
    interaction_count += 1

# Screen quality * Storage quality (premium display + premium storage)
if 'PPI' in x.columns and 'Storage_Type_Score' in x.columns:
    x['Display_Storage_Premium'] = x['PPI'] * x['Storage_Type_Score']
    logger.info("✓ Created Display_Storage_Premium interaction feature")
    interaction_count += 1

# Weight to size ratio (portability factor)
if 'Weight' in x.columns and 'Inches' in x.columns:
    x['Weight_Size_Ratio'] = x['Weight'] / x['Inches']
    logger.info("✓ Created Weight_Size_Ratio interaction feature")
    interaction_count += 1

# Total pixels per RAM (graphics capability estimation)
if 'Total_Pixels' in x.columns and 'Ram' in x.columns:
    x['Pixels_Per_RAM'] = x['Total_Pixels'] / (x['Ram'] * 1000000)
    logger.info("✓ Created Pixels_Per_RAM interaction feature")
    interaction_count += 1

# Storage per inch (how much storage per screen size)
if 'Storage_Capacity_GB' in x.columns and 'Inches' in x.columns:
    x['Storage_Per_Inch'] = x['Storage_Capacity_GB'] / x['Inches']
    logger.info("✓ Created Storage_Per_Inch interaction feature")
    interaction_count += 1

logger.info(f"Advanced interaction features created: {interaction_count} features. Total features: {x.shape[1]}")


# In[50]:


pip install scikit-learn


# In[51]:


from sklearn.model_selection import train_test_split
# Add random_state for reproducibility
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[52]:

logger.info("="*60)
logger.info("FEATURE SCALING - StandardScaler for linear models")
logger.info("="*60)

# IMPROVEMENT: Add feature scaling for linear models
from sklearn.preprocessing import StandardScaler

logger.info("Fitting StandardScaler on training data...")
# Create scaled versions for linear models
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logger.info("Transforming data and converting back to DataFrame...")
# Convert back to DataFrame for consistency
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

logger.info(f"Feature scaling complete. Scaled shape: {x_train_scaled_df.shape}")


# In[53]:


x_train.shape,x_test.shape


# In[53a]:

logger.info("="*60)
logger.info("OUTLIER DETECTION AND ANALYSIS")
logger.info("="*60)

# IMPROVEMENT: Basic outlier detection and reporting
from scipy import stats

logger.info("Detecting outliers in target variable using Z-score method (threshold=3)...")
# Detect outliers in target variable using Z-score
z_scores_target = np.abs(stats.zscore(y_train))
outliers_target = np.where(z_scores_target > 3)[0]

logger.info(f"Target variable (Price) outliers (Z-score > 3): {len(outliers_target)}")
if len(outliers_target) > 0:
    logger.debug(f"Outlier prices: {y_train.iloc[outliers_target].values[:5]} (showing first 5)")

# Detect outliers in key features
logger.info("Detecting outliers in key numeric features...")
key_numeric_features = ['Ram', 'Weight', 'Inches', 'Total_Pixels', 'Storage_Capacity_GB']
outlier_counts = {}

for feature in key_numeric_features:
    if feature in x_train.columns:
        z_scores = np.abs(stats.zscore(x_train[feature]))
        outliers = np.where(z_scores > 3)[0]
        outlier_counts[feature] = len(outliers)
        if len(outliers) > 0:
            logger.info(f"  {feature}: {len(outliers)} outliers detected")

logger.info("Note: Outliers are kept in the dataset as they may represent legitimate premium/budget laptops.")
logger.info("Tree-based models handle outliers well without removal.")


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

logger.info("="*60)
logger.info("MODEL COMPARISON - BASELINE MODELS")
logger.info("="*60)

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
logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - Random Forest")
logger.info("="*60)

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
logger.info(f"Parameter space: {len(rf_parameters)} parameters")

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

logger.info("Training Random Forest with RandomizedSearchCV (60 iterations, 5-fold CV)...")
grid_fit = grid_obj.fit(x_train, y_train)
logger.info("RandomizedSearchCV completed")

best_model = grid_fit.best_estimator_
logger.info(f"Best parameters: {grid_fit.best_params_}")
logger.info(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:

# IMPROVEMENT: Enhanced Gradient Boosting tuning
logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - Gradient Boosting")
logger.info("="*60)

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
logger.info(f"Parameter space: {len(gb_parameters)} parameters")

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

logger.info("Training Gradient Boosting with RandomizedSearchCV (60 iterations, 5-fold CV)...")
gb_fit = gb_search.fit(x_train, y_train)
logger.info("RandomizedSearchCV completed")

best_gb_model = gb_fit.best_estimator_
logger.info(f"Best parameters: {gb_fit.best_params_}")
logger.info(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:

# IMPROVEMENT: Add LightGBM hyperparameter tuning
logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - LightGBM")
logger.info("="*60)
best_lgb_model = None
try:
    import lightgbm as lgb
    
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
    logger.info(f"Parameter space: {len(lgb_parameters)} parameters")
    
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
    
    logger.info("Training LightGBM with RandomizedSearchCV (60 iterations, 5-fold CV)...")
    lgb_fit = lgb_search.fit(x_train, y_train)
    logger.info("RandomizedSearchCV completed")
    
    best_lgb_model = lgb_fit.best_estimator_
    logger.info(f"Best parameters: {lgb_fit.best_params_}")
    logger.info(f"Best CV score: {lgb_fit.best_score_:.4f}")
    
except ImportError:
    logger.warning("LightGBM not installed. Skipping LightGBM tuning.")


# In[58c]:

# Compare all best models
logger.info("="*60)
logger.info("FINAL MODEL COMPARISON")
logger.info("="*60)

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

logger.info("*"*60)
logger.info(f"WINNER: {best_model_name}")
logger.info(f"R² Score: {models_dict[best_model_name][1]:.4f}")
logger.info("*"*60)


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

logger.info("="*60)
logger.info("MODEL PERSISTENCE - Saving trained model")
logger.info("="*60)

import pickle
import os
import tempfile

# Save the best overall model with comprehensive error handling
pickle_filename = 'predictor.pickle'
temp_filename = None

logger.info(f"Saving best model ({best_model_name}) to '{pickle_filename}'...")
try:
    # Create a temporary file first to avoid corrupting existing file
    logger.debug("Creating temporary file for safe write operation...")
    temp_fd, temp_filename = tempfile.mkstemp(suffix='.pickle', dir='.')
    
    try:
        # Write to temporary file
        logger.debug(f"Writing model to temporary file: {temp_filename}")
        with os.fdopen(temp_fd, 'wb') as temp_file:
            pickle.dump(best_overall_model, temp_file)
        logger.debug("Model serialized successfully")
        
        # If successful, replace the target file
        # Remove existing file if it exists
        if os.path.exists(pickle_filename):
            logger.debug(f"Removing existing file: {pickle_filename}")
            try:
                os.remove(pickle_filename)
            except PermissionError:
                logger.error(f"Cannot remove existing file '{pickle_filename}'. Permission denied.")
                raise
        
        # Rename temp file to target file
        logger.debug(f"Renaming temporary file to: {pickle_filename}")
        os.rename(temp_filename, pickle_filename)
        temp_filename = None  # Mark as successfully moved
        
        # Verify the file was written correctly
        file_size = os.path.getsize(pickle_filename)
        logger.info(f"✓ Model successfully saved to '{pickle_filename}' ({file_size:,} bytes)")
        
    except Exception as e:
        # If anything fails, clean up temp file
        logger.error(f"Error during model save operation: {e}")
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

logger.info("="*60)
logger.info("MODEL PREDICTIONS - Sample predictions")
logger.info("="*60)

# Note: Predictions need to be updated with the new feature set
# The feature count has changed due to keeping 'Inches' and adding new features
logger.info(f"Number of features: {len(x_train.columns)}")
logger.info("Making sample predictions with best model...")
# Use actual test data for demonstration
sample_predictions = best_overall_model.predict(x_test[:5])
logger.info(f"Predictions for first 5 test samples: {sample_predictions}")
logger.info(f"Actual prices: {y_test.iloc[:5].values}")

# Calculate prediction errors
errors = sample_predictions - y_test.iloc[:5].values
logger.info(f"Prediction errors (predicted - actual): {errors}")
logger.info(f"Mean absolute error for samples: {np.abs(errors).mean():.2f} euros")

logger.info("="*60)
logger.info("LAPTOP PRICE PREDICTION MODEL - Execution Complete")
logger.info("="*60)


# In[ ]: