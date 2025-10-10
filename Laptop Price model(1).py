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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'laptop_price_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("LAPTOP PRICE PREDICTION MODEL - EXECUTION STARTED")
logger.info("="*60)


# In[2]:

logger.info("\n" + "="*60)
logger.info("PHASE 1: DATA LOADING AND EXPLORATION")
logger.info("="*60)

try:
    dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')
    logger.info(f"Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
except FileNotFoundError:
    logger.error("Dataset file 'laptop_price.csv' not found!")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise


# In[3]:


dataset.head()


# In[4]:

logger.debug(f"Dataset shape: {dataset.shape}")


dataset.shape


# In[5]:

logger.info("Dataset statistics computed")

dataset.describe()


# In[6]:

logger.info(f"Dataset info - dtypes: {dataset.dtypes.value_counts().to_dict()}")

dataset.info()


# In[7]:

null_counts = dataset.isnull().sum()
total_nulls = null_counts.sum()
if total_nulls > 0:
    logger.warning(f"Found {total_nulls} null values in dataset")
    logger.warning(f"Null counts by column: {null_counts[null_counts > 0].to_dict()}")
else:
    logger.info("No null values found in dataset")

dataset.isnull().sum()


# In[8]:

logger.info("\n" + "="*60)
logger.info("PHASE 2: DATA PREPROCESSING")
logger.info("="*60)

try:
    dataset['Ram'] = dataset['Ram'].str.replace('GB','').astype('int32')
    logger.info("Converted 'Ram' column from string to int32")
except Exception as e:
    logger.error(f"Error processing 'Ram' column: {str(e)}")
    raise


# In[9]:


dataset.head()


# In[10]:

try:
    dataset['Weight'] = dataset['Weight'].str.replace('kg','').astype('float64')
    logger.info("Converted 'Weight' column from string to float64")
except Exception as e:
    logger.error(f"Error processing 'Weight' column: {str(e)}")
    raise


# In[11]:


dataset.head(2)


# In[12]:


non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
logger.info(f"Non-numeric columns identified: {list(non_numeric_columns)}")
print(non_numeric_columns)


# In[13]:


numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']


# In[14]:


correlation


# In[15]:


dataset['Company'].value_counts()


# In[16]:

logger.info("Grouping low-frequency companies into 'Other' category")

def add_company(inpt):
    if inpt == 'Samsung'or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt

before_count = dataset['Company'].nunique()
dataset['Company'] = dataset['Company'].apply(add_company)
after_count = dataset['Company'].nunique()
logger.info(f"Company categories reduced from {before_count} to {after_count}")


# In[17]:


dataset['Company'].value_counts()


# In[18]:


len(dataset['Product'].value_counts())


# In[19]:


dataset['TypeName'].value_counts()


# In[20]:


dataset['ScreenResolution'].value_counts()


# In[21]:

logger.info("\n" + "="*60)
logger.info("PHASE 3: FEATURE ENGINEERING")
logger.info("="*60)

# Enhanced feature engineering from ScreenResolution
logger.info("Extracting screen features from ScreenResolution")
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
logger.info(f"  - Touchscreen devices: {dataset['Touchscreen'].sum()}")
logger.info(f"  - IPS displays: {dataset['IPS'].sum()}")

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
logger.info(f"  - Screen resolution features created (Width, Height, Total_Pixels)")
logger.info(f"  - Average resolution: {int(dataset['Screen_Width'].mean())}x{int(dataset['Screen_Height'].mean())}")

# Calculate PPI (Pixels Per Inch) - important quality metric
dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']
logger.info(f"  - PPI calculated (mean: {dataset['PPI'].mean():.2f})")


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:

logger.info("Processing CPU information")
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

before_cpu_count = dataset['Cpu_name'].nunique()
dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)
after_cpu_count = dataset['Cpu_name'].nunique()
logger.info(f"  - CPU categories reduced from {before_cpu_count} to {after_cpu_count}")
logger.info(f"  - CPU distribution: {dataset['Cpu_name'].value_counts().to_dict()}")


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:

logger.info("Processing GPU information")
dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))


# In[30]:


dataset['Gpu_name'].value_counts()


# In[29]:

rows_before = len(dataset)
dataset = dataset[dataset['Gpu_name'] != 'ARM']
rows_after = len(dataset)
logger.info(f"  - Filtered out ARM GPUs: {rows_before - rows_after} rows removed")
logger.info(f"  - Remaining GPU categories: {dataset['Gpu_name'].nunique()}")


# In[32]:


dataset.head(2)


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:

logger.info("Processing Operating System information")

def set_os(inpt):
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'

before_os_count = dataset['OpSys'].nunique()
dataset['OpSys']= dataset['OpSys'].apply(set_os)
after_os_count = dataset['OpSys'].nunique()
logger.info(f"  - OS categories reduced from {before_os_count} to {after_os_count}")
logger.info(f"  - OS distribution: {dataset['OpSys'].value_counts().to_dict()}")


# In[37]:


# Keep screen size and drop only redundant columns
cols_to_drop = ['laptop_ID','Product','ScreenResolution','Cpu','Gpu']
logger.info(f"Dropping redundant columns: {cols_to_drop}")
dataset=dataset.drop(columns=cols_to_drop)
logger.info(f"Dataset shape after dropping columns: {dataset.shape}")


# In[38]:


dataset.head()


# In[39]:

logger.info("Applying one-hot encoding to categorical variables")
cols_before = dataset.shape[1]
dataset = pd.get_dummies(dataset)
cols_after = dataset.shape[1]
logger.info(f"  - Columns expanded from {cols_before} to {cols_after} after one-hot encoding")


# In[40]:


dataset.head()


# In[41]:

logger.info("Splitting features and target variable")
x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']
logger.info(f"  - Features shape: {x.shape}")
logger.info(f"  - Target shape: {y.shape}")
logger.info(f"  - Target range: {y.min():.2f} to {y.max():.2f} euros (mean: {y.mean():.2f})")


# In[42]:

logger.info("Creating interaction and polynomial features")
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
    logger.info("  - Added Ram_squared feature")
    
# Add interaction between screen quality and size
if 'Total_Pixels' in x.columns and 'Inches' in x.columns:
    x['Screen_Quality'] = x['Total_Pixels'] / 1000000 * x['Inches']  # Normalized quality metric
    logger.info("  - Added Screen_Quality interaction feature")

logger.info(f"  - Final feature count: {x.shape[1]}")


# In[50]:


pip install scikit-learn


# In[51]:

logger.info("\n" + "="*60)
logger.info("PHASE 4: MODEL TRAINING AND EVALUATION")
logger.info("="*60)

from sklearn.model_selection import train_test_split
# Add random_state for reproducibility
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)
logger.info(f"Train-test split completed (75-25 split)")
logger.info(f"  - Training set: {x_train.shape[0]} samples")
logger.info(f"  - Test set: {x_test.shape[0]} samples")


# In[53]:


x_train.shape,x_test.shape


# In[54]:


# Improved evaluation function with multiple metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

def model_acc(model, model_name="Model"):
    logger.info(f"\nTraining {model_name}...")
    
    try:
        model.fit(x_train,y_train)
        logger.info(f"  - {model_name} training completed")
    except Exception as e:
        logger.error(f"  - Error training {model_name}: {str(e)}")
        raise
    
    # Test set predictions
    y_pred = model.predict(x_test)
    
    # Calculate multiple metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Log results
    logger.info(f"  - R² Score: {r2:.4f}")
    logger.info(f"  - MAE: {mae:.2f} euros")
    logger.info(f"  - RMSE: {rmse:.2f} euros")
    logger.info(f"  - CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} euros")
    print(f"  RMSE: {rmse:.2f} euros")
    print(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return r2, mae, rmse, cv_mean


# In[57]:

logger.info("\nEvaluating baseline models:")

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr, "Linear Regression")

from sklearn.linear_model import Lasso
lasso = Lasso(random_state=42)
model_acc(lasso, "Lasso Regression")

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
model_acc(dt, "Decision Tree")

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
model_acc(rf, "Random Forest (default)")

# Add Gradient Boosting models for better performance
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(random_state=42)
model_acc(gb, "Gradient Boosting")

# Add XGBoost if available (often performs best)
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    model_acc(xgb_model, "XGBoost")
except ImportError:
    logger.warning("XGBoost not installed. Install with: pip install xgboost")
    print("\nXGBoost not installed. Install with: pip install xgboost")


# In[58]:

logger.info("\n" + "="*60)
logger.info("PHASE 5: HYPERPARAMETER TUNING")
logger.info("="*60)

from sklearn.model_selection import GridSearchCV

# Comprehensive hyperparameter tuning for Random Forest
logger.info("\nStarting Random Forest hyperparameter tuning")
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - Random Forest")
print("="*60)

rf_parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
logger.info(f"  - Parameter grid: {len(rf_parameters)} parameters")
logger.info(f"  - Testing 50 random combinations with 5-fold CV")

# Use RandomizedSearchCV for efficiency with many parameters
from sklearn.model_selection import RandomizedSearchCV
grid_obj = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_parameters,
    n_iter=50,  # Test 50 random combinations
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

print("\nTraining Random Forest with RandomizedSearchCV...")
logger.info("  - Training in progress (this may take several minutes)...")
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
logger.info(f"  - Random Forest tuning completed")
logger.info(f"  - Best parameters: {grid_fit.best_params_}")
logger.info(f"  - Best CV score: {grid_fit.best_score_:.4f}")
print(f"\nBest parameters: {grid_fit.best_params_}")
print(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:


# Also tune Gradient Boosting
logger.info("\nStarting Gradient Boosting hyperparameter tuning")
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - Gradient Boosting")
print("="*60)

gb_parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}
logger.info(f"  - Parameter grid: {len(gb_parameters)} parameters")
logger.info(f"  - Testing 50 random combinations with 5-fold CV")

gb_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=gb_parameters,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Gradient Boosting with RandomizedSearchCV...")
logger.info("  - Training in progress (this may take several minutes)...")
gb_fit = gb_search.fit(x_train, y_train)

best_gb_model = gb_fit.best_estimator_
logger.info(f"  - Gradient Boosting tuning completed")
logger.info(f"  - Best parameters: {gb_fit.best_params_}")
logger.info(f"  - Best CV score: {gb_fit.best_score_:.4f}")
print(f"\nBest parameters: {gb_fit.best_params_}")
print(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:


# Compare best models
logger.info("\nComparing best models from tuning")
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

rf_r2, rf_mae, rf_rmse, rf_cv = model_acc(best_model, "Best Random Forest")
gb_r2, gb_mae, gb_rmse, gb_cv = model_acc(best_gb_model, "Best Gradient Boosting")

# Select the best overall model
if gb_r2 > rf_r2:
    best_overall_model = best_gb_model
    winner = "Gradient Boosting"
    logger.info(f"\n{'*'*60}")
    logger.info(f"WINNER: {winner} (R² = {gb_r2:.4f})")
    logger.info(f"{'*'*60}")
    print(f"\n{'*'*60}")
    print(f"WINNER: {winner}")
    print(f"{'*'*60}")
else:
    best_overall_model = best_model
    winner = "Random Forest"
    logger.info(f"\n{'*'*60}")
    logger.info(f"WINNER: {winner} (R² = {rf_r2:.4f})")
    logger.info(f"{'*'*60}")
    print(f"\n{'*'*60}")
    print(f"WINNER: {winner}")
    print(f"{'*'*60}")


# In[59]:


# Feature importance analysis
logger.info("\nAnalyzing feature importance")
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': best_overall_model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("\nTop 15 Most Important Features:")
top_features = feature_importance.head(15)
for idx, row in top_features.iterrows():
    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Final model performance
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE")
print("="*60)
final_score = best_overall_model.score(x_test, y_test)
logger.info(f"\nFinal model test R² score: {final_score:.4f}")
print(f"Test R² Score: {final_score:.4f}")


# In[60]:


x_train.columns


# In[68]:

logger.info("\n" + "="*60)
logger.info("PHASE 6: MODEL PERSISTENCE AND FINAL PREDICTIONS")
logger.info("="*60)

import pickle
# Save the best overall model (could be RF or GB)
try:
    with open('predictor.pickle','wb') as file:
        pickle.dump(best_overall_model,file)
    logger.info("Model successfully saved to 'predictor.pickle'")
    print("\nModel saved to predictor.pickle")
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")
    raise


# In[66]:


# Note: Predictions need to be updated with the new feature set
# The feature count has changed due to keeping 'Inches' and adding new features
logger.info(f"\nFinal model summary:")
logger.info(f"  - Number of features: {len(x_train.columns)}")
logger.info(f"  - Model type: {type(best_overall_model).__name__}")

print(f"\nNumber of features: {len(x_train.columns)}")
print("Sample prediction with best model:")

# Use actual test data for demonstration
sample_predictions = best_overall_model.predict(x_test[:5])
actual_prices = y_test.iloc[:5].values

logger.info("\nSample predictions on test data:")
for i, (pred, actual) in enumerate(zip(sample_predictions, actual_prices)):
    error = abs(pred - actual)
    error_pct = (error / actual) * 100
    logger.info(f"  Sample {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={error:.2f} ({error_pct:.1f}%)")

print(f"Predictions for first 5 test samples: {sample_predictions}")
print(f"Actual prices: {actual_prices}")

logger.info("\n" + "="*60)
logger.info("MODEL EXECUTION COMPLETED SUCCESSFULLY")
logger.info("="*60)


# In[ ]:




