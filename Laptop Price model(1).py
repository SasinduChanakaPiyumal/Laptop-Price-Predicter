#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# In[2]:

logger.info("="*60)
logger.info("LAPTOP PRICE PREDICTION MODEL - Starting Pipeline")
logger.info("="*60)

logger.info("Loading dataset from laptop_price.csv...")
dataset = pd.read_csv("laptop_price.csv",encoding = 'latin-1')
logger.info(f"Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")


# In[3]:


dataset.head()


# In[4]:

logger.debug(f"Dataset shape: {dataset.shape}")
dataset.shape


# In[5]:


dataset.describe()


# In[6]:

logger.info("Dataset info:")
dataset.info()
null_counts = dataset.isnull().sum()
total_nulls = null_counts.sum()
logger.info(f"Total null values in dataset: {total_nulls}")


# In[7]:


dataset.isnull().sum()


# In[8]:

logger.info("="*60)
logger.info("DATA PREPROCESSING")
logger.info("="*60)

logger.info("Converting RAM column: removing 'GB' suffix and converting to int32...")
dataset['Ram']=dataset['Ram'].str.replace('GB','').astype('int32')
logger.debug(f"RAM conversion complete. Range: {dataset['Ram'].min()} - {dataset['Ram'].max()} GB")


# In[9]:


dataset.head()


# In[10]:

logger.info("Converting Weight column: removing 'kg' suffix and converting to float64...")
dataset['Weight']=dataset['Weight'].str.replace('kg','').astype('float64')
logger.debug(f"Weight conversion complete. Range: {dataset['Weight'].min():.2f} - {dataset['Weight'].max():.2f} kg")


# In[11]:


dataset.head(2)


# In[12]:

logger.info("Identifying non-numeric columns for feature engineering...")
non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
logger.info(f"Non-numeric columns: {list(non_numeric_columns)}")
print(non_numeric_columns)


# In[13]:

logger.info("Computing correlation of numeric features with Price_euros...")
numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']
logger.debug(f"Top correlations with price: Ram={correlation.get('Ram', 0):.3f}, Weight={correlation.get('Weight', 0):.3f}")


# In[14]:


correlation


# In[15]:


dataset['Company'].value_counts()


# In[16]:

logger.info("="*60)
logger.info("FEATURE ENGINEERING")
logger.info("="*60)

logger.info("Consolidating Company names: grouping low-frequency brands into 'Other'...")
def add_company(inpt):
    if inpt == 'Samsung'or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt
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

logger.info("Extracting screen features from ScreenResolution...")
# Enhanced feature engineering from ScreenResolution
dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
logger.debug(f"Touchscreen laptops: {dataset['Touchscreen'].sum()}, IPS displays: {dataset['IPS'].sum()}")

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

logger.info("Extracting screen resolution dimensions (width, height, total pixels)...")
dataset['Screen_Width'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[0])
dataset['Screen_Height'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[1])
dataset['Total_Pixels'] = dataset['ScreenResolution'].apply(lambda x: extract_resolution(x)[2])

# Calculate PPI (Pixels Per Inch) - important quality metric
logger.info("Calculating PPI (Pixels Per Inch) from screen dimensions...")
dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']
logger.debug(f"PPI range: {dataset['PPI'].min():.1f} - {dataset['PPI'].max():.1f}")


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:

logger.info("Processing CPU information: extracting processor names...")
dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[24]:


dataset['Cpu_name'].value_counts()


# In[25]:

logger.info("Categorizing CPU names into major groups (Intel i3/i5/i7, AMD, Other)...")
def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)
logger.info(f"CPU categorization complete. Categories: {dataset['Cpu_name'].unique()}")


# In[26]:


dataset['Cpu_name'].value_counts()


# In[27]:

logger.info("Processing GPU information: extracting GPU brand names...")
dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))


# In[30]:


dataset['Gpu_name'].value_counts()


# In[29]:

logger.info("Filtering out ARM GPUs from dataset...")
initial_count = len(dataset)
dataset = dataset[dataset['Gpu_name'] != 'ARM']
logger.info(f"Removed {initial_count - len(dataset)} ARM GPU entries. Remaining: {len(dataset)} rows")


# In[32]:


dataset.head(2)


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:

logger.info("Categorizing Operating Systems (Windows, Mac, Linux, Other)...")
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
logger.info(f"OS categorization complete. Categories: {dataset['OpSys'].unique()}")


# In[37]:

logger.info("Dropping redundant columns: laptop_ID, Product, ScreenResolution, Cpu, Gpu...")
# Keep screen size and drop only redundant columns
dataset=dataset.drop(columns=['laptop_ID','Product','ScreenResolution','Cpu','Gpu'])
logger.info(f"Dataset shape after dropping columns: {dataset.shape}")


# In[38]:


dataset.head()


# In[39]:

logger.info("="*60)
logger.info("FEATURE ENCODING AND INTERACTION FEATURES")
logger.info("="*60)

logger.info("Applying one-hot encoding to categorical variables...")
dataset = pd.get_dummies(dataset)
logger.info(f"Dataset shape after encoding: {dataset.shape}")


# In[40]:


dataset.head()


# In[41]:

logger.info("Splitting features and target variable...")
x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']
logger.info(f"Features shape: {x.shape}, Target shape: {y.shape}")


# In[42]:

logger.info("Creating interaction and polynomial features...")
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
    logger.debug("Added Ram_squared feature")
    
# Add interaction between screen quality and size
if 'Total_Pixels' in x.columns and 'Inches' in x.columns:
    x['Screen_Quality'] = x['Total_Pixels'] / 1000000 * x['Inches']  # Normalized quality metric
    logger.debug("Added Screen_Quality interaction feature")

logger.info(f"Final feature set size: {x.shape[1]} features")


# In[50]:


pip install scikit-learn


# In[51]:


logger.info("="*60)
logger.info("TRAIN/TEST SPLIT")
logger.info("="*60)

from sklearn.model_selection import train_test_split
# Add random_state for reproducibility
logger.info("Splitting data into train (75%) and test (25%) sets...")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)
logger.info(f"Train set: {x_train.shape}, Test set: {x_test.shape}")


# In[53]:


x_train.shape,x_test.shape


# In[54]:


# Improved evaluation function with multiple metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

def model_acc(model, model_name="Model"):
    logger.info(f"Training {model_name}...")
    model.fit(x_train,y_train)
    
    # Test set predictions
    y_pred = model.predict(x_test)
    
    # Calculate multiple metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"Running 5-fold cross-validation for {model_name}...")
    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  MAE: {mae:.2f} euros")
    logger.info(f"  RMSE: {rmse:.2f} euros")
    logger.info(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return r2, mae, rmse, cv_mean


# In[57]:

logger.info("="*60)
logger.info("MODEL TRAINING AND COMPARISON")
logger.info("="*60)

logger.info("Testing baseline models...")

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


# In[58]:


from sklearn.model_selection import GridSearchCV

logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - Random Forest")
logger.info("="*60)

# Comprehensive hyperparameter tuning for Random Forest
rf_parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

logger.info(f"Parameter grid size: {len(rf_parameters)} parameters")
logger.debug(f"Parameters: {rf_parameters}")

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

logger.info("Starting RandomizedSearchCV with 50 iterations, 5-fold CV...")
logger.info("This may take several minutes...")
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
logger.info(f"Best parameters: {grid_fit.best_params_}")
logger.info(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:

logger.info("="*60)
logger.info("HYPERPARAMETER TUNING - Gradient Boosting")
logger.info("="*60)

# Also tune Gradient Boosting
gb_parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

logger.info(f"Parameter grid size: {len(gb_parameters)} parameters")
logger.debug(f"Parameters: {gb_parameters}")

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

logger.info("Starting RandomizedSearchCV with 50 iterations, 5-fold CV...")
logger.info("This may take several minutes...")
gb_fit = gb_search.fit(x_train, y_train)

best_gb_model = gb_fit.best_estimator_
logger.info(f"Best parameters: {gb_fit.best_params_}")
logger.info(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:

logger.info("="*60)
logger.info("FINAL MODEL COMPARISON")
logger.info("="*60)

# Compare best models
logger.info("Evaluating tuned Random Forest model...")
rf_r2, rf_mae, rf_rmse, rf_cv = model_acc(best_model, "Best Random Forest")

logger.info("Evaluating tuned Gradient Boosting model...")
gb_r2, gb_mae, gb_rmse, gb_cv = model_acc(best_gb_model, "Best Gradient Boosting")

# Select the best overall model
logger.info("Selecting best overall model based on R² score...")
if gb_r2 > rf_r2:
    best_overall_model = best_gb_model
    logger.info("*"*60)
    logger.info("WINNER: Gradient Boosting")
    logger.info(f"R² improvement over Random Forest: {(gb_r2 - rf_r2):.4f}")
    logger.info("*"*60)
else:
    best_overall_model = best_model
    logger.info("*"*60)
    logger.info("WINNER: Random Forest")
    logger.info(f"R² improvement over Gradient Boosting: {(rf_r2 - gb_r2):.4f}")
    logger.info("*"*60)


# In[59]:

logger.info("="*60)
logger.info("FEATURE IMPORTANCE ANALYSIS")
logger.info("="*60)

# Feature importance analysis
logger.info("Analyzing feature importances from best model...")
feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': best_overall_model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("\nTop 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

# Final model performance
logger.info("="*60)
logger.info("FINAL MODEL PERFORMANCE")
logger.info("="*60)
final_score = best_overall_model.score(x_test, y_test)
logger.info(f"Test R² Score: {final_score:.4f}")


# In[60]:


x_train.columns


# In[68]:

logger.info("="*60)
logger.info("MODEL SAVING AND PREDICTIONS")
logger.info("="*60)

import pickle
# Save the best overall model (could be RF or GB)
logger.info("Saving best model to predictor.pickle...")
with open('predictor.pickle','wb') as file:
    pickle.dump(best_overall_model,file)
    
logger.info("Model saved successfully to predictor.pickle")


# In[66]:

logger.info(f"Number of features in final model: {len(x_train.columns)}")
logger.info("Generating sample predictions...")

# Note: Predictions need to be updated with the new feature set
# The feature count has changed due to keeping 'Inches' and adding new features
# Use actual test data for demonstration
sample_predictions = best_overall_model.predict(x_test[:5])
logger.info("Sample predictions (first 5 test samples):")
for i in range(5):
    logger.info(f"  Sample {i+1}: Predicted=${sample_predictions[i]:.2f}, Actual=${y_test.iloc[i]:.2f}, Error=${abs(sample_predictions[i] - y_test.iloc[i]):.2f}")

logger.info("="*60)
logger.info("PIPELINE COMPLETE")
logger.info("="*60)


# In[ ]:




