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

# PERFORMANCE OPTIMIZATION: Extract actual screen resolution (width x height)
# Previously this function was called 3 times per row, now called once
import re

def extract_resolution(res_string):
    # Find pattern like "1920x1080" or "3840x2160"
    match = re.search(r'(\d+)x(\d+)', res_string)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height, width * height  # width, height, total pixels
    return 1366, 768, 1366*768  # default resolution if not found

# OPTIMIZED: Call extract_resolution once per row instead of 3 times
# This reduces regex operations from 3N to N (3x speedup for this section)
resolution_data = dataset['ScreenResolution'].apply(extract_resolution)
dataset['Screen_Width'] = resolution_data.apply(lambda x: x[0])
dataset['Screen_Height'] = resolution_data.apply(lambda x: x[1])
dataset['Total_Pixels'] = resolution_data.apply(lambda x: x[2])

# Calculate PPI (Pixels Per Inch) - important quality metric
dataset['PPI'] = np.sqrt(dataset['Total_Pixels']) / dataset['Inches']


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


# In[37]:


# Keep screen size and drop only redundant columns
dataset=dataset.drop(columns=['laptop_ID','Product','ScreenResolution','Cpu','Gpu'])


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


# In[50]:


pip install scikit-learn


# In[51]:


from sklearn.model_selection import train_test_split
# Add random_state for reproducibility
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[53]:


x_train.shape,x_test.shape


# In[54]:


# Improved evaluation function with multiple metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

def model_acc(model, model_name="Model"):
    model.fit(x_train,y_train)
    
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
    
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} euros")
    print(f"  RMSE: {rmse:.2f} euros")
    print(f"  CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return r2, mae, rmse, cv_mean


# In[57]:


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
    print("\nXGBoost not installed. Install with: pip install xgboost")


# In[58]:


from sklearn.model_selection import GridSearchCV

# Comprehensive hyperparameter tuning for Random Forest
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
grid_fit = grid_obj.fit(x_train, y_train)

best_model = grid_fit.best_estimator_
print(f"\nBest parameters: {grid_fit.best_params_}")
print(f"Best CV score: {grid_fit.best_score_:.4f}")
best_model


# In[58a]:


# Also tune Gradient Boosting
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
gb_fit = gb_search.fit(x_train, y_train)

best_gb_model = gb_fit.best_estimator_
print(f"\nBest parameters: {gb_fit.best_params_}")
print(f"Best CV score: {gb_fit.best_score_:.4f}")


# In[58b]:


# Compare best models
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

rf_r2, rf_mae, rf_rmse, rf_cv = model_acc(best_model, "Best Random Forest")
gb_r2, gb_mae, gb_rmse, gb_cv = model_acc(best_gb_model, "Best Gradient Boosting")

# Select the best overall model
if gb_r2 > rf_r2:
    best_overall_model = best_gb_model
    print(f"\n{'*'*60}")
    print("WINNER: Gradient Boosting")
    print(f"{'*'*60}")
else:
    best_overall_model = best_model
    print(f"\n{'*'*60}")
    print("WINNER: Random Forest")
    print(f"{'*'*60}")


# In[59]:


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




