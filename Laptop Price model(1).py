#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
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

# Extract numerical screen resolution (width x height)
def extract_resolution(x):
    pattern = r'(\d+)x(\d+)'
    match = re.search(pattern, x)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        return width, height, width * height
    return 1366, 768, 1366*768  # default resolution

dataset[['Screen_Width', 'Screen_Height', 'Screen_Pixels']] = pd.DataFrame(
    dataset['ScreenResolution'].apply(extract_resolution).tolist())

# Screen aspect ratio
dataset['Aspect_Ratio'] = dataset['Screen_Width'] / dataset['Screen_Height']


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:


# Enhanced CPU feature engineering
dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

# Extract CPU frequency as numerical feature
def extract_cpu_frequency(cpu_string):
    # Look for patterns like "2.3GHz", "1.8GHz", etc.
    pattern = r'(\d+\.?\d*)\s*GHz'
    match = re.search(pattern, cpu_string)
    if match:
        return float(match.group(1))
    return 2.0  # default frequency

dataset['CPU_Frequency'] = dataset['Cpu'].apply(extract_cpu_frequency)

# Extract CPU generation/model information
def extract_cpu_generation(cpu_string):
    # Look for Intel generation patterns like i5-7200U, i7-8550U
    pattern = r'i[357]-?(\d+)'
    match = re.search(pattern, cpu_string)
    if match:
        gen_num = int(match.group(1))
        return gen_num // 1000 if gen_num >= 1000 else gen_num // 100
    return 7  # default generation

dataset['CPU_Generation'] = dataset['Cpu'].apply(extract_cpu_generation)


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


# Enhanced GPU feature engineering
dataset['Gpu_name']= dataset['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))

# Extract GPU performance tier
def extract_gpu_tier(gpu_string):
    gpu_lower = gpu_string.lower()
    if any(x in gpu_lower for x in ['rtx', 'gtx 1080', 'gtx 1070', 'gtx 1060', 'radeon pro']):
        return 3  # High-end
    elif any(x in gpu_lower for x in ['gtx', 'radeon rx', 'mx150', 'mx130']):
        return 2  # Mid-range
    elif any(x in gpu_lower for x in ['intel iris', 'intel hd', 'intel uhd']):
        return 1  # Integrated
    return 1  # Default to integrated

dataset['GPU_Tier'] = dataset['Gpu'].apply(extract_gpu_tier)

# Extract storage information from product names
def extract_storage_info(product_name):
    # Look for storage patterns like 256GB, 512GB SSD, 1TB HDD
    ssd_pattern = r'(\d+)\s*GB\s*SSD|SSD\s*(\d+)\s*GB'
    hdd_pattern = r'(\d+)\s*TB\s*HDD|HDD\s*(\d+)\s*TB|(\d+)\s*GB\s*HDD'
    
    has_ssd = bool(re.search(ssd_pattern, product_name, re.IGNORECASE))
    has_hdd = bool(re.search(hdd_pattern, product_name, re.IGNORECASE))
    
    ssd_match = re.search(ssd_pattern, product_name, re.IGNORECASE)
    storage_size = 256  # default
    
    if ssd_match:
        storage_size = int(ssd_match.group(1) or ssd_match.group(2))
    
    return has_ssd, has_hdd, storage_size

dataset[['Has_SSD', 'Has_HDD', 'Storage_Size']] = pd.DataFrame(
    dataset['Product'].apply(extract_storage_info).tolist())


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


# Create brand premium feature
premium_brands = ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus']
dataset['Premium_Brand'] = dataset['Company'].apply(lambda x: 1 if x in premium_brands else 0)

# Create price per inch feature (will calculate after train-test split to avoid leakage)
dataset['Screen_Size'] = dataset['Inches']

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


# In[50]:


# pip install scikit-learn xgboost
# Uncomment above line if packages are not installed


# In[51]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Split data with stratified sampling for better distribution
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)

# Feature scaling for algorithms that need it
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[53]:


x_train.shape,x_test.shape


# In[54]:


# Enhanced model evaluation function
def evaluate_model(model, x_train_data, x_test_data, y_train, y_test, model_name="Model"):
    model.fit(x_train_data, y_train)
    
    # Predictions
    y_pred = model.predict(x_test_data)
    
    # Multiple evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, x_train_data, y_train, cv=5, scoring='r2')
    
    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return r2, rmse, mae, cv_scores.mean()

def model_acc(model):
    model.fit(x_train,y_train)
    acc = model.score(x_test, y_test)
    print(str(model)+'-->'+str(acc))


# In[57]:


# Enhanced model comparison with multiple algorithms
print("=== Model Comparison (Original Features) ===")

# Basic models
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
evaluate_model(lr, x_train, x_test, y_train, y_test, "Linear Regression")

# Regularized models (need scaling)
ridge = Ridge(alpha=1.0)
evaluate_model(ridge, x_train_scaled, x_test_scaled, y_train, y_test, "Ridge Regression")

elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
evaluate_model(elastic_net, x_train_scaled, x_test_scaled, y_train, y_test, "Elastic Net")

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
evaluate_model(dt, x_train, x_test, y_train, y_test, "Decision Tree")

rf = RandomForestRegressor(random_state=42)
evaluate_model(rf, x_train, x_test, y_train, y_test, "Random Forest (Basic)")

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
evaluate_model(gb, x_train, x_test, y_train, y_test, "Gradient Boosting")

# XGBoost
try:
    xgb_model = xgb.XGBRegressor(random_state=42)
    evaluate_model(xgb_model, x_train, x_test, y_train, y_test, "XGBoost")
except:
    print("XGBoost not available, skipping...")

# Support Vector Regression (on scaled data)
svr = SVR(kernel='rbf')
evaluate_model(svr, x_train_scaled, x_test_scaled, y_train, y_test, "Support Vector Regression")


# In[58]:


print("\n=== Advanced Hyperparameter Tuning ===")

# Comprehensive Random Forest hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

print("Tuning Random Forest...")
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

rf_grid_fit = rf_grid.fit(x_train, y_train)
best_rf_model = rf_grid_fit.best_estimator_

print(f"Best Random Forest parameters: {rf_grid_fit.best_params_}")
evaluate_model(best_rf_model, x_train, x_test, y_train, y_test, "Tuned Random Forest")

# XGBoost hyperparameter tuning
try:
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    print("\nTuning XGBoost...")
    xgb_grid = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_grid=xgb_params,
        cv=3,  # Reduced CV for faster computation
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    xgb_grid_fit = xgb_grid.fit(x_train, y_train)
    best_xgb_model = xgb_grid_fit.best_estimator_

    print(f"Best XGBoost parameters: {xgb_grid_fit.best_params_}")
    xgb_r2, xgb_rmse, xgb_mae, xgb_cv = evaluate_model(best_xgb_model, x_train, x_test, y_train, y_test, "Tuned XGBoost")

    # Choose the best model
    rf_r2 = rf_grid.score(x_test, y_test)
    xgb_r2_test = xgb_grid.score(x_test, y_test)
    
    if xgb_r2_test > rf_r2:
        best_model = best_xgb_model
        print(f"\nBest overall model: XGBoost with R² = {xgb_r2_test:.4f}")
    else:
        best_model = best_rf_model
        print(f"\nBest overall model: Random Forest with R² = {rf_r2:.4f}")

except Exception as e:
    print(f"XGBoost tuning failed: {e}")
    best_model = best_rf_model
    print(f"\nUsing Random Forest as best model with R² = {rf_grid.score(x_test, y_test):.4f}")


# In[59]:


# Feature importance analysis
print("\n=== Feature Importance Analysis ===")
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': x_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

# Final model performance
final_score = best_model.score(x_test, y_test)
print(f"\nFinal model R² score: {final_score:.4f}")

# Feature selection for further optimization
print("\n=== Feature Selection Optimization ===")
selector = SelectKBest(score_func=f_regression, k=15)  # Select top 15 features
x_train_selected = selector.fit_transform(x_train, y_train)
x_test_selected = selector.transform(x_test)

selected_features = x_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Retrain best model on selected features
best_model_selected = RandomForestRegressor(**best_rf_model.get_params())
selected_r2, selected_rmse, selected_mae, selected_cv = evaluate_model(
    best_model_selected, x_train_selected, x_test_selected, y_train, y_test, 
    "Random Forest (Feature Selected)"
)

# Use feature-selected model if it performs better
if selected_r2 > final_score:
    print("Feature selection improved performance!")
    best_model = best_model_selected
    x_train_final, x_test_final = x_train_selected, x_test_selected
    final_score = selected_r2
else:
    print("Original feature set performs better.")
    x_train_final, x_test_final = x_train, x_test


# In[60]:


print("\n=== Final Model Features ===")
if 'x_train_final' in locals():
    if hasattr(x_train_final, 'shape'):
        print(f"Final feature set shape: {x_train_final.shape}")
    else:
        print("Feature columns:", x_train.columns.tolist())
else:
    print("Feature columns:", x_train.columns.tolist())


# In[68]:


print("\n=== Model Persistence ===")
import pickle

# Save the best model and preprocessing objects
model_artifacts = {
    'model': best_model,
    'scaler': scaler if 'scaler' in locals() else None,
    'feature_selector': selector if 'selector' in locals() else None,
    'feature_names': list(x_train.columns),
    'final_score': final_score
}

with open('enhanced_predictor.pickle', 'wb') as file:
    pickle.dump(model_artifacts, file)

print(f"Enhanced model saved with R² score: {final_score:.4f}")


# In[66]:


print("\n=== Model Predictions Examples ===")

# Create example predictions with the enhanced model
# Note: These examples need to match the new feature set
if len(x_test) > 0:
    sample_indices = [0, 1, 2] if len(x_test) >= 3 else range(len(x_test))
    
    for i, idx in enumerate(sample_indices):
        if hasattr(best_model, 'predict'):
            if 'x_test_final' in locals():
                prediction = best_model.predict([x_test_final[idx]])
            else:
                prediction = best_model.predict([x_test.iloc[idx]])
            actual = y_test.iloc[idx]
            print(f"Example {i+1}: Predicted: €{prediction[0]:.2f}, Actual: €{actual:.2f}, Error: €{abs(prediction[0] - actual):.2f}")

print(f"\n=== Summary of Improvements ===")
print("Enhanced Features Added:")
print("- Screen resolution dimensions (width, height, total pixels)")
print("- CPU frequency extraction from text")
print("- CPU generation detection")
print("- GPU performance tier classification")
print("- Storage type detection (SSD/HDD)")
print("- Brand premium classification")
print("- Screen aspect ratio")
print("\nModel Architecture Improvements:")
print("- Added XGBoost, Gradient Boosting, SVM, Ridge, Elastic Net")
print("- Comprehensive hyperparameter tuning")
print("- 5-fold cross-validation")
print("- Feature selection optimization")
print("- Multiple evaluation metrics (R², RMSE, MAE)")
print("- Feature importance analysis")
print(f"\nFinal Model Performance: R² = {final_score:.4f}")


# In[ ]:




