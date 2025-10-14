#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv("laptop_price.csv",encoding = 'latin-1')


# In[3]:


# dataset.head() removed for performance


# In[4]:


# dataset.shape removed for performance


# In[5]:


# dataset.describe() removed for performance


# In[6]:


# dataset.info() removed for performance


# In[7]:


dataset.isnull().sum()


# In[8]:


# Use smaller integer dtype for memory efficiency
dataset['Ram']=dataset['Ram'].str.replace('GB','').astype('int16')


# In[9]:


# dataset.head() removed for performance


# In[10]:


# Use smaller float dtype for memory efficiency
dataset['Weight']=dataset['Weight'].str.replace('kg','').astype('float32')


# In[11]:


# dataset.head(2) removed for performance


# In[12]:


non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)


# In[13]:


numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']


# In[14]:


correlation


# In[15]:


# dataset['Company'].value_counts() removed for performance


# In[16]:


# Vectorized company consolidation for performance
_rare_companies = {'Samsung','Razer','Mediacom','Microsoft','Xiaomi','Vero','Chuwi','Google','Fujitsu','LG','Huawei'}
dataset['Company'] = np.where(dataset['Company'].isin(_rare_companies), 'Other', dataset['Company'])


# In[17]:


# dataset['Company'].value_counts() removed for performance


# In[18]:


# len(dataset['Product'].value_counts()) removed for performance


# In[19]:


# dataset['TypeName'].value_counts() removed for performance


# In[20]:


# dataset['ScreenResolution'].value_counts() removed for performance


# In[21]:


# Vectorized flags from ScreenResolution
_sr = dataset['ScreenResolution'].astype(str)
dataset['Touchscreen'] = _sr.str.contains('Touchscreen', na=False).astype('uint8')
dataset['IPS'] = _sr.str.contains('IPS', na=False).astype('uint8')


# In[22]:


# dataset['Cpu'].value_counts() removed for performance


# In[23]:


# Vectorized CPU name extraction
_cpu_name = dataset['Cpu'].astype(str).str.split().str[:3].str.join(' ')
dataset['Cpu_name'] = _cpu_name


# In[24]:


# dataset['Cpu_name'].value_counts() removed for performance


# In[25]:


# Vectorized processor normalization
_intel_set = { 'Intel Core i7', 'Intel Core i5', 'Intel Core i3' }
_is_intel = dataset['Cpu_name'].isin(_intel_set)
_is_amd = dataset['Cpu_name'].str.startswith('AMD')
dataset['Cpu_name'] = np.where(_is_intel, dataset['Cpu_name'], np.where(_is_amd, 'AMD', 'Other'))


# In[26]:


# dataset['Cpu_name'].value_counts() removed for performance


# In[27]:


# Vectorized GPU vendor extraction
dataset['Gpu_name'] = dataset['Gpu'].astype(str).str.split().str[0]


# In[30]:


# dataset['Gpu_name'].value_counts() removed for performance


# In[29]:


dataset = dataset[dataset['Gpu_name'] != 'ARM']


# In[32]:


# dataset.head(2) removed for performance


# In[35]:


dataset['OpSys'].value_counts()


# In[34]:


# Vectorized OS consolidation
_ops = dataset['OpSys'].astype(str)
dataset['OpSys'] = np.select([
    _ops.isin(['Windows 10','Windows 7','Windows 10 S']),
    _ops.isin(['macOS','Mac OS X']),
    _ops.eq('Linux')
], ['Windows','Mac','Linux'], default='Other')


# In[37]:


dataset=dataset.drop(columns=['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu'])


# In[38]:


# dataset.head() removed for performance


# In[39]:


dataset = pd.get_dummies(dataset)


# In[40]:


# dataset.head() removed for performance


# In[41]:


x = dataset.drop('Price_euros',axis=1)
y = dataset['Price_euros']


# In[50]:


# Removed inline pip install to speed up script; ensure dependencies are installed externally


# In[51]:


from sklearn.model_selection import train_test_split
# Reproducible split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[53]:


x_train.shape,x_test.shape


# In[54]:


def model_acc(model):
    model.fit(x_train,y_train)
    acc = model.score(x_test, y_test)
    print(str(model)+'-->'+str(acc))


# In[57]:


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
# Enable parallelism and determinism for faster, reproducible training
rf = RandomForestRegressor(n_jobs=-1, random_state=42)
model_acc(rf)


# In[58]:


from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10,50,100],'criterion':['squared_error','absolute_error','poisson']}

# Parallelize grid search and fix CV seed for determinism
grid_obj = GridSearchCV(estimator=rf, param_grid=parameters, n_jobs=-1, cv=5)

grid_fit = grid_obj.fit(x_train,y_train)

best_model = grid_fit.best_estimator_
best_model


# In[59]:


best_model.score(x_test,y_test)


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




