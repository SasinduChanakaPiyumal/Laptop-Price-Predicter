#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from laptop_price_model import add_company, set_processor, set_os, model_acc, preprocess_data


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


# Use the refactored add_company function from the module
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


dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[22]:


dataset['Cpu'].value_counts()


# In[23]:


dataset['Cpu_name']= dataset['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[24]:


dataset['Cpu_name'].value_counts()


# In[25]:


# Use the refactored set_processor function from the module
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


# Use the refactored set_os function from the module
dataset['OpSys']= dataset['OpSys'].apply(set_os)


# In[37]:


dataset=dataset.drop(columns=['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu'])


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


pip install scikit-learn


# In[51]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)


# In[53]:


x_train.shape,x_test.shape


# In[54]:


# Use the refactored model_acc function from the module
def print_model_acc(model):
    """Wrapper to maintain the original print behavior"""
    acc = model_acc(model, x_train, y_train, x_test, y_test)
    print(str(model)+'-->'+str(acc))
    return acc


# In[57]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print_model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
print_model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
print_model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print_model_acc(rf)


# In[58]:


from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10,50,100],'criterion':['squared_error','absolute_error','poisson']}

grid_obj = GridSearchCV(estimator = rf ,param_grid = parameters)

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




