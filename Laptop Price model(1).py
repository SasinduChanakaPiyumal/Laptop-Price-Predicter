#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load and explore dataset
dataset = pd.read_csv("laptop_price.csv", encoding="latin-1")

dataset.head()
dataset.shape
dataset.describe()
dataset.info()
dataset.isnull().sum()

# Data preprocessing
dataset["Ram"] = dataset["Ram"].str.replace("GB", "").astype("int32")
dataset.head()

dataset["Weight"] = dataset["Weight"].str.replace("kg", "").astype("float64")
dataset.head(2)

non_numeric_columns = dataset.select_dtypes(exclude=["number"]).columns
print(non_numeric_columns)

numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()["Price_euros"]
correlation

dataset["Company"].value_counts()


def add_company(inpt):
    other_companies = [
        "Samsung",
        "Razer",
        "Mediacom",
        "Microsoft",
        "Xiaomi",
        "Vero",
        "Chuwi",
        "Google",
        "Fujitsu",
        "LG",
        "Huawei",
    ]
    if inpt in other_companies:
        return "Other"
    else:
        return inpt


dataset["Company"] = dataset["Company"].apply(add_company)
dataset["Company"].value_counts()

len(dataset["Product"].value_counts())
dataset["TypeName"].value_counts()
dataset["ScreenResolution"].value_counts()

# Feature engineering
dataset["Touchscreen"] = dataset["ScreenResolution"].apply(
    lambda x: 1 if "Touchscreen" in x else 0
)
dataset["IPS"] = dataset["ScreenResolution"].apply(
    lambda x: 1 if "IPS" in x else 0
)

dataset["Cpu"].value_counts()
dataset["Cpu_name"] = dataset["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
dataset["Cpu_name"].value_counts()


def set_processor(name):
    intel_processors = ["Intel Core i7", "Intel Core i5", "Intel Core i3"]
    if name in intel_processors:
        return name
    else:
        if name.split()[0] == "AMD":
            return "AMD"
        else:
            return "Other"


dataset["Cpu_name"] = dataset["Cpu_name"].apply(set_processor)
dataset["Cpu_name"].value_counts()

dataset["Gpu_name"] = dataset["Gpu"].apply(lambda x: " ".join(x.split()[0:1]))
dataset["Gpu_name"].value_counts()

# Filter out ARM GPUs
dataset = dataset[dataset["Gpu_name"] != "ARM"]
dataset.head(2)

dataset["OpSys"].value_counts()


def set_os(inpt):
    windows_versions = ["Windows 10", "Windows 7", "Windows 10 S"]
    mac_versions = ["macOS", "Mac OS X"]
    
    if inpt in windows_versions:
        return "Windows"
    elif inpt in mac_versions:
        return "Mac"
    elif inpt == "Linux":
        return inpt
    else:
        return "Other"


dataset["OpSys"] = dataset["OpSys"].apply(set_os)

# Drop unnecessary columns
dataset = dataset.drop(
    columns=["laptop_ID", "Inches", "Product", "ScreenResolution", "Cpu", "Gpu"]
)
dataset.head()

# Create dummy variables
dataset = pd.get_dummies(dataset)
dataset.head()

# Prepare features and target
x = dataset.drop("Price_euros", axis=1)
y = dataset["Price_euros"]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train.shape, x_test.shape


def model_acc(model):
    """Calculate and print model accuracy."""
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(str(model) + "-->" + str(acc))


# Train different models
lr = LinearRegression()
model_acc(lr)

lasso = Lasso()
model_acc(lasso)

dt = DecisionTreeRegressor()
model_acc(dt)

rf = RandomForestRegressor()
model_acc(rf)

# Hyperparameter tuning
parameters = {
    "n_estimators": [10, 50, 100],
    "criterion": ["squared_error", "absolute_error", "poisson"],
}

grid_obj = GridSearchCV(estimator=rf, param_grid=parameters)
grid_fit = grid_obj.fit(x_train, y_train)
best_model = grid_fit.best_estimator_
best_model

best_model.score(x_test, y_test)
x_train.columns

# Save the model
with open("predictor.pickle", "wb") as file:
    pickle.dump(best_model, file)

# Make predictions
sample_features = [
    [8, 1.4, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
]
best_model.predict(sample_features)

sample_features_2 = [
    [8, 0.9, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
]
best_model.predict(sample_features_2)

sample_features_3 = [
    [8, 1.2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
]
best_model.predict(sample_features_3)

sample_features_4 = [
    [8, 0.9, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
]
best_model.predict(sample_features_4)
