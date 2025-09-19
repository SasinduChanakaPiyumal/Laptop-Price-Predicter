#!/usr/bin/env python
"""
Baseline model training script - modified from original to generate artifacts.
This script replicates the original monolithic behavior but with proper output paths.
"""

import pandas as pd
import numpy as np
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import joblib

# Create artifacts directory
artifacts_dir = Path("artifacts/baseline")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Load and process data (using original script logic)
dataset = pd.read_csv("laptop_price.csv", encoding='latin-1')

# Data preprocessing (exact replication of original script)
dataset['Ram'] = dataset['Ram'].str.replace('GB', '').astype('int32')
dataset['Weight'] = dataset['Weight'].str.replace('kg', '').astype('float64')

non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)

numeric_dataset = dataset.drop(columns=non_numeric_columns)
correlation = numeric_dataset.corr()['Price_euros']

def add_company(inpt):
    if inpt == 'Samsung' or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt

dataset['Company'] = dataset['Company'].apply(add_company)

dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

dataset['Cpu_name'] = dataset['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'

dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)

dataset['Gpu_name'] = dataset['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
dataset = dataset[dataset['Gpu_name'] != 'ARM']

def set_os(inpt):
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'

dataset['OpSys'] = dataset['OpSys'].apply(set_os)

dataset = dataset.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
dataset = pd.get_dummies(dataset)

# Prepare features and target
x = dataset.drop('Price_euros', axis=1)
y = dataset['Price_euros']

# Train-test split (replicating original)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(f"Training set shape: {x_train.shape}, Test set shape: {x_test.shape}")

# Model training - replicating original GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()
parameters = {'n_estimators': [10, 50, 100], 'criterion': ['squared_error', 'absolute_error', 'poisson']}
grid_obj = GridSearchCV(estimator=rf, param_grid=parameters)
grid_fit = grid_obj.fit(x_train, y_train)
best_model = grid_fit.best_estimator_

# Compute R² score
r2_score = best_model.score(x_test, y_test)
print(f"Test R² score: {r2_score}")

# Save model using joblib
model_path = artifacts_dir / "model.joblib"
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Save metrics
metrics = {"r2": round(r2_score, 4)}
metrics_path = artifacts_dir / "metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {metrics_path}")

# Generate manifest
# Compute SHA-256 of CSV file
csv_path = Path("laptop_price.csv")
with open(csv_path, 'rb') as f:
    csv_content = f.read()
    dataset_hash = hashlib.sha256(csv_content).hexdigest()

# Get git commit (short SHA)
try:
    git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
except:
    git_commit = "unknown"  # fallback if not in git repo

# Current UTC timestamp
timestamp = datetime.utcnow().isoformat() + 'Z'

manifest = {
    "dataset_sha256": dataset_hash,
    "created_at": timestamp,
    "script_commit": git_commit
}

manifest_path = artifacts_dir / "manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved to: {manifest_path}")

print("\nBaseline artifacts generated successfully!")
print(f"- Model: {model_path}")
print(f"- Metrics: {metrics_path}")
print(f"- Manifest: {manifest_path}")
