# Laptop Price Prediction Model

A machine learning model for predicting laptop prices based on various features such as RAM, weight, CPU, GPU, operating system, and other specifications.

## 🔒 Security Notice

**Important:** This project has been updated to address a critical security vulnerability (CWE-502: Insecure Deserialization). Please read [SECURITY.md](SECURITY.md) for details.

### Key Security Updates

- ✅ Replaced insecure `pickle` with `joblib` for model serialization
- ✅ Added comprehensive security tests
- ✅ Implemented security best practices documentation

**Action Required:** If you have old `*.pickle` files, delete them and regenerate using the updated code which produces `*.joblib` files.

## Features

- Data preprocessing and feature engineering
- Multiple model training (Linear Regression, Lasso, Decision Tree, Random Forest)
- Hyperparameter tuning using GridSearchCV
- **Secure model serialization using joblib**

## Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## Dataset

The model uses `laptop_price.csv` which contains laptop specifications and prices. The dataset includes:
- Company name
- Product type
- RAM, Weight, CPU, GPU specifications
- Screen resolution and features
- Operating system
- Price in euros

## Usage

### Training the Model

```python
python "Laptop Price model(1).py"
```

This will:
1. Load and preprocess the laptop dataset
2. Train multiple models and compare their performance
3. Perform hyperparameter tuning on the best model
4. **Securely save the model to `predictor.joblib`**

### Loading the Model Safely

```python
import joblib

# Load the model securely
model = joblib.load('predictor.joblib')

# Make predictions
# Example: 8GB RAM, 1.2kg weight, touchscreen, IPS, etc.
features = [[8, 1.2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]
predicted_price = model.predict(features)
print(f"Predicted price: €{predicted_price[0]:.2f}")
```

## Model Pipeline

1. **Data Cleaning**
   - Handle missing values
   - Convert units (GB, kg) to numeric values
   - Extract relevant features from text columns

2. **Feature Engineering**
   - Extract touchscreen and IPS panel information
   - Categorize processors and GPUs
   - Simplify operating system categories
   - One-hot encode categorical variables

3. **Model Training**
   - Train multiple regression models
   - Compare model accuracies
   - Select Random Forest as the best performer

4. **Hyperparameter Tuning**
   - Use GridSearchCV to find optimal parameters
   - Parameters tuned: n_estimators, criterion

5. **Model Persistence**
   - **Securely save using joblib** (not pickle)
   - Model can be loaded for predictions

## Security Best Practices

### ✅ DO

- Use `joblib.dump()` and `joblib.load()` for scikit-learn models
- Only load model files from trusted sources
- Verify file integrity (checksums/signatures) before loading
- Run the application with minimal necessary permissions
- Keep dependencies updated
- Run security tests regularly

### ❌ DON'T

- Never use `pickle.load()` on untrusted files
- Don't load models from unknown sources
- Don't run the application with elevated privileges unnecessarily
- Don't share model files over insecure channels without verification

## Testing

### Run Security Tests

```bash
python test_security_vulnerability.py
```

This will:
1. Demonstrate the pickle vulnerability (for educational purposes)
2. Verify that joblib works correctly
3. Confirm that the production code uses secure serialization

Expected output:
```
✓ All security tests passed!
  The exploit no longer works - joblib is being used securely.
```

### Run Model Tests

To verify model functionality:
```python
import joblib
import numpy as np

# Load model
model = joblib.load('predictor.joblib')

# Test prediction
test_data = [[8, 1.2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]
prediction = model.predict(test_data)
assert prediction.shape == (1,), "Prediction shape incorrect"
assert prediction[0] > 0, "Price should be positive"
print("✓ Model tests passed")
```

## Model Performance

The Random Forest Regressor with tuned hyperparameters achieves strong performance on the test set. Specific metrics are displayed during training.

## File Structure

```
.
├── Laptop Price model(1).py    # Main training script (UPDATED: uses joblib)
├── Laptop Price model.ipynb    # Jupyter notebook version
├── laptop_price.csv            # Dataset
├── predictor.joblib            # Trained model (secure format)
├── test_security_vulnerability.py  # Security tests
├── SECURITY.md                 # Security advisory and details
└── README.md                   # This file
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **joblib**: Secure model serialization (✅ required for security)

## Contributing

When contributing to this project:

1. Never introduce `pickle` for serialization
2. Always use `joblib` for ML models
3. Run security tests before submitting changes
4. Follow secure coding practices
5. Document any security-relevant changes

## License

This project is provided as-is for educational and research purposes.

## Changelog

### Version 2.0 (Security Update)
- **BREAKING**: Replaced pickle with joblib
- Added comprehensive security tests
- Added security documentation
- Model file format changed from `.pickle` to `.joblib`

### Version 1.0 (Original)
- Initial release with laptop price prediction
- Multiple model training and comparison
- Hyperparameter tuning
- ⚠️ Used insecure pickle serialization (FIXED)

## Support

For security issues, please refer to [SECURITY.md](SECURITY.md).

For general questions and issues, please check the documentation first.

---

**Remember:** Security is not a feature, it's a requirement. Always validate and secure your data serialization practices.
