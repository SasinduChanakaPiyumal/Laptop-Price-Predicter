# Laptop Price Prediction Model

A comprehensive machine learning project for predicting laptop prices based on hardware specifications and features. This project implements advanced feature engineering, multiple ML algorithms, and robust hyperparameter tuning to achieve high prediction accuracy.

## 🎯 Purpose

This project aims to:
- **Predict laptop prices** accurately based on technical specifications (RAM, CPU, GPU, storage, display, etc.)
- **Compare multiple ML algorithms** including Linear Regression, Random Forest, Gradient Boosting, LightGBM, and XGBoost
- **Demonstrate best practices** in feature engineering, model selection, and hyperparameter optimization
- **Provide a production-ready model** with comprehensive evaluation metrics and reproducibility

## 📊 Project Overview

The model processes laptop specifications through multiple stages:
1. **Data preprocessing** - Cleaning and transforming raw specifications
2. **Feature engineering** - Creating derived features (PPI, storage quality, interaction terms)
3. **Model training** - Training and tuning multiple regression models
4. **Model selection** - Comparing models and selecting the best performer
5. **Prediction** - Using the trained model to estimate laptop prices

### Key Features

- ✨ **Advanced feature engineering**: Storage type detection, screen resolution parsing, component interaction features
- 🔧 **Multiple ML algorithms**: Linear models (with regularization), tree-based models, gradient boosting variants
- 🎛️ **Comprehensive hyperparameter tuning**: RandomizedSearchCV with 60 iterations across 7-9 parameters per model
- 📈 **Robust evaluation**: R², MAE, RMSE, and 5-fold cross-validation metrics
- 🔒 **Security-conscious**: Dependency vulnerability scanning with pip-audit and secret scanning with gitleaks
- 📦 **Reproducible**: Fixed random seeds and hashed dependency specifications

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # On Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **Option A: Quick install (unpinned versions)**
   ```bash
   pip install -r requirements.in
   ```
   
   **Option B: Secure install (with hashed dependencies - recommended for production)**
   ```bash
   # Install pip-tools first
   pip install pip-tools
   
   # Generate requirements.txt with hashes
   pip-compile --generate-hashes -o requirements.txt requirements.in
   
   # Install dependencies
   pip-sync requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost; print('All dependencies installed successfully!')"
   ```

### Dataset

The project uses `laptop_price.csv` which should be in the project root directory. The dataset contains:
- Laptop specifications (Company, TypeName, RAM, Weight, etc.)
- Screen details (Inches, ScreenResolution)
- CPU and GPU information
- Memory/Storage configuration
- Operating system
- **Target variable**: Price_euros

## 📖 Usage

### Training the Model

Run the main Python script to train and evaluate models:

```bash
python "Laptop Price model(1).py"
```

The script will:
1. Load and preprocess the dataset
2. Engineer features (storage parsing, PPI calculation, interaction terms)
3. Train multiple baseline models
4. Perform hyperparameter tuning (this takes 10-30 minutes depending on CPU)
5. Compare all models and select the best performer
6. Save the best model to `predictor.pickle`
7. Display feature importance and performance metrics

### Expected Output

The script provides detailed output including:

```
===========================================================
MODEL COMPARISON - BASELINE MODELS
===========================================================

--- LINEAR MODELS (with scaling) ---

Linear Regression:
  R² Score: 0.7845
  MAE: 215.32 euros
  RMSE: 334.67 euros
  CV R² Score: 0.7721 (+/- 0.0234)

[... more models ...]

===========================================================
HYPERPARAMETER TUNING - Random Forest
===========================================================

Training Random Forest with RandomizedSearchCV (60 iterations)...
Best parameters: {'n_estimators': 300, 'max_depth': 25, ...}
Best CV score: 0.8654

[... Gradient Boosting and LightGBM tuning ...]

===========================================================
FINAL MODEL COMPARISON
===========================================================

[Comparison of all tuned models]

************************************************************
WINNER: LightGBM
R² Score: 0.8845
************************************************************

===========================================================
FEATURE IMPORTANCE ANALYSIS
===========================================================

Top 15 Most Important Features:
[Feature importance table]
```

### Making Predictions

Load the trained model and make predictions:

```python
import pickle
import pandas as pd

# Load the trained model
with open('predictor.pickle', 'rb') as file:
    model = pickle.load(file)

# Prepare your data (must match training features)
# Example: Create a DataFrame with laptop specifications
sample_data = pd.DataFrame({
    'Ram': [16],
    'Weight': [1.8],
    'Touchscreen': [1],
    'IPS': [1],
    # ... (include all required features)
})

# Make prediction
predicted_price = model.predict(sample_data)
print(f"Predicted price: €{predicted_price[0]:.2f}")
```

**Note**: The input data must have the exact same features (in the same order) as the training data. See the script output for the complete feature list.

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed Mermaid diagram showing the data flow and system components.

## 🔧 Project Structure

```
.
├── Laptop Price model(1).py  # Main training script
├── laptop_price.csv           # Dataset (Latin-1 encoding)
├── requirements.in            # Core dependencies
├── requirements.txt           # Pinned dependencies (generated)
├── predictor.pickle           # Trained model (generated)
├── README.md                  # This file
├── ARCHITECTURE.md            # Architecture diagram
├── CONTRIBUTING.md            # Contribution guidelines
├── SECURITY.md               # Security policy and scanning procedures
├── IMPROVEMENTS_IMPLEMENTED.md # Detailed improvement documentation
├── ML_IMPROVEMENTS_SUMMARY.md  # ML enhancement summary
└── .gitignore                # Git ignore rules
```

## 📊 Model Performance

The project implements and compares multiple models with the following typical performance:

| Model | R² Score | MAE (€) | RMSE (€) | Training Time |
|-------|----------|---------|----------|---------------|
| Linear Regression | ~0.78 | ~215 | ~335 | Fast |
| Ridge Regression | ~0.80 | ~205 | ~320 | Fast |
| Lasso Regression | ~0.79 | ~210 | ~328 | Fast |
| ElasticNet | ~0.79 | ~208 | ~325 | Fast |
| Decision Tree | ~0.82 | ~185 | ~300 | Fast |
| Random Forest (tuned) | ~0.87 | ~165 | ~260 | Moderate |
| Gradient Boosting (tuned) | ~0.88 | ~155 | ~245 | Slow |
| **LightGBM (tuned)** | **~0.89** | **~145** | **~235** | **Moderate** |
| XGBoost (tuned) | ~0.88 | ~150 | ~240 | Slow |

**Best Model**: LightGBM typically wins with ~89% variance explained (R²) and Mean Absolute Error of ~€145.

## 🔬 Advanced Features

### Feature Engineering Highlights

1. **Storage Feature Extraction**
   - Detects storage types (SSD, HDD, Flash, Hybrid)
   - Extracts total capacity in GB (handles multiple drives)
   - Creates storage quality score (SSD=3, Flash=2.5, Hybrid=2, HDD=1)

2. **Screen Quality Features**
   - Parses resolution (e.g., "1920x1080")
   - Calculates PPI (Pixels Per Inch)
   - Computes total pixels

3. **Interaction Features**
   - RAM × Storage Quality (workstation/gaming indicator)
   - Display PPI × Storage Score (premium laptop indicator)
   - Weight/Size ratio (portability metric)
   - Premium Storage = Capacity × (SSD+1) / 1000

### Model Features

- **Feature Scaling**: StandardScaler for linear models
- **Outlier Detection**: Z-score based detection and reporting (outliers retained)
- **Cross-Validation**: 5-fold CV for all models
- **Multiple Metrics**: R², MAE, RMSE, CV scores
- **Feature Importance**: Top 15 features displayed for interpretability

## 🔐 Security

This project follows secure dependency management practices:

- **Hashed Dependencies**: Use `pip-compile --generate-hashes` to pin exact versions with SHA256 hashes
- **Vulnerability Scanning**: Regular `pip-audit` scans (see [SECURITY.md](SECURITY.md))
- **Secret Scanning**: Gitleaks integration to prevent credential leaks
- **No Hardcoded Secrets**: No API keys or passwords in code

Run security scans:
```bash
# Install tools
pip install pip-audit

# Scan dependencies
pip-audit -r requirements.txt

# For secret scanning, install gitleaks and run:
# gitleaks detect --source . --report-path gitleaks.sarif
```

## 🧪 Testing and Validation

The script includes:
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Test set evaluation**: 25% held-out test set
- **Multiple metrics**: R², MAE, RMSE to assess different aspects of performance
- **Feature importance**: Identifies key predictive features

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up a development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, GPL, etc.]

## 📚 Documentation

- **[IMPROVEMENTS_IMPLEMENTED.md](IMPROVEMENTS_IMPLEMENTED.md)**: Detailed documentation of all ML improvements
- **[ML_IMPROVEMENTS_SUMMARY.md](ML_IMPROVEMENTS_SUMMARY.md)**: Summary of enhancements and expected performance gains
- **[SECURITY.md](SECURITY.md)**: Security policy, scanning procedures, and vulnerability tracking
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and data flow diagrams

## 🙏 Acknowledgments

- Dataset source: [Specify dataset source if applicable]
- Built with: scikit-learn, XGBoost, LightGBM, pandas, numpy

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

---

**Note**: This project is for educational and research purposes. Price predictions are estimates and may not reflect actual market prices.
