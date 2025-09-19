# Laptop Price Prediction - Baseline Model

This project implements a baseline machine learning model for predicting laptop prices.

## Baseline Artifacts

The baseline model artifacts are stored in `artifacts/baseline/`:

- `model.joblib` - Trained RandomForestRegressor model
- `metrics.json` - Model performance metrics (R² score)
- `manifest.json` - Dataset hash, creation timestamp, and git commit

## Running the Baseline

To execute the baseline model training:

```bash
poetry run python "Laptop Price model(1).py"
```

Or use the standalone script:

```bash
poetry run python create_baseline.py
```

## Model Details

The baseline model uses:
- RandomForestRegressor with GridSearchCV optimization
- Features: Company, RAM, Weight, CPU type, GPU type, OS, Touchscreen, IPS
- Test set size: 25% of data
- Performance metric: R² score

## Dependencies

Install dependencies using Poetry:

```bash
poetry install
```

Required packages:
- pandas
- numpy  
- scikit-learn
- joblib
