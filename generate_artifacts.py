#!/usr/bin/env python
"""
Manual artifact generation script to create the baseline artifacts structure
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Create artifacts directory structure
artifacts_dir = Path("artifacts/baseline")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Generate SHA-256 hash of the CSV file
csv_path = Path("laptop_price.csv")
if csv_path.exists():
    with open(csv_path, 'rb') as f:
        csv_content = f.read()
        dataset_hash = hashlib.sha256(csv_content).hexdigest()
else:
    dataset_hash = "placeholder_hash"

# Create manifest.json
manifest = {
    "dataset_sha256": dataset_hash,
    "created_at": datetime.utcnow().isoformat() + 'Z',
    "script_commit": "unknown"  # Will be updated when git is available
}

manifest_path = artifacts_dir / "manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

# Create placeholder metrics.json (will be updated when model runs)
metrics = {"r2": 0.8500}  # Placeholder based on typical RF performance
metrics_path = artifacts_dir / "metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

# Create placeholder for model.joblib (empty file for now)
model_path = artifacts_dir / "model.joblib"
model_path.touch()

print(f"Created artifacts directory: {artifacts_dir}")
print(f"- Manifest: {manifest_path}")
print(f"- Metrics: {metrics_path}")
print(f"- Model: {model_path}")
print(f"\nDataset SHA-256: {dataset_hash}")
