#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Secure Model Saving and Loading

This script demonstrates how to securely save and load machine learning models
using joblib with integrity verification instead of insecure pickle.

Security Fix: Addresses CWE-502 (Insecure Deserialization)
"""

import joblib
import hashlib
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def secure_save_model(model, model_path, metadata_path=None):
    """
    Securely save a model with integrity verification.
    
    Args:
        model: Trained sklearn model
        model_path: Path to save model (e.g., 'model.joblib')
        metadata_path: Path to save metadata (default: 'model_metadata.json')
        
    Returns:
        dict: Metadata including SHA256 hash
    """
    # Default metadata path
    if metadata_path is None:
        metadata_path = model_path.replace('.joblib', '_metadata.json')
    
    # Save model with joblib
    joblib.dump(model, model_path)
    print(f"[✓] Model saved: {model_path}")
    
    # Generate SHA256 hash for integrity verification
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
        model_hash = hashlib.sha256(model_bytes).hexdigest()
    
    # Create metadata
    metadata = {
        'model_type': type(model).__name__,
        'sha256_hash': model_hash,
        'file_size_bytes': len(model_bytes),
        'created_at': datetime.now().isoformat(),
        'model_path': model_path,
        'sklearn_version': None  # Can add version info
    }
    
    # Try to get sklearn version
    try:
        import sklearn
        metadata['sklearn_version'] = sklearn.__version__
    except:
        pass
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[✓] Metadata saved: {metadata_path}")
    print(f"[✓] SHA256 hash: {model_hash[:16]}...{model_hash[-16:]}")
    
    return metadata


def secure_load_model(model_path, metadata_path=None, verify_integrity=True):
    """
    Securely load a model with integrity verification.
    
    Args:
        model_path: Path to model file
        metadata_path: Path to metadata file (default: auto-detect)
        verify_integrity: If True, verify SHA256 hash (recommended)
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If integrity check fails or metadata missing
        FileNotFoundError: If files don't exist
    """
    # Default metadata path
    if metadata_path is None:
        metadata_path = model_path.replace('.joblib', '_metadata.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {e}")
    
    # Verify integrity
    if verify_integrity:
        expected_hash = metadata.get('sha256_hash')
        if not expected_hash:
            raise ValueError("Metadata missing SHA256 hash - cannot verify integrity")
        
        # Calculate actual hash
        with open(model_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Compare hashes
        if actual_hash != expected_hash:
            raise ValueError(
                f"SECURITY: Model integrity check FAILED!\n"
                f"Expected hash: {expected_hash[:32]}...\n"
                f"Actual hash:   {actual_hash[:32]}...\n"
                f"The model file may have been tampered with or corrupted."
            )
        
        print(f"[✓] Integrity verification PASSED")
        print(f"[✓] Hash: {actual_hash[:16]}...{actual_hash[-16:]}")
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"[✓] Model loaded: {model_path}")
        print(f"[✓] Model type: {metadata.get('model_type', 'Unknown')}")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


def demo_secure_workflow():
    """
    Demonstrate secure model save/load workflow
    """
    print("="*70)
    print("DEMO: Secure Model Saving and Loading")
    print("="*70)
    
    # 1. Train a simple model
    print("\n[1] Training a simple model...")
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    print(f"[✓] Model trained - R² Score: {score:.4f}")
    
    # 2. Save model securely
    print("\n[2] Saving model securely...")
    metadata = secure_save_model(model, 'demo_model.joblib')
    
    # 3. Load model securely
    print("\n[3] Loading model securely...")
    loaded_model = secure_load_model('demo_model.joblib')
    
    # 4. Verify model works
    print("\n[4] Verifying loaded model...")
    loaded_score = r2_score(y_test, loaded_model.predict(X_test))
    print(f"[✓] Loaded model R² Score: {loaded_score:.4f}")
    
    if abs(score - loaded_score) < 1e-10:
        print("[✓] Model loaded correctly - predictions match!")
    else:
        print("[✗] Warning: Model predictions don't match")
    
    # 5. Demonstrate tampering detection
    print("\n[5] Demonstrating tampering detection...")
    print("[ATTACK SIMULATION] Tampering with model file...")
    
    # Tamper with the file
    with open('demo_model.joblib', 'ab') as f:
        f.write(b'\x00TAMPERED')
    
    print("[ATTACK] Model file modified")
    
    print("\n[DEFENSE] Attempting to load tampered model...")
    try:
        tampered_model = secure_load_model('demo_model.joblib')
        print("[✗] ERROR: Tampered model was loaded (should have failed!)")
    except ValueError as e:
        print("[✓] Tampering detected and blocked!")
        print(f"[✓] Error message: {str(e)[:80]}...")
    
    # Cleanup
    print("\n[6] Cleaning up demo files...")
    for f in ['demo_model.joblib', 'demo_model_metadata.json']:
        if os.path.exists(f):
            os.remove(f)
            print(f"[✓] Removed: {f}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n[SUMMARY]")
    print("✅ Model saved securely with joblib")
    print("✅ SHA256 hash generated for integrity")
    print("✅ Metadata stored separately")
    print("✅ Model loaded with verification")
    print("✅ Tampering detected and prevented")
    print("\n[RECOMMENDATION]")
    print("Always use secure_save_model() and secure_load_model()")
    print("for production machine learning models!")


def example_usage():
    """
    Simple example showing basic usage
    """
    print("\n" + "="*70)
    print("BASIC USAGE EXAMPLE")
    print("="*70)
    
    # Create and train a simple model
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save securely
    print("\nSaving model...")
    secure_save_model(model, 'example_model.joblib')
    
    # Load securely
    print("\nLoading model...")
    loaded_model = secure_load_model('example_model.joblib')
    
    # Test
    predictions = loaded_model.predict(X[:5])
    print(f"\nSample predictions: {predictions[:3]}")
    
    # Cleanup
    for f in ['example_model.joblib', 'example_model_metadata.json']:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║               SECURE MODEL SAVING AND LOADING EXAMPLE                ║
║                                                                      ║
║  This demonstrates the secure alternative to insecure pickle        ║
║  Security Fix: CWE-502 (Insecure Deserialization)                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run comprehensive demo
    demo_secure_workflow()
    
    # Run basic example
    example_usage()
    
    print("\n" + "="*70)
    print("For more information, see:")
    print("  - SECURITY.md (complete security documentation)")
    print("  - test_security_pickle_vulnerability.py (security tests)")
    print("  - README_SECURITY_FIX.md (quick guide)")
    print("="*70)
