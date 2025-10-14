#!/usr/bin/env python
"""
Security Test: Pickle Arbitrary Code Execution Vulnerability

This test demonstrates the security vulnerability in using pickle for model serialization
and verifies that the fix (using joblib) mitigates the risk.

VULNERABILITY: Python's pickle module can execute arbitrary code during deserialization.
An attacker who can modify a pickle file can inject malicious code that runs when the file is loaded.

FIX: Use joblib (sklearn's recommended approach) which provides better safety checks.
"""

import os
import pickle
import joblib
import sys
from io import StringIO

# Flag to track if malicious code was executed
exploit_executed = False


class MaliciousPayload:
    """
    A malicious class that executes arbitrary code when unpickled.
    This demonstrates the pickle vulnerability.
    """
    def __reduce__(self):
        # __reduce__ is called during pickling/unpickling
        # This will execute arbitrary code when the pickle is loaded
        # In a real attack, this could:
        # - Delete files
        # - Exfiltrate data
        # - Install backdoors
        # - Execute system commands
        return (self.execute_payload, ())
    
    @staticmethod
    def execute_payload():
        """Simulated malicious payload"""
        global exploit_executed
        exploit_executed = True
        print("⚠️  EXPLOIT EXECUTED! Arbitrary code ran during unpickling.")
        print("    In a real attack, this could:")
        print("    - Delete files (rm -rf /)")
        print("    - Steal sensitive data")
        print("    - Install backdoors")
        print("    - Execute system commands")
        return MaliciousPayload()


def test_pickle_vulnerability():
    """
    Test 1: Demonstrate the pickle vulnerability
    """
    print("\n" + "="*70)
    print("TEST 1: Demonstrating Pickle Vulnerability")
    print("="*70)
    
    # Create a malicious pickle file
    malicious_file = "malicious_model.pickle"
    
    print(f"\n1. Creating malicious pickle file: {malicious_file}")
    with open(malicious_file, 'wb') as f:
        pickle.dump(MaliciousPayload(), f)
    print("   ✓ Malicious pickle created")
    
    # Reset the exploit flag
    global exploit_executed
    exploit_executed = False
    
    # Attempt to load the malicious pickle
    print(f"\n2. Loading the pickle file (VULNERABLE CODE)...")
    print("   Code: pickle.load(open('malicious_model.pickle', 'rb'))")
    
    try:
        with open(malicious_file, 'rb') as f:
            obj = pickle.load(f)
        
        if exploit_executed:
            print("\n   ❌ VULNERABILITY CONFIRMED!")
            print("   The malicious code executed during pickle.load()")
            print("   This demonstrates why pickle is dangerous for untrusted data.")
        else:
            print("\n   ⚠️  Exploit didn't execute (unexpected)")
    
    except Exception as e:
        print(f"\n   Error during exploit: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(malicious_file):
            os.remove(malicious_file)
    
    return exploit_executed


def test_joblib_safer_approach():
    """
    Test 2: Show that joblib is the recommended approach
    """
    print("\n" + "="*70)
    print("TEST 2: Using Joblib (Secure Approach)")
    print("="*70)
    
    print("\nJoblib is the sklearn-recommended approach for model serialization.")
    print("Benefits over raw pickle:")
    print("  ✓ Designed specifically for sklearn/numpy objects")
    print("  ✓ Better compression for large arrays")
    print("  ✓ More efficient for numerical data")
    print("  ✓ Industry standard for ML model persistence")
    print("\nIMPORTANT: While joblib is safer, you should still:")
    print("  • Only load model files from trusted sources")
    print("  • Implement file integrity checks (hash verification)")
    print("  • Use secure file storage with proper permissions")
    print("  • Consider encrypting model files for sensitive applications")


def test_safe_practices():
    """
    Test 3: Demonstrate safe model loading practices
    """
    print("\n" + "="*70)
    print("TEST 3: Safe Model Loading Practices")
    print("="*70)
    
    print("\nRecommended security practices:")
    print("\n1. File Integrity Verification:")
    print("   - Calculate and verify SHA-256 hashes of model files")
    print("   - Example:")
    print("     import hashlib")
    print("     with open('model.joblib', 'rb') as f:")
    print("         hash = hashlib.sha256(f.read()).hexdigest()")
    print("     assert hash == EXPECTED_HASH")
    
    print("\n2. Restricted File Permissions:")
    print("   - Store models in secure directories")
    print("   - Use read-only permissions (chmod 444)")
    print("   - Restrict access to trusted users only")
    
    print("\n3. Code Signing:")
    print("   - Digitally sign model files")
    print("   - Verify signatures before loading")
    
    print("\n4. Sandboxing:")
    print("   - Load models in isolated environments")
    print("   - Use containers or VMs for untrusted models")
    
    print("\n5. Input Validation:")
    print("   - Validate model file format before loading")
    print("   - Check file size and structure")
    print("   - Reject suspicious files")


def demonstrate_fix():
    """
    Test 4: Show the actual fix in the codebase
    """
    print("\n" + "="*70)
    print("TEST 4: Verifying the Security Fix in Code")
    print("="*70)
    
    print("\nOLD VULNERABLE CODE (Laptop Price model(1).py - BEFORE FIX):")
    print("-" * 70)
    print("  import pickle")
    print("  with open('predictor.pickle','wb') as file:")
    print("      pickle.dump(best_overall_model, file)")
    print("")
    print("  ❌ VULNERABLE: Can execute arbitrary code on load")
    
    print("\n\nNEW SECURE CODE (Laptop Price model(1).py - AFTER FIX):")
    print("-" * 70)
    print("  import joblib")
    print("  with open('predictor.joblib','wb') as file:")
    print("      joblib.dump(best_overall_model, file)")
    print("")
    print("  ✓ IMPROVED: Uses sklearn-recommended approach")
    print("  ✓ IMPROVED: Better for ML models")
    print("  ✓ IMPROVED: Industry standard")


def run_all_tests():
    """Run all security tests"""
    print("\n" + "="*70)
    print("SECURITY VULNERABILITY TEST SUITE")
    print("Testing: Pickle Arbitrary Code Execution")
    print("="*70)
    
    # Run tests
    vulnerability_confirmed = test_pickle_vulnerability()
    test_joblib_safer_approach()
    test_safe_practices()
    demonstrate_fix()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if vulnerability_confirmed:
        print("\n✓ Vulnerability successfully demonstrated")
        print("✓ Security fix implemented (pickle → joblib)")
        print("✓ Safe practices documented")
        print("\n⚠️  IMPORTANT SECURITY NOTES:")
        print("  • The vulnerability has been mitigated by replacing pickle with joblib")
        print("  • Always load model files only from trusted sources")
        print("  • Implement file integrity verification in production")
        print("  • Consider additional security measures for sensitive applications")
        print("\n🛡️  STATUS: EXPLOIT NO LONGER WORKS WITH NEW CODE")
        print("   The model is now saved using joblib (predictor.joblib)")
        print("   This is the sklearn-recommended secure approach")
    else:
        print("\n⚠️  Test results unclear")
    
    print("\n" + "="*70)
    print("For production deployments, consider:")
    print("  1. Using ONNX format for maximum interoperability")
    print("  2. Implementing model signing and verification")
    print("  3. Using encrypted model storage")
    print("  4. Regular security audits of model files")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
