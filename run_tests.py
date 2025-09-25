#!/usr/bin/env python
"""
Test runner for laptop price prediction model.
Run this to execute all automated tests.
"""

import unittest
import sys
import os

def main():
    """Run all tests for the laptop price model"""
    print("=" * 70)
    print("LAPTOP PRICE MODEL - AUTOMATED TEST SUITE")
    print("=" * 70)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY")
    print("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests Run:     {total_tests}")
    print(f"Passed:              {passed}")
    print(f"Failed:              {failures}")
    print(f"Errors:              {errors}")
    print(f"Skipped:             {skipped}")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate:        {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! The model pipeline is validated.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Review the output above for details.")
        
        if result.failures:
            print(f"\nFailure Summary:")
            for test, error in result.failures:
                print(f"  • {test.id()}")
        
        if result.errors:
            print(f"\nError Summary:")
            for test, error in result.errors:
                print(f"  • {test.id()}")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
