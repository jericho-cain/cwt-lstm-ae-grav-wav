#!/usr/bin/env python3
"""
Test runner for Gravitational Wave Hunter v2.0

This script runs all tests and provides comprehensive reporting.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def run_all_tests():
    """Run all tests and return results."""
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print detailed summary
    print(f"\n{'='*80}")
    print(f"TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = (success_count / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n{'='*80}")
        print(f"FAILURES ({len(result.failures)}):")
        print(f"{'='*80}")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n{'='*80}")
        print(f"ERRORS ({len(result.errors)}):")
        print(f"{'='*80}")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback.split('Exception:')[-1].strip()}")
    
    print(f"\n{'='*80}")
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*80}")
    
    return result.wasSuccessful()


def run_specific_test(test_name):
    """Run a specific test class or method."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Gravitational Wave Hunter v2.0")
    parser.add_argument("--test", help="Run specific test (e.g., TestConfigValidator.test_valid_downloader_config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.test:
        print(f"Running specific test: {args.test}")
        success = run_specific_test(args.test)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
