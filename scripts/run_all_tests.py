"""
Unified Test Runner for Qlib & Institutional Trading Production Test Suites.

Usage:
    python scripts/run_all_tests.py             # Run Core Institutional Production Suite (Default, 51 tests)
    python scripts/run_all_tests.py --all       # Run full discovery across all tests/ modules
    python scripts/run_all_tests.py --suite gex # Run a specific suite (engine, pead, gex, microstructure, bocd, download)
    python scripts/run_all_tests.py -v          # Verbose output
"""

import sys
import os
import argparse
import unittest
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_SUITES = {
    "engine": ("Stock Analysis Multi-Model Engine", "tests.test_stock_analysis_engine"),
    "pead": ("Event Risk & Post-Earnings Announcement Drift (PEAD)", "tests.test_events_pead"),
    "gex": ("Dealer Gamma Exposure (GEX) & Derivatives", "tests.test_derivatives_gex"),
    "microstructure": ("Market Microstructure (AVWAP & Volume Profile)", "tests.test_microstructure"),
    "bocd": ("Bayesian Online Changepoint Detection (BOCD)", "tests.test_bocd_regime"),
    "download": ("US Selected Market Data Ingestion & Calendar", "tests.test_download_us_selected_data"),
    "data": ("Stock Analysis JSON Data Contract Pipeline", "tests.test_stock_analysis_data"),
    "visualize": ("Interactive Visualizer & Two-Step Pipeline", "tests.test_visualize_stock_analysis_refactor"),
}


def run_core_suite(verbosity=1, suite_key=None):
    """Runs the core production test suite(s)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    selected = [suite_key] if suite_key else list(CORE_SUITES.keys())
    print("=" * 70)
    print("INSTITUTIONAL PRODUCTION TEST SUITE RUNNER")
    print("=" * 70)

    for key in selected:
        if key not in CORE_SUITES:
            print(f"Unknown suite '{key}'. Available: {', '.join(CORE_SUITES.keys())}")
            return False
        name, mod = CORE_SUITES[key]
        print(f"[*] Loading: {name} ({mod})")
        try:
            tests = loader.loadTestsFromName(mod)
            suite.addTests(tests)
        except Exception as e:
            print(f"    ERROR loading {mod}: {e}")

    print("-" * 70)
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"SUMMARY: Ran {result.testsRun} tests in {elapsed:.2f}s")
    print(f"Passed:   {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors:   {len(result.errors)}")
    print("Status:   " + ("ALL PASSED [OK]" if result.wasSuccessful() else "FAILED"))
    print("=" * 70)
    return result.wasSuccessful()


def run_all_discovery(verbosity=1):
    """Runs full discovery across the tests/ directory."""
    print("=" * 70)
    print("FULL DISCOVERY TEST RUNNER (tests/)")
    print("=" * 70)
    print("Note: Academic benchmark tests requiring optional packages (mlflow, joblib, scipy)")
    print("will report import errors if optional dependencies are not installed.")
    print("-" * 70)

    start_time = time.time()
    loader = unittest.TestLoader()
    tests_dir = os.path.join(PROJECT_ROOT, "tests")
    suite = loader.discover(tests_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"DISCOVERY SUMMARY: Ran {result.testsRun} tests in {elapsed:.2f}s")
    print(f"Passed:   {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors:   {len(result.errors)} (includes missing optional dependencies)")
    print("=" * 70)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Unified Test Runner for Qlib & Institutional Trading Suites")
    parser.add_argument("--all", action="store_true", help="Discover and run all tests in tests/ (including legacy)")
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(CORE_SUITES.keys()),
        help=f"Run a specific core suite: {', '.join(CORE_SUITES.keys())}",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test execution output")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    if args.all:
        success = run_all_discovery(verbosity=verbosity)
    else:
        success = run_core_suite(verbosity=verbosity, suite_key=args.suite)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

