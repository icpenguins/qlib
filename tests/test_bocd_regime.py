#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Unit Tests: Bayesian Online Changepoint Detection (BOCD) & Macro Regime
================================================================================
Verifies:
1. Exact Student-t predictive probability calculations with pure NumPy.
2. Synthetic step-change detection accuracy with Adams & MacKay (2007) BOCD.
3. Multi-horizon realized volatility surface computation (5d, 21d, 63d, vol_ratio).
4. Credit spread ratio and momentum calculations.
5. 4-state market regime classification and posterior probabilities.
6. Integration into stock_analysis_engine.py.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure qlib/contrib and scripts directories are accessible
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CONTRIB_DIR = REPO_ROOT / "qlib" / "contrib"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for p in [str(REPO_ROOT), str(CONTRIB_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from qlib.contrib.regime.bocd import (
    BayesianOnlineChangepointDetector,
    StudentTConjugatePrior,
    ConstantHazard,
    _student_t_pdf,
)
from qlib.contrib.regime.macro_vol_features import MacroVolFeatureExtractor
from qlib.contrib.regime.regime_classifier import MarketRegimeClassifier
from stock_analysis_engine import detect_market_regime


class TestBOCDRegime(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_student_t_pdf_accuracy(self):
        """Verify that zero-dependency Student-t PDF produces mathematically correct density."""
        # For standard Student-t with df=1 (Cauchy distribution):
        # f(0) = Gamma(1) / (sqrt(pi) * Gamma(0.5)) = 1 / pi ~= 0.318309886
        df = np.array([1.0], dtype=np.float64)
        loc = np.array([0.0], dtype=np.float64)
        scale = np.array([1.0], dtype=np.float64)
        val = _student_t_pdf(0.0, df=df, loc=loc, scale=scale)

        expected = 1.0 / np.pi
        self.assertAlmostEqual(float(val[0]), expected, places=5)

        # For df=2: f(0) = Gamma(1.5) / (sqrt(2*pi) * Gamma(1)) = (0.5 * sqrt(pi)) / sqrt(2*pi) = 1 / (2*sqrt(2)) ~= 0.353553
        df2 = np.array([2.0], dtype=np.float64)
        val2 = _student_t_pdf(0.0, df=df2, loc=loc, scale=scale)
        expected2 = 1.0 / (2.0 * np.sqrt(2.0))
        self.assertAlmostEqual(float(val2[0]), expected2, places=5)

    def test_constant_hazard(self):
        """Test constant hazard rate initialization and evaluation."""
        hazard = ConstantHazard(expected_run_length=50.0)
        self.assertAlmostEqual(hazard(10), 0.02)
        arr = hazard(np.arange(5))
        self.assertEqual(len(arr), 5)
        self.assertTrue(np.allclose(arr, 0.02))

        with self.assertRaises(ValueError):
            ConstantHazard(expected_run_length=-5.0)

    def test_bocd_step_change_detection(self):
        """Verify that BOCD detects a clear mean shift in synthetic data."""
        # Segment 1: N(0, 0.2), 60 points
        seg1 = np.random.normal(loc=0.0, scale=0.2, size=60)
        # Segment 2: N(4.0, 0.2), 60 points (sharp structural break)
        seg2 = np.random.normal(loc=4.0, scale=0.2, size=60)
        series = np.concatenate([seg1, seg2])

        detector = BayesianOnlineChangepointDetector(expected_run_length=60.0)
        df_bocd = detector.batch_process(series)

        # The changepoint occurs at index 60
        # Check that changepoint probability at or immediately around index 60 spikes significantly
        cp_window = df_bocd["changepoint_prob"].iloc[59:63]
        max_cp = cp_window.max()
        self.assertGreater(max_cp, 0.70, f"Expected changepoint probability > 0.70 at mean shift, got {max_cp}")

        # Check run length resets
        rl_after_cp = df_bocd["expected_run_length"].iloc[61]
        self.assertLess(rl_after_cp, 5.0, "Expected run-length to reset near 0 after changepoint")

    def test_run_length_posterior_normalization(self):
        """Verify that the run-length posterior integrates to 1.0 at every step."""
        detector = BayesianOnlineChangepointDetector(expected_run_length=30.0)
        data = np.random.normal(0, 1, size=20)

        for x in data:
            cp_prob, exp_rl, post = detector.update(x)
            self.assertAlmostEqual(float(np.sum(post)), 1.0, places=5)
            self.assertGreaterEqual(cp_prob, 0.0)
            self.assertLessEqual(cp_prob, 1.0)

    def test_volatility_surface_extraction(self):
        """Verify multi-horizon realized volatility surface features."""
        extractor = MacroVolFeatureExtractor()
        # Generate 100 days of synthetic price data
        dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=100)))

        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
        })

        surface_df = extractor.compute_volatility_surface(df)
        self.assertIn("vol_5d", surface_df.columns)
        self.assertIn("vol_21d", surface_df.columns)
        self.assertIn("vol_63d", surface_df.columns)
        self.assertIn("vol_ratio", surface_df.columns)
        self.assertIn("vov_21d", surface_df.columns)
        self.assertIn("vol_parkinson_21d", surface_df.columns)

        # Volatilities must be positive numbers
        self.assertTrue((surface_df["vol_5d"] > 0).all())
        self.assertTrue((surface_df["vol_21d"] > 0).all())
        self.assertTrue((surface_df["vol_ratio"] > 0).all())

    def test_credit_spread_features(self):
        """Verify macro credit spread calculation and fallback."""
        extractor = MacroVolFeatureExtractor()
        dates = pd.date_range("2024-01-01", periods=50, freq="B").strftime("%Y-%m-%d")

        # Case 1: Fallback when ETF data is None
        fallback_df = extractor.compute_credit_spread_features(dates=dates)
        self.assertEqual(len(fallback_df), 50)
        self.assertIn("credit_ratio", fallback_df.columns)

        # Case 2: Realistic ETF Data
        hyg_df = pd.DataFrame({"date": dates, "close": 75.0 + np.random.normal(0, 0.5, 50)})
        iei_df = pd.DataFrame({"date": dates, "close": 115.0 + np.random.normal(0, 0.3, 50)})

        credit_df = extractor.compute_credit_spread_features(hyg_df=hyg_df, iei_df=iei_df, dates=dates)
        self.assertIn("credit_ratio", credit_df.columns)
        self.assertIn("credit_mom_21d", credit_df.columns)
        self.assertIn("credit_zscore", credit_df.columns)
        self.assertTrue((credit_df["credit_ratio"] > 0).all())

    def test_market_regime_classifier(self):
        """Verify 4-state regime classifier and summary extraction."""
        classifier = MarketRegimeClassifier(expected_run_length=63.0)

        # Create 120 days of synthetic trending and volatile data
        dates = pd.date_range("2023-01-01", periods=120, freq="B").strftime("%Y-%m-%d")
        prices = 150.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, size=120)))

        df = pd.DataFrame({"date": dates, "close": prices})
        analyzed = classifier.analyze(df)

        self.assertIn("regime_state", analyzed.columns)
        self.assertIn("regime_name", analyzed.columns)
        self.assertIn("changepoint_prob", analyzed.columns)
        self.assertIn("prob_state_0", analyzed.columns)
        self.assertIn("prob_state_1", analyzed.columns)
        self.assertIn("prob_state_2", analyzed.columns)
        self.assertIn("prob_state_3", analyzed.columns)

        # Check discrete state bounds
        self.assertTrue(analyzed["regime_state"].isin([0, 1, 2, 3]).all())

        # Check summary dictionary
        summary = classifier.get_current_regime_summary(analyzed)
        self.assertIn("state", summary)
        self.assertIn("name", summary)
        self.assertIn("changepoint_prob_pct", summary)
        self.assertIn("vol_ratio", summary)
        self.assertIn("action", summary)
        self.assertIn("probabilities", summary)
        self.assertAlmostEqual(
            sum(summary["probabilities"].values()), 1.0, places=2
        )

    def test_stock_analysis_engine_regime_integration(self):
        """Verify that detect_market_regime integrates smoothly with stock_analysis_engine."""
        dates = pd.date_range("2023-01-01", periods=80, freq="B").strftime("%Y-%m-%d")
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, size=80)))
        df = pd.DataFrame({
            "date": dates,
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": [1000000] * 80,
        })

        summary, df_regime = detect_market_regime(df, data_dir=None, symbol="TEST")
        self.assertIsNotNone(summary)
        self.assertIn("changepoint_prob", df_regime.columns)
        self.assertIn("regime_state", df_regime.columns)
        self.assertIn("state", summary)
        self.assertIn(summary["state"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

