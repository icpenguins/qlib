#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for Shared Technical Indicators Module (scripts/indicators.py)
========================================================================
Validates mathematical correctness, boundary behavior, and edge cases
for compute_rsi, compute_bollinger_bands, and compute_rolling_drawdown.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_rolling_drawdown,
)


class TestIndicators(unittest.TestCase):
    """Test suite for technical indicator functions."""

    def setUp(self):
        np.random.seed(42)
        # 100 days of trending upward prices with some noise
        returns = np.random.normal(0.001, 0.015, 100)
        self.prices = pd.Series(100.0 * np.cumprod(1.0 + returns), name="close")

    def test_compute_rsi_values_in_bounds(self):
        """Verify that RSI values always fall strictly between 0 and 100."""
        rsi = compute_rsi(self.prices, period=14)
        valid = rsi.dropna()
        self.assertGreater(len(valid), 0)
        self.assertTrue((valid >= 0.0).all())
        self.assertTrue((valid <= 100.0).all())

    def test_compute_rsi_monotonically_rising(self):
        """Monotonically rising prices should produce RSI near 100."""
        rising = pd.Series([float(i) for i in range(1, 40)])
        rsi = compute_rsi(rising, period=14)
        last_rsi = float(rsi.iloc[-1])
        self.assertGreater(last_rsi, 95.0)

    def test_compute_rsi_monotonically_falling(self):
        """Monotonically falling prices should produce RSI near 0."""
        falling = pd.Series([float(100 - i) for i in range(40)])
        rsi = compute_rsi(falling, period=14)
        last_rsi = float(rsi.iloc[-1])
        self.assertLess(last_rsi, 5.0)

    def test_compute_rsi_empty_and_single_element(self):
        """Edge cases: empty series and single row should return without error."""
        empty_s = pd.Series(dtype=float)
        rsi_empty = compute_rsi(empty_s, period=14)
        self.assertTrue(rsi_empty.empty)

        single_s = pd.Series([100.0])
        rsi_single = compute_rsi(single_s, period=14)
        self.assertEqual(len(rsi_single), 1)
        self.assertTrue(np.isnan(rsi_single.iloc[0]))

    def test_compute_bollinger_bands_geometry(self):
        """Upper band must be strictly greater than middle, and lower strictly less."""
        mid, upper, lower, pct_b = compute_bollinger_bands(self.prices, window=20, num_std=2.0)
        valid_idx = mid.dropna().index
        for idx in valid_idx:
            m = mid.loc[idx]
            u = upper.loc[idx]
            l = lower.loc[idx]
            self.assertGreaterEqual(u, m)
            self.assertLessEqual(l, m)

    def test_compute_rolling_drawdown_bounds(self):
        """Drawdown must always be <= 0.0."""
        dd = compute_rolling_drawdown(self.prices, window=50)
        valid = dd.dropna()
        self.assertTrue((valid <= 0.00001).all())

    def test_compute_rolling_drawdown_new_high_is_zero(self):
        """At an all-time high, drawdown must equal 0.0."""
        rising = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        dd = compute_rolling_drawdown(rising, window=10)
        self.assertAlmostEqual(float(dd.iloc[-1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
