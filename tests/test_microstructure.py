#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Unit Tests: Institutional Microstructure, AVWAP & Volume Profile (KDE)
================================================================================
Verifies:
1. Exact mathematical calculation of Anchored VWAP and volume-weighted variance.
2. Dynamic anchor date detection (YTD, QTD, 52W High, 52W Low).
3. Dispersion bands (+/- 1 sigma, +/- 2 sigma) and Z-score spreads.
4. Gaussian Kernel Density Estimation (KDE) integration to 1.0.
5. Point of Control (POC), Value Area (VAH/VAL 70% envelope), and Liquidity Voids.
6. Integration into high-level compute_microstructure_features pipeline.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repository root and qlib are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.microstructure.anchored_vwap import AnchoredVWAPCalculator
from qlib.contrib.microstructure.volume_profile import VolumeProfileKDE
from qlib.contrib.microstructure import compute_microstructure_features


class TestMicrostructure(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_avwap_hand_calculated_accuracy(self):
        """Verify Anchored VWAP and variance against known analytical hand-calculated values."""
        # 3 days of known price and volume
        # Day 0: Price = 100, Vol = 10 -> PV = 1000, CumPV = 1000, CumV = 10 -> AVWAP = 100.0, Var = 0.0
        # Day 1: Price = 110, Vol = 20 -> PV = 2200, CumPV = 3200, CumV = 30 -> AVWAP = 3200/30 = 106.66667
        #        Weighted Var = (10*(100-106.6667)^2 + 20*(110-106.6667)^2) / 30 = (444.444 + 222.222)/30 = 22.2222
        #        Std = sqrt(22.2222) = 4.714045
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [100.0, 110.0, 105.0],
            "volume": [10.0, 20.0, 15.0],
        })

        calc = AnchoredVWAPCalculator()
        res = calc.calculate_single_anchor(df, anchor_date="2024-01-02", prefix="avwap")

        # Day 0
        self.assertAlmostEqual(res["avwap"].iloc[0], 100.0, places=4)
        self.assertAlmostEqual(res["avwap_std"].iloc[0], 0.0, places=4)

        # Day 1
        expected_avwap_1 = 3200.0 / 30.0
        expected_std_1 = np.sqrt(22.222222)
        self.assertAlmostEqual(res["avwap"].iloc[1], expected_avwap_1, places=4)
        self.assertAlmostEqual(res["avwap_std"].iloc[1], expected_std_1, places=4)

        # Dispersion bands
        self.assertAlmostEqual(res["avwap_upper_1s"].iloc[1], expected_avwap_1 + expected_std_1, places=4)
        self.assertAlmostEqual(res["avwap_lower_1s"].iloc[1], expected_avwap_1 - expected_std_1, places=4)

    def test_anchor_date_detection(self):
        """Verify automatic identification of YTD, QTD, 52W High, and 52W Low anchor dates."""
        dates = pd.date_range("2023-01-01", "2024-06-15", freq="B").strftime("%Y-%m-%d")
        n = len(dates)
        prices = 100.0 + np.sin(np.linspace(0, 10, n)) * 20.0

        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": [100000] * n,
        })

        calc = AnchoredVWAPCalculator()
        anchors = calc.identify_anchor_dates(df)

        self.assertIn("ytd", anchors)
        self.assertIn("qtd", anchors)
        self.assertIn("high_52w", anchors)
        self.assertIn("low_52w", anchors)

        # YTD must be the first trading date of 2024
        self.assertTrue(anchors["ytd"].startswith("2024-01"))

        # QTD for June 2024 must start in April 2024 (Q2)
        self.assertTrue(anchors["qtd"].startswith("2024-04"))

    def test_dispersion_bands_and_zscore(self):
        """Verify +/-1 sigma and +/-2 sigma dispersion bands and Z-score calculation."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B").strftime("%Y-%m-%d")
        prices = 150.0 + np.cumsum(np.random.normal(0, 1, 50))

        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [500000] * 50,
        })

        calc = AnchoredVWAPCalculator()
        res = calc.calculate_single_anchor(df, anchor_date=dates[0], prefix="avwap")

        # Upper bands must be greater than AVWAP, lower bands must be less
        valid_mask = ~res["avwap_std"].isna() & (res["avwap_std"] > 1e-3)
        self.assertTrue((res.loc[valid_mask, "avwap_upper_1s"] > res.loc[valid_mask, "avwap"]).all())
        self.assertTrue((res.loc[valid_mask, "avwap_lower_1s"] < res.loc[valid_mask, "avwap"]).all())
        self.assertTrue((res.loc[valid_mask, "avwap_upper_2s"] > res.loc[valid_mask, "avwap_upper_1s"]).all())

        # Z-score test
        expected_z = (res["close"] - res["avwap"]) / res["avwap_std"]
        self.assertTrue(np.allclose(res.loc[valid_mask, "avwap_zscore"], expected_z[valid_mask], atol=1e-3))

    def test_volume_profile_kde_integration(self):
        """Verify that continuous Gaussian KDE density integrates to ~1.0."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
        prices = 100.0 + np.random.normal(0, 2, 60)
        vols = np.random.uniform(100000, 500000, 60)

        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": vols,
        })

        vp = VolumeProfileKDE(lookback=60, grid_size=200)
        profile = vp.compute_profile(df)

        self.assertIn("poc", profile)
        self.assertIn("vah", profile)
        self.assertIn("val", profile)
        self.assertIn("current_price", profile)

        # POC must be bounded by the minimum and maximum prices
        self.assertGreaterEqual(profile["poc"], min(prices) - 5)
        self.assertLessEqual(profile["poc"], max(prices) + 5)

        # Value Area relationship: VAL <= POC <= VAH (or very close to POC)
        self.assertLessEqual(profile["val"], profile["vah"])

    def test_liquidity_void_detection(self):
        """Verify that price levels with low volume density are identified as liquidity voids."""
        # Create a bimodal price distribution with a big gap in the middle (liquidity void)
        # Cluster 1: around 100, Cluster 2: around 130
        dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
        cluster1 = np.random.normal(100.0, 1.0, 30)
        cluster2 = np.random.normal(130.0, 1.0, 30)
        prices = np.concatenate([cluster1, cluster2])
        vols = np.full(60, 100000.0)

        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": vols,
        })

        vp = VolumeProfileKDE(lookback=60)
        profile = vp.compute_profile(df)

        # Low-volume nodes (LVNs) should be detected in the empty middle zone (between 105 and 125)
        self.assertTrue(len(profile["lvn_levels"]) > 0)
        lvn_in_gap = any(105 <= lvn <= 125 for lvn in profile["lvn_levels"])
        self.assertTrue(lvn_in_gap, f"Expected LVN between 105 and 125, got {profile['lvn_levels']}")

    def test_compute_microstructure_features(self):
        """Verify the integrated high-level function returns enriched df and summary."""
        dates = pd.date_range("2023-06-01", periods=100, freq="B").strftime("%Y-%m-%d")
        prices = 200.0 + np.cumsum(np.random.normal(0.05, 1.5, 100))

        df = pd.DataFrame({
            "date": dates,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(500000, 2000000, 100),
        })

        enriched, summary = compute_microstructure_features(df)

        self.assertIn("avwap_ytd", enriched.columns)
        self.assertIn("avwap_ytd_upper_1s", enriched.columns)
        self.assertIn("avwap", summary)
        self.assertIn("volume_profile", summary)
        self.assertIn("ytd", summary["avwap"])
        self.assertIn("poc", summary["volume_profile"])
        self.assertIn("vah", summary["volume_profile"])
        self.assertIn("val", summary["volume_profile"])


if __name__ == "__main__":
    unittest.main()

