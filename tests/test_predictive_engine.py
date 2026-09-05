#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for Decomposed Predictive Engine (scripts/predictive_engine.py)
========================================================================
Validates collaborating services:
- RegimeParameterExtractor
- GEXParameterExtractor
- EventParameterExtractor
- MonteCarloSimulator
- SupportResistanceSynthesizer
- RecommendationEngine
- predict_future_buy_timing orchestration
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.domain_models import GEXParams, RegimeParams, PEADParams
from scripts.predictive_engine import (
    RegimeParameterExtractor,
    GEXParameterExtractor,
    EventParameterExtractor,
    SupportResistanceSynthesizer,
    MonteCarloSimulator,
    RecommendationEngine,
    predict_future_buy_timing,
)


class TestPredictiveEngine(unittest.TestCase):
    """Test suite for decomposed predictive engine services."""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
        returns = np.random.normal(0.0005, 0.015, 100)
        closes = 500.0 * np.cumprod(1.0 + returns)
        self.df = pd.DataFrame({
            "date": dates,
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.uniform(1_000_000, 5_000_000, 100),
        })

    def test_regime_extractor_defaults(self):
        params = RegimeParameterExtractor.extract(None, forecast_days=63)
        self.assertIsNone(params.state)
        self.assertEqual(params.risk_multiplier, 1.0)
        self.assertEqual(params.daily_hazard, 0.0)

    def test_regime_extractor_populated(self):
        mock_regime = {
            "state": 0,
            "name": "Bull Trend",
            "changepoint_prob_pct": 15.0,
            "expected_run_length_days": 50.0,
            "risk_multiplier": 1.2,
        }
        params = RegimeParameterExtractor.extract(mock_regime, forecast_days=63)
        self.assertEqual(params.state, 0)
        self.assertEqual(params.risk_multiplier, 1.2)
        self.assertAlmostEqual(params.daily_hazard, 1.0 / 50.0)

    def test_gex_extractor(self):
        mock_gex = {
            "regime": "+GEX Pinned",
            "net_gex_millions": 25.0,
            "call_wall": 520.0,
            "put_wall": 490.0,
            "gamma_flip_price": 500.0,
            "max_pain": 505.0,
        }
        params = GEXParameterExtractor.extract(mock_gex)
        self.assertEqual(params.regime_state, 1)
        self.assertEqual(params.vol_multiplier, 0.85)
        self.assertEqual(params.call_wall, 520.0)

    def test_support_resistance_synthesizer(self):
        gex_p = GEXParams(call_wall=530.0, put_wall=480.0, max_pain=495.0)
        support, resistance = SupportResistanceSynthesizer.synthesize(
            current_price=500.0,
            df=self.df,
            sma50=495.0,
            bb_upper=520.0,
            bb_lower=485.0,
            microstructure=None,
            gex_params=gex_p,
        )
        self.assertLessEqual(support, 500.0)
        self.assertGreaterEqual(resistance, 500.0)

    def test_monte_carlo_simulator_shapes(self):
        future_dates = [f"2026-09-{i:02d}" for i in range(1, 22)]
        p10, p50, p90 = MonteCarloSimulator.simulate(
            current_price=500.0,
            forecast_days=len(future_dates),
            simulations=100,
            daily_vol=0.015,
            drift=0.0005,
            sma50=498.0,
            regime_params=RegimeParams(),
            gex_params=GEXParams(),
            pead_params=PEADParams(),
            future_dates=future_dates,
            seed=42,
        )
        self.assertEqual(len(p10), len(future_dates))
        self.assertEqual(len(p50), len(future_dates))
        self.assertEqual(len(p90), len(future_dates))
        self.assertTrue((p10 <= p50).all())
        self.assertTrue((p50 <= p90).all())

    def test_predict_future_buy_timing_end_to_end(self):
        res = predict_future_buy_timing(self.df, forecast_days=21, simulations=100)
        self.assertIn("current_price", res)
        self.assertIn("recommendation", res)
        self.assertIn("optimal_buy_window", res)
        self.assertIn("forecast_series", res)
        self.assertEqual(len(res["forecast_series"]), 21)


if __name__ == "__main__":
    unittest.main()
