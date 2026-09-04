#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Event Risk, Catalyst Awareness & PEAD Engine
===========================================================================
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.events import (
    EventCalendarEngine,
    EventProximity,
    PEADEngine,
    RiskDegrossingEngine,
    EventsDataLoader,
    SyntheticEventScheduleGenerator,
    compute_event_risk_features,
)


class TestEventRiskPEAD(unittest.TestCase):
    """Test suite for Event Risk, Catalyst Calendar, and PEAD Engine."""

    def setUp(self):
        """Create synthetic price history."""
        np.random.seed(42)
        dates = pd.bdate_range(start="2024-01-02", periods=252).strftime("%Y-%m-%d")
        base = 150.0
        prices = [base]
        for _ in range(1, len(dates)):
            prices.append(prices[-1] * (1.0 + np.random.normal(0.0008, 0.015)))

        self.df = pd.DataFrame({
            "date": dates,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1_500_000 for _ in prices],
            "symbol": "TEST",
        })

    def test_event_calendar_proximity_and_business_days(self):
        """Test business day counting and threat proximity classification."""
        cal = EventCalendarEngine()

        # Business days calculation
        self.assertEqual(cal.count_business_days("2026-09-01", "2026-09-04"), 3)
        self.assertEqual(cal.count_business_days("2026-09-04", "2026-09-01"), -3)
        self.assertEqual(cal.count_business_days("2026-09-04", "2026-09-04"), 0)

        # Proximity classification
        self.assertEqual(cal.classify_proximity(1), EventProximity.CRITICAL_EVENT)
        self.assertEqual(cal.classify_proximity(3), EventProximity.IMMINENT_DEGROSS)
        self.assertEqual(cal.classify_proximity(7), EventProximity.APPROACHING)
        self.assertEqual(cal.classify_proximity(15), EventProximity.SAFE)

    def test_pead_sue_and_drift_calculation(self):
        """Test Standardized Unexpected Earnings (SUE) and PEAD drift dynamics."""
        engine = PEADEngine(drift_half_life_days=21.0)

        # SUE calculation
        sue = engine.compute_sue(actual_eps=1.50, estimate_eps=1.30, surprise_history=[0.10, 0.15, 0.05, 0.20])
        self.assertGreater(sue, 0.0)

        # Mock earnings history
        earnings_hist = [
            {"date": "2024-02-15", "actual": 1.20, "estimate": 1.10, "surprisePercent": 9.1},
            {"date": "2024-05-16", "actual": 1.35, "estimate": 1.25, "surprisePercent": 8.0},
            {"date": "2024-08-15", "actual": 1.50, "estimate": 1.38, "surprisePercent": 8.7},
        ]

        pead = engine.evaluate_recent_pead(
            df=self.df,
            earnings_history=earnings_hist,
            current_date="2024-09-01",
        )

        self.assertTrue(pead["has_pead"])
        self.assertEqual(pead["latest_report_date"], "2024-08-15")
        self.assertGreater(pead["sue_score"], 0.0)
        self.assertIn("Bullish", pead["drift_regime"])

    def test_risk_degrossing_and_buy_window_adjustment(self):
        """Test risk de-grossing multiplier and pre-event buy window adjustment."""
        # Multiplier
        self.assertEqual(RiskDegrossingEngine.calculate_degross_multiplier(1, "earnings"), 0.0)
        self.assertEqual(RiskDegrossingEngine.calculate_degross_multiplier(3, "earnings"), 0.50)
        self.assertEqual(RiskDegrossingEngine.calculate_degross_multiplier(5, "earnings"), 0.75)
        self.assertEqual(RiskDegrossingEngine.calculate_degross_multiplier(15, "earnings"), 1.0)

        # Window adjustment: event falls inside window -> shifts start past event
        adj_start, adj_end, delayed = RiskDegrossingEngine.adjust_buy_window(
            window_start_date="2026-09-10",
            window_end_date="2026-09-30",
            event_date="2026-09-12",
            current_date="2026-09-03",
            min_buffer_days=2,
        )
        self.assertTrue(delayed)
        self.assertGreater(adj_start, "2026-09-12")

    def test_synthetic_event_schedule_generator(self):
        """Test deterministic synthetic quarterly schedule generator."""
        sched = SyntheticEventScheduleGenerator.generate_schedule(
            symbol="NVDA",
            current_date="2026-09-03",
            seed=42,
        )
        self.assertEqual(sched["symbol"], "NVDA")
        self.assertIn("next_earnings_date", sched)
        self.assertGreater(sched["next_earnings_date"], "2026-09-03")
        self.assertGreater(len(sched["earnings_history"]), 10)

    def test_compute_event_risk_features_and_momentum_events(self):
        """Test full compute_event_risk_features workflow and key momentum event extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features = compute_event_risk_features(
                df=self.df,
                symbol="TEST_PEAD",
                data_dir=tmpdir,
                current_date="2024-12-01",
                bocd_changepoints=[{"date": "2024-04-10", "state": 0, "name": "Bull Trend"}],
            )

            self.assertEqual(features["symbol"], "TEST_PEAD")
            self.assertIn("catalyst", features)
            self.assertIn("pead", features)
            self.assertIn("degross_multiplier", features)
            self.assertIn("momentum_events", features)

            # Check momentum events structure
            m_events = features["momentum_events"]
            self.assertIsInstance(m_events, list)
            if m_events:
                ev0 = m_events[0]
                self.assertIn("date", ev0)
                self.assertIn("badge", ev0)
                self.assertIn("type", ev0)
                self.assertIn("price", ev0)


if __name__ == "__main__":
    unittest.main()

