#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for Domain Models (scripts/domain_models.py)
======================================================
Validates initialization, immutability, default values,
and dictionary conversion for all domain DTO classes.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.domain_models import (
    RegimeParams,
    GEXParams,
    PEADParams,
    BuyWindow,
    ForecastSeriesPoint,
    PredictiveForecastResult,
)


class TestDomainModels(unittest.TestCase):
    """Test suite for domain model DTOs."""

    def test_regime_params_defaults(self):
        p = RegimeParams()
        self.assertIsNone(p.state)
        self.assertEqual(p.risk_multiplier, 1.0)
        self.assertEqual(p.expected_run_length_days, 63.0)

    def test_gex_params_immutability(self):
        g = GEXParams(net_gex_millions=15.5, regime_state=1, call_wall=520.0)
        self.assertEqual(g.net_gex_millions, 15.5)
        self.assertEqual(g.call_wall, 520.0)
        with self.assertRaises(Exception):
            g.net_gex_millions = 20.0  # Immutable frozen dataclass

    def test_buy_window_to_dict(self):
        w = BuyWindow(
            start_date="2026-09-10",
            end_date="2026-09-25",
            is_active=True,
            status="ACTIVE",
            description="Optimal Window",
            modeled_window_dates=["2026-09-10", "2026-09-25"],
        )
        d = w.to_dict()
        self.assertEqual(d["start_date"], "2026-09-10")
        self.assertEqual(d["status"], "ACTIVE")
        self.assertTrue(d["is_active"])

    def test_forecast_series_point_to_dict(self):
        pt = ForecastSeriesPoint(date="2026-09-10", bear_p10=490.5, median_p50=510.0, bull_p90=535.2)
        d = pt.to_dict()
        self.assertEqual(d["date"], "2026-09-10")
        self.assertEqual(d["median_p50"], 510.0)


if __name__ == "__main__":
    unittest.main()
