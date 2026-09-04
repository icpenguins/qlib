#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for Stock Analysis JSON Data Contract Engine & CLI (stock_analysis_data.py)
=====================================================================================
Validates canonical schema serialization, type sanitization, path resolution,
disk export/import round-trips, and command-line interface execution.
"""

import os
import sys
import json
import tempfile
import unittest
import datetime
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add scripts directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stock_analysis_data import (
    resolve_json_path,
    _sanitize_for_json,
    prepare_analysis_json_payload,
    export_analysis_json,
    load_analysis_json,
    generate_stock_analysis_data,
)


class TestStockAnalysisData(unittest.TestCase):
    """
    Test suite for stock_analysis_data.py data contract engine and CLI.
    """

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2025-10-31", freq="B")
        n = len(dates)
        base_price = 150.0
        returns = np.random.normal(0.0005, 0.015, n)
        prices = base_price * np.exp(np.cumsum(returns))

        self.df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": prices * (1 + np.random.normal(0, 0.002, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n),
        })

        self.mock_analysis = {
            "symbol": "AAPL",
            "request_date": "2025-10-31",
            "latest_data_date": "2025-10-31",
            "is_up_to_date": True,
            "forecast_days": 63,
            "historical_data": self.df,
            "performance": {
                "latest_date": "2025-10-31",
                "latest_close": round(float(prices[-1]), 2),
                "latest_price": round(float(prices[-1]), 2),
                "periods": {
                    "1Y": {
                        "available": True,
                        "start_price": 140.0,
                        "end_price": round(float(prices[-1]), 2),
                        "total_return_pct": 28.5,
                        "cagr_pct": 28.5,
                        "max_drawdown_pct": -10.5,
                        "sharpe_ratio": 1.62,
                        "annual_volatility_pct": 17.5,
                        "win_rate_pct": 55.0,
                    }
                },
            },
            "best_buys": [
                {
                    "rank": 1,
                    "date": "2024-04-19",
                    "entry_price": 165.0,
                    "peak_price": 220.0,
                    "peak_date": "2024-10-15",
                    "holding_days": 180,
                    "max_gain_pct": 33.3,
                    "return_to_present_pct": 25.0,
                    "trigger_type": "AVWAP Value Area Bounce",
                    "drawdown_before_entry_pct": -12.0,
                }
            ],
            "predictive": {
                "recommendation": "STRONG BUY",
                "action_summary": "Accumulate dips at YTD AVWAP support.",
                "optimal_entry_range": [215.0, 222.0],
                "optimal_buy_window": {"start_date": "2025-11-01", "end_date": "2025-11-15", "description": "Earnings Run-up"},
                "target_price_3m": 245.0,
                "expected_return_pct": 11.5,
                "stop_loss": 208.0,
                "key_support": 212.0,
                "key_resistance": 248.0,
                "risk_reward_ratio": 3.4,
            },
            "projections": {
                "6M": {
                    "label": "6-Month",
                    "projected_return_pct": 9.2,
                    "base_target_price": 240.0,
                    "bear_price": 205.0,
                    "bull_price": 265.0,
                    "probability_score": 75.0,
                    "confidence": "High",
                    "projected_cagr_pct": 19.1,
                    "effective_drift_pct": 8.5,
                    "effective_vol_pct": 16.5,
                }
            },
            "regime": {
                "state": 1,
                "name": "Bull Trend",
                "action": "Accumulate dips",
                "badge_class": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
                "description": "Macro credit expansion and low volatility.",
                "changepoint_prob_pct": 8.5,
                "expected_run_length_days": 60,
                "vol_ratio": 0.92,
                "vol_21d_pct": 14.8,
                "vol_5d_pct": 13.6,
                "credit_mom_pct": 1.5,
                "risk_multiplier": 1.0,
                "probabilities": {"bull": 0.88, "risk_off": 0.12},
            },
            "microstructure": {
                "avwap": {
                    "ytd": {"value": 218.0, "zscore": 0.6, "lower_1s": 210.0, "upper_1s": 226.0, "date": "2025-01-02", "action": "Hold above AVWAP"},
                    "high_52w": {"value": 235.0, "date": "2025-07-20", "spread_pct": -4.0},
                    "low_52w": {"value": 170.0, "date": "2024-11-10", "spread_pct": 32.0},
                },
                "volume_profile": {
                    "poc": 220.0,
                    "val": 212.0,
                    "vah": 228.0,
                    "dist_to_poc_pct": 1.8,
                    "in_value_area": True,
                    "void_status": "Balanced Liquidity",
                    "in_liquidity_void": False,
                }
            },
            "derivatives": {
                "net_gex_millions": 45.2,
                "regime": "+GEX (Dealer Long Gamma / Volatility Dampening)",
                "gamma_flip_price": 214.0,
                "dist_to_flip_pct": 5.2,
                "call_wall": 240.0,
                "put_wall": 210.0,
                "max_pain": 225.0,
                "atm_iv_pct": 21.5,
                "vrp_pct": 2.8,
                "rr25_skew": 1.2,
                "strike_profile": [
                    {"strike": 210.0, "call_gex_m": 3.5, "put_gex_m": -12.0, "net_gex_m": -8.5, "open_interest": 15000},
                    {"strike": 220.0, "call_gex_m": 15.0, "put_gex_m": -4.0, "net_gex_m": 11.0, "open_interest": 32000},
                ],
            },
            "events": {
                "catalyst_status": {
                    "status_code": "SAFE",
                    "status_description": "Clear trading window.",
                    "next_earnings_date": "2025-11-20",
                    "days_to_earnings": 20,
                    "next_macro_event": "CPI",
                    "next_macro_date": "2025-11-12",
                    "days_to_macro": 12,
                },
                "pead": {
                    "drift_regime": "Bullish Post-Earnings Drift",
                    "sue_score": 1.40,
                    "announcement_gap_pct": 4.1,
                    "post_earnings_drift_pct": 5.0,
                    "recent_announcement_date": "2025-08-01",
                },
                "degrossing": {
                    "position_haircut": 1.0,
                    "risk_advice": "Normal position sizing.",
                    "binary_gap_sd": 0.042,
                },
                "recent_earnings_history": [
                    {
                        "date": "2025-08-01",
                        "actual_eps": 1.50,
                        "estimated_eps": 1.35,
                        "surprise_pct": 11.1,
                        "sue_score": 1.40,
                        "announcement_gap_pct": 4.1,
                        "drift_30d_pct": 5.0,
                    }
                ],
            }
        }

    def test_resolve_json_path_default_and_custom(self):
        """Test resolve_json_path across default, custom report_dir, and explicit outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Custom report_dir
            p1 = resolve_json_path("MSFT", report_dir=tmpdir, report_date="2026-09-04")
            self.assertEqual(p1.name, "MSFT_analysis_report_2026-09-04.json")
            self.assertEqual(p1.suffix, ".json")
            self.assertEqual(p1.parent.resolve(), Path(tmpdir).resolve())

            # 2. Explicit output with .json extension
            custom_json = Path(tmpdir) / "custom_output.json"
            p2 = resolve_json_path("MSFT", output=custom_json, report_date="2026-09-04")
            self.assertEqual(p2.resolve(), custom_json.resolve())

            # 3. Explicit output with .html extension (should convert to .json)
            custom_html = Path(tmpdir) / "custom_output.html"
            p3 = resolve_json_path("MSFT", output=custom_html, report_date="2026-09-04")
            self.assertEqual(p3.resolve(), custom_json.resolve())

            # 4. Explicit output as directory
            custom_sub = Path(tmpdir) / "subfolder"
            p4 = resolve_json_path("MSFT", output=custom_sub, report_date="2026-09-04")
            self.assertEqual(p4.name, "MSFT_analysis_report_2026-09-04.json")
            self.assertEqual(p4.parent.resolve(), custom_sub.resolve())

    def test_sanitize_for_json(self):
        """Test recursive sanitization of NumPy, Pandas, NaN, Inf, and date types."""
        test_dict = {
            "none": None,
            "bool": True,
            "str": "alpha",
            "np_int": np.int32(100),
            "np_float": np.float64(45.67),
            "nan": np.nan,
            "inf": np.inf,
            "neg_inf": -np.inf,
            "date": datetime.date(2026, 9, 4),
            "dt": datetime.datetime(2026, 9, 4, 15, 30),
            "ts": pd.Timestamp("2026-09-04 15:30:00"),
            "series": pd.Series([10.0, np.nan, 30.0]),
            "df": pd.DataFrame({"col1": [1, 2], "col2": [np.nan, 4.5]}),
            "nested": {
                "arr": np.array([1, 2, 3]),
                "sub_nan": np.nan,
            }
        }
        sanitized = _sanitize_for_json(test_dict)
        self.assertIsNone(sanitized["none"])
        self.assertTrue(sanitized["bool"])
        self.assertEqual(sanitized["str"], "alpha")
        self.assertEqual(sanitized["np_int"], 100)
        self.assertIsInstance(sanitized["np_int"], int)
        self.assertAlmostEqual(sanitized["np_float"], 45.67)
        self.assertIsInstance(sanitized["np_float"], float)
        self.assertIsNone(sanitized["nan"])
        self.assertIsNone(sanitized["inf"])
        self.assertIsNone(sanitized["neg_inf"])
        self.assertEqual(sanitized["date"], "2026-09-04")
        self.assertIn("2026-09-04", sanitized["dt"])
        self.assertIn("2026-09-04", sanitized["ts"])
        self.assertEqual(sanitized["series"], [10.0, None, 30.0])
        self.assertEqual(sanitized["df"], [{"col1": 1, "col2": None}, {"col1": 2, "col2": 4.5}])
        self.assertEqual(sanitized["nested"]["arr"], [1, 2, 3])
        self.assertIsNone(sanitized["nested"]["sub_nan"])

    def test_prepare_analysis_json_payload(self):
        """Test canonical schema preparation, versioning, and SMA computation."""
        payload = prepare_analysis_json_payload(self.mock_analysis)

        # Metadata verification
        self.assertIn("metadata", payload)
        meta = payload["metadata"]
        self.assertEqual(meta["contract_version"], "1.2.0")
        self.assertEqual(meta["symbol"], "AAPL")
        self.assertEqual(meta["request_date"], "2025-10-31")
        self.assertEqual(meta["latest_data_date"], "2025-10-31")
        self.assertTrue(meta["is_up_to_date"])
        self.assertEqual(meta["forecast_days"], 63)
        self.assertIn("generated_at", meta)

        # Historical data verification
        self.assertIn("historical_data", payload)
        self.assertIsInstance(payload["historical_data"], list)
        self.assertTrue(len(payload["historical_data"]) > 0)
        last_bar = payload["historical_data"][-1]
        self.assertIn("date", last_bar)
        self.assertIn("close", last_bar)
        self.assertIn("volume", last_bar)
        self.assertIn("sma50", last_bar)
        self.assertIn("sma200", last_bar)

        # Sub-model verification
        self.assertIn("performance", payload)
        self.assertIn("best_buys", payload)
        self.assertIn("predictive", payload)
        self.assertIn("projections", payload)
        self.assertIn("regime", payload)
        self.assertIn("microstructure", payload)
        self.assertIn("derivatives", payload)
        self.assertIn("events", payload)
        self.assertIn("earnings_gamma_squeeze", payload)
        self.assertIn("backtesting_protocol", payload)
        self.assertIn("evaluation_matrix", payload)

        # Nested institutional keys inside earnings_gamma_squeeze
        egs = payload["earnings_gamma_squeeze"]
        self.assertIn("calibrate_post_earnings_volatility_surface", egs)
        self.assertIn("factor_orthogonalization", egs)
        self.assertIn("earnings_event_clock", egs)

        # Institutional backtesting protocol checks
        bp = payload["backtesting_protocol"]
        self.assertIn("purged_walk_forward_cv", bp)
        self.assertIn("almgren_chriss_market_impact", bp)
        self.assertIn("borrow_fee_engine", bp)
        self.assertIn("deflated_sharpe_ratio", bp)
        self.assertIn("verifiable_replication_event_panel", bp)

    def test_export_and_load_json_roundtrip(self):
        """Test disk serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "AAPL_analysis_report_2025-10-31.json"

            # Export
            exported = export_analysis_json(self.mock_analysis, json_file)
            self.assertTrue(exported.exists())
            self.assertEqual(exported, json_file.resolve())

            # Load
            loaded = load_analysis_json(json_file)
            self.assertEqual(loaded["metadata"]["symbol"], "AAPL")
            self.assertEqual(loaded["metadata"]["contract_version"], "1.2.0")
            self.assertEqual(loaded["predictive"]["recommendation"], "STRONG BUY")
            self.assertEqual(loaded["derivatives"]["regime"], "+GEX (Dealer Long Gamma / Volatility Dampening)")

            # Verify non-existent file error
            missing_file = Path(tmpdir) / "does_not_exist.json"
            with self.assertRaises(FileNotFoundError):
                load_analysis_json(missing_file)

    def test_generate_stock_analysis_data(self):
        """Test generate_stock_analysis_data with mocked analytical engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stock_analysis_data.run_stock_analysis", return_value=self.mock_analysis) as mock_run:
                out_path = generate_stock_analysis_data(
                    symbol="AAPL",
                    data_dir=tmpdir,
                    report_dir=tmpdir,
                    forecast_days=63,
                    auto_download=False,
                    request_date="2025-10-31",
                )
                self.assertTrue(out_path.exists())
                self.assertEqual(out_path.name, "AAPL_analysis_report_2025-10-31.json")
                mock_run.assert_called_once()

    def test_cli_execution_with_custom_output(self):
        """Test standalone CLI execution via subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_json = Path(tmpdir) / "TEST_cli_out.json"

            # Use mock script execution by invoking python directly
            python_bin = sys.executable
            cli_script = SCRIPTS_DIR / "stock_analysis_data.py"

            # Test --help
            res_help = subprocess.run([python_bin, str(cli_script), "--help"], capture_output=True, text=True)
            self.assertEqual(res_help.returncode, 0)
            self.assertIn("Institutional Stock Analysis JSON Data Contract Generator", res_help.stdout)

            # Test missing --symbol errors
            res_missing = subprocess.run([python_bin, str(cli_script)], capture_output=True, text=True)
            self.assertNotEqual(res_missing.returncode, 0)

    def test_cli_main_execution(self):
        """Test main() CLI execution end-to-end with local synthetic dataset."""
        from stock_analysis_data import main
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            self.df.to_csv(data_dir / "TEST.csv", index=False)

            report_dir = Path(tmpdir) / "reports"
            test_argv = [
                "stock_analysis_data.py",
                "--symbol", "TEST",
                "--data_dir", str(data_dir),
                "--report_dir", str(report_dir),
                "--request_date", "2025-10-31",
                "--no-auto_download",
                "--quiet",
            ]
            with patch("sys.argv", test_argv):
                ret = main()
                self.assertEqual(ret, 0)

            expected_json = report_dir / "TEST_analysis_report_2025-10-31.json"
            self.assertTrue(expected_json.exists())
            data = load_analysis_json(expected_json)
            self.assertEqual(data["metadata"]["symbol"], "TEST")
            self.assertEqual(data["metadata"]["request_date"], "2025-10-31")
            self.assertEqual(data["metadata"]["contract_version"], "1.2.0")
            self.assertIn("backtesting_protocol", data)
            self.assertIn("evaluation_matrix", data)
            self.assertIn("calibrate_post_earnings_volatility_surface", data["earnings_gamma_squeeze"])


if __name__ == "__main__":
    unittest.main()

