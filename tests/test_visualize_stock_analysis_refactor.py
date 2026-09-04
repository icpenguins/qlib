#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Two-Step Reporting Pipeline & JSON Decoupling in visualize_stock_analysis.py
"""

import sys
import json
import tempfile
import unittest
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from visualize_stock_analysis import (
    resolve_report_path,
    resolve_json_path,
    _sanitize_for_json,
    prepare_analysis_json_payload,
    export_analysis_json,
    load_analysis_json,
    generate_html_dashboard,
    build_projection_cards_html,
    build_regime_card_html,
    build_microstructure_card_html,
    build_derivatives_card_html,
    build_events_card_html,
)


class TestVisualizeStockAnalysisRefactor(unittest.TestCase):

    def setUp(self):
        # Create synthetic historical data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2025-10-31", freq="B")
        n = len(dates)
        base_price = 100.0
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
            "symbol": "TEST",
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
                        "start_price": 120.0,
                        "end_price": round(float(prices[-1]), 2),
                        "total_return_pct": 25.5,
                        "cagr_pct": 25.5,
                        "max_drawdown_pct": -12.0,
                        "sharpe_ratio": 1.45,
                        "annual_volatility_pct": 18.2,
                        "win_rate_pct": 54.0,
                    },
                    "3Y": {
                        "available": True,
                        "start_price": 90.0,
                        "end_price": round(float(prices[-1]), 2),
                        "total_return_pct": 65.0,
                        "cagr_pct": 18.2,
                        "max_drawdown_pct": -22.0,
                        "sharpe_ratio": 1.15,
                        "annual_volatility_pct": 19.5,
                        "win_rate_pct": 53.0,
                    },
                    "5Y": {
                        "available": True,
                        "start_price": 80.0,
                        "end_price": round(float(prices[-1]), 2),
                        "total_return_pct": 110.0,
                        "cagr_pct": 16.0,
                        "max_drawdown_pct": -28.0,
                        "sharpe_ratio": 1.05,
                        "annual_volatility_pct": 20.1,
                        "win_rate_pct": 52.5,
                    },
                },
            },
            "best_buys": [
                {
                    "rank": 1,
                    "date": "2023-10-25",
                    "entry_price": 105.0,
                    "peak_price": 150.0,
                    "peak_date": "2024-03-15",
                    "holding_days": 142,
                    "max_gain_pct": 42.8,
                    "return_to_present_pct": 35.0,
                    "trigger_type": "Dip Accumulation",
                    "drawdown_before_entry_pct": -15.2,
                }
            ],
            "predictive": {
                "recommendation": "STRONG BUY",
                "action_summary": "Favorable risk/reward profile conditioned on market microstructure.",
                "optimal_entry_range": [round(float(prices[-1]) * 0.97, 2), round(float(prices[-1]) * 1.01, 2)],
                "optimal_buy_window": {"start_date": "2025-11-03", "end_date": "2025-11-21", "description": "Optimal Entry Window"},
                "target_price_3m": round(float(prices[-1]) * 1.12, 2),
                "expected_return_pct": 12.0,
                "stop_loss": round(float(prices[-1]) * 0.92, 2),
                "key_support": round(float(prices[-1]) * 0.94, 2),
                "key_resistance": round(float(prices[-1]) * 1.15, 2),
                "risk_reward_ratio": 3.2,
            },
            "projections": {
                "6M": {
                    "label": "6-Month",
                    "projected_return_pct": 8.5,
                    "base_target_price": round(float(prices[-1]) * 1.085, 2),
                    "bear_price": round(float(prices[-1]) * 0.95, 2),
                    "bull_price": round(float(prices[-1]) * 1.20, 2),
                    "probability_score": 72.0,
                    "confidence": "High",
                    "projected_cagr_pct": 17.5,
                    "effective_drift_pct": 8.0,
                    "effective_vol_pct": 18.0,
                }
            },
            "regime": {
                "state": 1,
                "name": "Bull Trend",
                "action": "Accumulate dips",
                "badge_class": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
                "description": "Macro credit expansion and low volatility.",
                "changepoint_prob_pct": 12.5,
                "expected_run_length_days": 45,
                "vol_ratio": 0.95,
                "vol_21d_pct": 15.2,
                "vol_5d_pct": 14.5,
                "credit_mom_pct": 1.2,
                "risk_multiplier": 1.0,
                "probabilities": {"bull": 0.85, "risk_off": 0.15},
            },
            "microstructure": {
                "avwap": {
                    "ytd": {"value": 135.0, "zscore": 0.8, "lower_1s": 128.0, "upper_1s": 142.0, "date": "2025-01-02", "action": "Hold above AVWAP"},
                    "high_52w": {"value": 150.0, "date": "2025-07-15", "spread_pct": -5.0},
                    "low_52w": {"value": 110.0, "date": "2024-11-05", "spread_pct": 28.0},
                },
                "volume_profile": {
                    "poc": 138.0,
                    "val": 130.0,
                    "vah": 145.0,
                    "dist_to_poc_pct": 2.5,
                    "in_value_area": True,
                    "void_status": "Balanced Liquidity",
                    "in_liquidity_void": False,
                }
            },
            "derivatives": {
                "net_gex_millions": 25.4,
                "regime": "+GEX (Dealer Long Gamma / Volatility Dampening)",
                "gamma_flip_price": 132.0,
                "dist_to_flip_pct": 7.5,
                "call_wall": 155.0,
                "put_wall": 130.0,
                "max_pain": 140.0,
                "atm_iv_pct": 22.0,
                "vrp_pct": 3.2,
                "rr25_skew": 1.5,
                "strike_profile": [
                    {"strike": 130.0, "call_gex_m": 2.1, "put_gex_m": -8.5, "net_gex_m": -6.4, "open_interest": 12000},
                    {"strike": 140.0, "call_gex_m": 12.0, "put_gex_m": -3.2, "net_gex_m": 8.8, "open_interest": 25000},
                    {"strike": 150.0, "call_gex_m": 18.5, "put_gex_m": -1.1, "net_gex_m": 17.4, "open_interest": 32000},
                    {"strike": 155.0, "call_gex_m": 22.0, "put_gex_m": -0.5, "net_gex_m": 21.5, "open_interest": 40000},
                ],
            },
            "events": {
                "catalyst_status": {
                    "status_code": "SAFE",
                    "status_description": "Clear trading window.",
                    "next_earnings_date": "2025-12-15",
                    "days_to_earnings": 30,
                    "next_macro_event": "FOMC",
                    "next_macro_date": "2025-11-06",
                    "days_to_macro": 4,
                },
                "pead": {
                    "drift_regime": "Bullish Post-Earnings Drift",
                    "sue_score": 1.25,
                    "announcement_gap_pct": 3.5,
                    "post_earnings_drift_pct": 4.2,
                    "recent_announcement_date": "2025-08-15",
                },
                "degrossing": {
                    "position_haircut": 1.0,
                    "risk_advice": "Normal position sizing.",
                    "binary_gap_sd": 0.045,
                },
                "recent_earnings_history": [
                    {
                        "date": "2025-08-15",
                        "actual_eps": 1.45,
                        "estimated_eps": 1.30,
                        "surprise_pct": 11.5,
                        "sue_score": 1.25,
                        "announcement_gap_pct": 3.5,
                        "drift_30d_pct": 4.2,
                    }
                ],
            }
        }

    def test_resolve_json_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = resolve_json_path("NVDA", report_dir=tmpdir, report_date="2026-09-04")
            self.assertEqual(json_p.name, "NVDA_analysis_report_2026-09-04.json")
            self.assertEqual(json_p.suffix, ".json")

            html_p = resolve_report_path("NVDA", report_dir=tmpdir, report_date="2026-09-04")
            self.assertEqual(html_p.name, "NVDA_analysis_report_2026-09-04.html")
            self.assertEqual(json_p.stem, html_p.stem)

    def test_sanitize_for_json(self):
        raw_obj = {
            "nan_val": np.nan,
            "inf_val": np.inf,
            "np_int": np.int64(42),
            "np_float": np.float64(3.1415),
            "ts": pd.Timestamp("2026-09-04 12:00:00"),
            "series": pd.Series([1.0, 2.0, np.nan]),
        }
        clean = _sanitize_for_json(raw_obj)
        self.assertIsNone(clean["nan_val"])
        self.assertIsNone(clean["inf_val"])
        self.assertEqual(clean["np_int"], 42)
        self.assertAlmostEqual(clean["np_float"], 3.1415)
        self.assertTrue(isinstance(clean["ts"], str))
        self.assertEqual(clean["series"], [1.0, 2.0, None])

    def test_prepare_analysis_json_payload(self):
        payload = prepare_analysis_json_payload(self.mock_analysis)
        self.assertIn("metadata", payload)
        self.assertEqual(payload["metadata"]["symbol"], "TEST")
        self.assertEqual(payload["metadata"]["contract_version"], "1.0.0")
        self.assertIn("historical_data", payload)
        self.assertIsInstance(payload["historical_data"], list)
        self.assertTrue(len(payload["historical_data"]) > 0)
        self.assertIn("sma50", payload["historical_data"][-1])
        self.assertIn("sma200", payload["historical_data"][-1])
        self.assertIn("performance", payload)
        self.assertIn("best_buys", payload)
        self.assertIn("predictive", payload)
        self.assertIn("projections", payload)
        self.assertIn("regime", payload)
        self.assertIn("microstructure", payload)
        self.assertIn("derivatives", payload)
        self.assertIn("events", payload)

    def test_two_step_export_and_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.json"
            
            # Step 1: Export JSON
            exported_path = export_analysis_json(self.mock_analysis, json_file)
            self.assertTrue(exported_path.exists())
            self.assertEqual(exported_path, json_file.resolve())

            # Step 2: Load JSON
            loaded_data = load_analysis_json(json_file)
            self.assertEqual(loaded_data["symbol"], "TEST")
            self.assertEqual(loaded_data["metadata"]["request_date"], "2025-10-31")
            self.assertEqual(len(loaded_data["best_buys"]), 1)
            self.assertEqual(loaded_data["predictive"]["recommendation"], "STRONG BUY")

    def test_generate_html_dashboard_with_embedded_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.html"
            json_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.json"

            # Step 1: Export JSON
            export_analysis_json(self.mock_analysis, json_file)
            self.assertTrue(json_file.exists())

            # Step 2: Read JSON and generate HTML
            loaded_data = load_analysis_json(json_file)
            out_html = generate_html_dashboard(loaded_data, html_file, json_path=json_file)
            self.assertTrue(out_html.exists())

            html_text = out_html.read_text(encoding="utf-8")

            # Verify embedded data container exists
            self.assertIn('<script id="report-data" type="application/json">', html_text)
            self.assertIn('</script>', html_text)

            # Verify JavaScript reads from embedded data container
            self.assertIn("const REPORT_DATA = JSON.parse(document.getElementById('report-data').textContent);", html_text)
            self.assertIn("const RAW_HISTORY = REPORT_DATA.historical_data || [];", html_text)
            self.assertIn("const BEST_BUYS = REPORT_DATA.best_buys || [];", html_text)
            self.assertIn("const PREDICTIVE = REPORT_DATA.predictive || {};", html_text)

            # Verify embedded payload can be extracted and parsed identically
            start_marker = '<script id="report-data" type="application/json">\n'
            end_marker = '\n  </script>'
            start_idx = html_text.find(start_marker) + len(start_marker)
            end_idx = html_text.find(end_marker, start_idx)
            embedded_json_str = html_text[start_idx:end_idx].strip()

            parsed_embedded = json.loads(embedded_json_str)
            self.assertEqual(parsed_embedded["symbol"], "TEST")
            self.assertEqual(parsed_embedded["metadata"]["symbol"], "TEST")
            self.assertEqual(len(parsed_embedded["historical_data"]), len(self.df))

    def test_generate_html_dashboard_from_path_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.html"
            json_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.json"

            export_analysis_json(self.mock_analysis, json_file)

            # Pass string path directly as data_input
            out_html = generate_html_dashboard(str(json_file), html_file)
            self.assertTrue(out_html.exists())
            html_text = out_html.read_text(encoding="utf-8")
            self.assertIn('<script id="report-data" type="application/json">', html_text)

    def test_backward_compatibility_with_raw_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.html"
            # Passing raw dict with pd.DataFrame should auto-generate companion .json
            out_html = generate_html_dashboard(self.mock_analysis, html_file)
            self.assertTrue(out_html.exists())

            companion_json = html_file.with_suffix(".json")
            self.assertTrue(companion_json.exists())

    def test_modular_card_builders(self):
        # 1. Projections
        proj_html = build_projection_cards_html(self.mock_analysis["projections"])
        self.assertIn("6-Month Projection", proj_html)
        self.assertIn("72% Prob", proj_html)
        self.assertEqual(build_projection_cards_html({}), "")

        # 2. Regime
        regime_html = build_regime_card_html(self.mock_analysis["regime"])
        self.assertIn("Bayesian Online Changepoint Detection (BOCD)", regime_html)
        self.assertIn("Bull Trend", regime_html)
        self.assertEqual(build_regime_card_html(None), "")

        # 3. Microstructure
        micro_html = build_microstructure_card_html(self.mock_analysis["microstructure"])
        self.assertIn("YTD Anchored VWAP", micro_html)
        self.assertIn("Volume Profile (KDE)", micro_html)
        self.assertEqual(build_microstructure_card_html(None), "")

        # 4. Derivatives
        deriv_html = build_derivatives_card_html(self.mock_analysis["derivatives"], spot_price=140.0)
        self.assertIn("Institutional Derivatives &amp; Dealer Gamma Exposure (GEX)", deriv_html)
        self.assertIn("Gamma Flip Point", deriv_html)
        self.assertIn("CALL WALL", deriv_html)
        self.assertEqual(build_derivatives_card_html(None), "")

        # 5. Events
        events_html = build_events_card_html(self.mock_analysis["events"])
        self.assertIn("Corporate Catalyst Awareness &amp; Event Risk (PEAD Models)", events_html)
        self.assertIn("Quarterly Earnings Surprise &amp; Post-Announcement Drift History", events_html)
        self.assertEqual(build_events_card_html(None), "")

    def test_cli_from_json(self):
        from unittest.mock import patch
        from visualize_stock_analysis import main
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "TEST_analysis_report_2025-10-31.json"
            export_analysis_json(self.mock_analysis, json_file)

            report_dir = Path(tmpdir) / "out_reports"
            test_argv = ["visualize_stock_analysis.py", "--from_json", str(json_file), "--report_dir", str(report_dir)]
            with patch("sys.argv", test_argv):
                main()

            expected_html = report_dir / "TEST_analysis_report_2025-10-31.html"
            self.assertTrue(expected_html.exists())
            html_text = expected_html.read_text(encoding="utf-8")
            self.assertIn('<script id="report-data" type="application/json">', html_text)

    def test_cli_json_only(self):
        from unittest.mock import patch
        from visualize_stock_analysis import main
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            self.df.to_csv(data_dir / "TEST.csv", index=False)

            report_dir = Path(tmpdir) / "reports"
            test_argv = [
                "visualize_stock_analysis.py",
                "--symbol", "TEST",
                "--data_dir", str(data_dir),
                "--report_dir", str(report_dir),
                "--request_date", "2025-10-31",
                "--json_only",
                "--no-auto_download",
            ]
            with patch("sys.argv", test_argv):
                main()

            expected_json = report_dir / "TEST_analysis_report_2025-10-31.json"
            expected_html = report_dir / "TEST_analysis_report_2025-10-31.html"
            self.assertTrue(expected_json.exists())
            self.assertFalse(expected_html.exists())


if __name__ == "__main__":
    unittest.main()
