#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Stock Performance & Predictive Buy Timing Engine
==============================================================================
"""

import sys
import tempfile
import unittest
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root and scripts directory to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stock_analysis_engine import (
    load_stock_data,
    is_data_up_to_date,
    compute_performance_summary,
    detect_historical_best_buys,
    predict_future_buy_timing,
    compute_multi_period_projections,
    run_stock_analysis,
    _standardize_stock_df,
)
from visualize_stock_analysis import generate_html_dashboard, resolve_report_path


class TestStockAnalysisEngine(unittest.TestCase):
    """Test suite for the stock analysis and predictive buy timing engine."""

    def setUp(self):
        """Generate a realistic synthetic 5-year stock price history."""
        np.random.seed(42)
        # 5 years of daily trading data (~1260 days)
        dates = pd.bdate_range(start="2021-01-04", periods=1260).strftime("%Y-%m-%d")
        
        # Start at 100, add cyclical swings, upward drift, and dips
        base_price = 100.0
        prices = [base_price]
        for i in range(1, len(dates)):
            # Random walk with slight positive drift and periodic cycle
            cycle = 0.005 * np.sin(i / 30.0)
            ret = np.random.normal(0.0005, 0.015) + cycle
            new_p = max(10.0, prices[-1] * (1.0 + ret))
            prices.append(new_p)

        self.df_synthetic = pd.DataFrame({
            "date": dates,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1_000_000 + int(np.random.uniform(-200_000, 200_000)) for _ in prices],
            "symbol": "TEST",
        })

    def test_standardize_stock_df(self):
        """Test standardization of messy column names and non-sorted dates."""
        messy_df = pd.DataFrame({
            "Date": ["2024-01-05", "2024-01-02", "2024-01-03"],
            "Close": [105.0, 100.0, 102.0],
            "Volume": [1000, 2000, 1500],
        })
        std_df = _standardize_stock_df(messy_df, "TEST")
        self.assertEqual(list(std_df["date"]), ["2024-01-02", "2024-01-03", "2024-01-05"])
        self.assertTrue("open" in std_df.columns)
        self.assertTrue("high" in std_df.columns)
        self.assertTrue("low" in std_df.columns)

    def test_compute_performance_summary(self):
        """Test calculation of 1Y, 3Y, and 5Y performance metrics."""
        summary = compute_performance_summary(self.df_synthetic, periods_years=[1, 3, 5])
        
        self.assertEqual(summary["symbol"], "TEST")
        self.assertEqual(summary["total_history_days"], 1260)
        
        # Verify 1Y metrics
        p1 = summary["periods"]["1Y"]
        self.assertTrue(p1["available"])
        self.assertGreater(p1["trading_days"], 200)
        self.assertIn("total_return_pct", p1)
        self.assertIn("cagr_pct", p1)
        self.assertIn("max_drawdown_pct", p1)
        self.assertLessEqual(p1["max_drawdown_pct"], 0.0)
        self.assertIn("sharpe_ratio", p1)
        self.assertIn("annual_volatility_pct", p1)

        # Verify 3Y & 5Y metrics
        p3 = summary["periods"]["3Y"]
        p5 = summary["periods"]["5Y"]
        self.assertTrue(p3["available"])
        self.assertTrue(p5["available"])
        self.assertGreater(p5["trading_days"], 1000)

    def test_detect_historical_best_buys(self):
        """Test identification of optimal historical buy points."""
        best_buys = detect_historical_best_buys(self.df_synthetic, periods_years=[1, 3, 5])
        
        self.assertIn("1Y", best_buys)
        self.assertIn("3Y", best_buys)
        self.assertIn("5Y", best_buys)

        # 5Y should have identified major troughs
        buys_5y = best_buys["5Y"]
        self.assertGreater(len(buys_5y), 0)
        
        first_buy = buys_5y[0]
        self.assertIn("date", first_buy)
        self.assertIn("price", first_buy)
        self.assertIn("peak_date", first_buy)
        self.assertIn("peak_price", first_buy)
        self.assertIn("max_gain_pct", first_buy)
        self.assertGreater(first_buy["max_gain_pct"], 0)
        self.assertIn("rationale", first_buy)

    def test_predict_future_buy_timing(self):
        """Test 3-month forward predictive analysis."""
        pred = predict_future_buy_timing(self.df_synthetic, forecast_days=63)
        
        self.assertEqual(pred["forecast_days"], 63)
        self.assertIn("recommendation", pred)
        self.assertIn("action_summary", pred)
        self.assertIn("optimal_entry_range", pred)
        self.assertEqual(len(pred["optimal_entry_range"]), 2)
        self.assertLessEqual(pred["optimal_entry_range"][0], pred["optimal_entry_range"][1])
        
        self.assertIn("optimal_buy_window", pred)
        self.assertIn("start_date", pred["optimal_buy_window"])
        self.assertIn("end_date", pred["optimal_buy_window"])
        
        self.assertIn("target_price_3m", pred)
        self.assertIn("expected_return_pct", pred)
        self.assertIn("stop_loss", pred)
        self.assertIn("risk_reward_ratio", pred)

        # Verify forecast series
        series = pred["forecast_series"]
        self.assertEqual(len(series), 63)
        self.assertLessEqual(series[0]["bear_p10"], series[0]["bull_p90"])

    def test_load_stock_data_from_csv(self):
        """Test loading from custom CSV directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source_dir = data_dir / "source"
            source_dir.mkdir()
            
            csv_path = source_dir / "AAPL.csv"
            self.df_synthetic.to_csv(csv_path, index=False)
            
            max_date = self.df_synthetic["date"].max()
            loaded_df = load_stock_data("AAPL", data_dir, request_date=max_date)
            self.assertEqual(len(loaded_df), 1260)
            self.assertEqual(loaded_df["symbol"].iloc[0], "AAPL")

    def test_html_dashboard_generation(self):
        """Test end-to-end HTML visual dashboard output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "AAPL.csv").write_text(self.df_synthetic.to_csv(index=False), encoding="utf-8")

            max_date = self.df_synthetic["date"].max()
            analysis = run_stock_analysis("AAPL", data_dir, forecast_days=63, request_date=max_date)
            report_path = data_dir / "report.html"
            
            out_file = generate_html_dashboard(analysis, report_path)
            self.assertTrue(out_file.exists())
            
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("AAPL", content)
            self.assertIn("Historical Performance & Best Times to Buy", content)
            self.assertIn("3-Month Predictive Buy Analysis", content)
            self.assertIn("Historical Best Buy Opportunities Ranked", content)
            self.assertIn("optimal_buy_window", content)
            # Verify 'Historical' instead of 'Horizon'
            self.assertIn("1-Year Historical", content)
            self.assertIn("3-Year Historical", content)
            self.assertIn("5-Year Historical", content)
            self.assertNotIn("1-Year Horizon", content)
            # Verify Forward Return Projections row
            self.assertIn("Forward Return Projections &amp; Probability Analysis", content)
            self.assertIn("Probability Score", content)

    def test_multi_period_projections(self):
        """Test 6M, 1Y, 2Y, and 3Y multi-period return projections and probability scoring."""
        proj = compute_multi_period_projections(self.df_synthetic)
        self.assertIsInstance(proj, dict)
        self.assertEqual(set(proj.keys()), {"6M", "1Y", "2Y", "3Y"})

        for key in ["6M", "1Y", "2Y", "3Y"]:
            item = proj[key]
            self.assertIn("projected_return_pct", item)
            self.assertIn("projected_cagr_pct", item)
            self.assertIn("base_target_price", item)
            self.assertIn("bear_price", item)
            self.assertIn("bull_price", item)
            self.assertIn("probability_score", item)
            self.assertIn("confidence", item)

            self.assertGreater(item["base_target_price"], 0)
            self.assertLessEqual(item["bear_price"], item["bull_price"])
            self.assertGreaterEqual(item["probability_score"], 0.0)
            self.assertLessEqual(item["probability_score"], 100.0)
            self.assertIsInstance(item["confidence"], str)

    def test_auto_download_missing_symbol(self):
        """Test that missing symbol raises FileNotFoundError if auto_download=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_stock_data("NONEXISTENT", data_dir, auto_download=False)

    def test_resolve_report_path(self):
        """Test default and custom report directory path resolution with request date."""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        # 1. Default directory ('reports')
        p_default = resolve_report_path("MSFT")
        self.assertEqual(p_default.name, f"MSFT_analysis_report_{today_str}.html")
        self.assertEqual(p_default.parent.name, "reports")

        # Custom date string
        p_custom_date = resolve_report_path("MSFT", report_date="2026-01-15")
        self.assertEqual(p_custom_date.name, "MSFT_analysis_report_2026-01-15.html")

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = (Path(tmpdir) / "custom_reports").resolve()
            
            # 2. Custom report_dir
            p_custom = resolve_report_path("NVDA", report_dir=custom_dir)
            self.assertEqual(p_custom.name, f"NVDA_analysis_report_{today_str}.html")
            self.assertEqual(p_custom.parent, custom_dir)
            self.assertTrue(custom_dir.exists())

            # 3. Custom output file (.html)
            specific_file = (Path(tmpdir) / "my_report.html").resolve()
            p_file = resolve_report_path("AAPL", output=specific_file)
            self.assertEqual(p_file, specific_file)

            # 4. Custom output directory (without .html extension)
            dir_output = (Path(tmpdir) / "dir_output").resolve()
            p_dir_out = resolve_report_path("TSLA", output=dir_output)
            self.assertEqual(p_dir_out.name, f"TSLA_analysis_report_{today_str}.html")
    def test_is_data_up_to_date(self):
        """Test data freshness checking logic across weekdays and weekends."""
        max_date = self.df_synthetic["date"].max()

        # 1. Exact match with max_date -> Fresh
        is_fresh, latest, expected = is_data_up_to_date(self.df_synthetic, request_date=max_date)
        self.assertTrue(is_fresh)
        self.assertEqual(latest, max_date)
        self.assertEqual(expected, max_date)

        # 2. Past date -> Fresh
        is_fresh, latest, expected = is_data_up_to_date(self.df_synthetic, request_date="2022-06-15")
        self.assertTrue(is_fresh)

        # 3. Future date -> Stale
        future_req = "2028-12-01"
        is_fresh, latest, expected = is_data_up_to_date(self.df_synthetic, request_date=future_req)
        self.assertFalse(is_fresh)
        self.assertEqual(expected, future_req)

        # 4. Weekend check: Saturday request looks for Friday
        # 2026-09-05 is Saturday -> expected Friday 2026-09-04
        # If df has data up to 2026-09-04, Saturday request is up-to-date
        df_friday = pd.DataFrame({"date": ["2026-09-04"], "close": [100.0]})
        is_fresh, latest, expected = is_data_up_to_date(df_friday, request_date="2026-09-05")
        self.assertTrue(is_fresh)
        self.assertEqual(expected, "2026-09-04")

        # 5. Weekend check: Sunday request looks for Friday
        is_fresh, latest, expected = is_data_up_to_date(df_friday, request_date="2026-09-06")
        self.assertTrue(is_fresh)
        self.assertEqual(expected, "2026-09-04")

    def test_freshness_in_run_stock_analysis(self):
        """Test that run_stock_analysis includes freshness verification metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "AAPL.csv").write_text(self.df_synthetic.to_csv(index=False), encoding="utf-8")

            max_date = self.df_synthetic["date"].max()
            analysis = run_stock_analysis("AAPL", data_dir, auto_download=False, request_date=max_date)
            
            self.assertTrue(analysis["is_up_to_date"])
            self.assertEqual(analysis["latest_data_date"], max_date)
            self.assertEqual(analysis["request_date"], max_date)

    def test_predict_future_buy_timing_with_bocd(self):
        """Test that predict_future_buy_timing correctly integrates BOCD regime states and hazard."""
        regime_risk_off = {
            "state": 2,
            "name": "High-Vol Liquidation / Risk-Off",
            "changepoint_prob_pct": 32.5,
            "expected_run_length_days": 25.0,
            "vol_21d_pct": 38.0,
            "vol_ratio": 1.25,
            "risk_multiplier": 0.4,
        }
        pred_risk_off = predict_future_buy_timing(
            self.df_synthetic,
            forecast_days=63,
            regime=regime_risk_off,
        )
        self.assertEqual(pred_risk_off["recommendation"], "RISK-OFF / CAPITAL PRESERVATION")
        self.assertIn("BOCD State 2", pred_risk_off["action_summary"])
        self.assertEqual(pred_risk_off["bocd_regime_state"], 2)
        self.assertEqual(pred_risk_off["bocd_regime_name"], "High-Vol Liquidation / Risk-Off")
        self.assertIsNotNone(pred_risk_off["bocd_forward_changepoint_prob_pct"])
        self.assertGreater(pred_risk_off["bocd_forward_changepoint_prob_pct"], 50.0)

        # Bull Regime test
        regime_bull = {
            "state": 0,
            "name": "Low-Vol Trending Bull",
            "changepoint_prob_pct": 2.1,
            "expected_run_length_days": 80.0,
            "vol_21d_pct": 14.0,
            "vol_ratio": 0.85,
            "risk_multiplier": 1.0,
        }
        pred_bull = predict_future_buy_timing(
            self.df_synthetic,
            forecast_days=63,
            regime=regime_bull,
        )
        self.assertEqual(pred_bull["recommendation"], "STRONG BUY / TREND ACCUMULATION")
        self.assertEqual(pred_bull["bocd_regime_state"], 0)

    def test_multi_period_projections_conditioned(self):
        """Test multi-period projections dynamically conditioned on BOCD regime, vol surface, and AVWAP."""
        uncond = compute_multi_period_projections(self.df_synthetic)
        
        regime_risk_off = {
            "state": 2,
            "name": "High-Vol Liquidation / Risk-Off",
            "changepoint_prob_pct": 25.0,
            "expected_run_length_days": 30.0,
            "vol_21d_pct": 40.0,
            "vol_ratio": 1.20,
            "risk_multiplier": 0.4,
        }
        cond_risk_off = compute_multi_period_projections(
            self.df_synthetic,
            regime=regime_risk_off,
        )

        # 6M projection under Risk-Off should have lower return and lower confidence score
        self.assertLess(
            cond_risk_off["6M"]["projected_return_pct"],
            uncond["6M"]["projected_return_pct"],
        )
        self.assertLess(
            cond_risk_off["6M"]["probability_score"],
            uncond["6M"]["probability_score"],
        )
        self.assertTrue(cond_risk_off["6M"]["regime_conditioned"])

        # Forward changepoint probability should increase with horizon length
        self.assertIsNotNone(cond_risk_off["6M"]["bocd_changepoint_prob_pct"])
        self.assertLessEqual(
            cond_risk_off["6M"]["bocd_changepoint_prob_pct"],
            cond_risk_off["1Y"]["bocd_changepoint_prob_pct"],
        )
        self.assertLessEqual(
            cond_risk_off["1Y"]["bocd_changepoint_prob_pct"],
            cond_risk_off["3Y"]["bocd_changepoint_prob_pct"],
        )

    def test_predict_future_buy_timing_gex_conditioning(self):
        """Test that Dealer GEX (+GEX vs -GEX) conditions future buy timing, support/resistance, and volatility."""
        last_price = float(self.df_synthetic["close"].iloc[-1])

        # Positive GEX (Mean-reverting, pinned to call wall, volatility dampened)
        pos_gex_deriv = {
            "gex": {
                "regime": "+GEX (Low Volatility / Mean Reversion)",
                "net_gex_dollar_per_1pct": 25_000_000.0,
                "call_wall_strike": round(last_price * 1.10, 2),
                "put_wall_strike": round(last_price * 0.92, 2),
                "gamma_flip_price": round(last_price * 0.95, 2),
                "max_pain_strike": round(last_price * 1.02, 2),
            },
            "vol_surface": {
                "atm_iv_30d_pct": 22.5,
                "vrp_pct": 3.2,
                "skew_25d_rr_pct": -1.8,
            },
        }

        pred_pos = predict_future_buy_timing(
            self.df_synthetic,
            forecast_days=63,
            derivatives=pos_gex_deriv,
        )

        self.assertEqual(pred_pos["gex_regime"], "+GEX (Low Volatility / Mean Reversion)")
        self.assertEqual(pred_pos["gex_vol_multiplier"], 0.85)
        self.assertEqual(pred_pos["call_wall_price"], pos_gex_deriv["gex"]["call_wall_strike"])
        self.assertEqual(pred_pos["put_wall_price"], pos_gex_deriv["gex"]["put_wall_strike"])
        self.assertEqual(pred_pos["gamma_flip_price"], pos_gex_deriv["gex"]["gamma_flip_price"])
        self.assertIn("+GEX", pred_pos["action_summary"])
        self.assertIn("Dealer counter-trading pins price", pred_pos["action_summary"])

        # Negative GEX (High volatility, trending, volatility amplified)
        neg_gex_deriv = {
            "gex": {
                "regime": "-GEX (High Volatility / Directional Trend)",
                "net_gex_dollar_per_1pct": -18_000_000.0,
                "call_wall_strike": round(last_price * 1.08, 2),
                "put_wall_strike": round(last_price * 0.88, 2),
                "gamma_flip_price": round(last_price * 1.04, 2),
                "max_pain_strike": round(last_price * 0.98, 2),
            },
            "vol_surface": {
                "atm_iv_30d_pct": 45.0,
                "vrp_pct": -4.5,
                "skew_25d_rr_pct": -6.2,
            },
        }

        pred_neg = predict_future_buy_timing(
            self.df_synthetic,
            forecast_days=63,
            derivatives=neg_gex_deriv,
        )

        self.assertEqual(pred_neg["gex_regime"], "-GEX (High Volatility / Directional Trend)")
        self.assertEqual(pred_neg["gex_vol_multiplier"], 1.25)
        self.assertIn("-GEX", pred_neg["action_summary"])
        self.assertIn("Dealer dynamic hedging accelerates drops", pred_neg["action_summary"])

    def test_run_stock_analysis_with_derivatives(self):
        """Test full run_stock_analysis pipeline including derivatives computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "TEST_GEX.csv").write_text(self.df_synthetic.to_csv(index=False), encoding="utf-8")
            max_date = self.df_synthetic["date"].max()
            analysis = run_stock_analysis("TEST_GEX", data_dir, auto_download=False, request_date=max_date)

            self.assertIn("derivatives", analysis)
            deriv = analysis["derivatives"]
            self.assertIsNotNone(deriv)
            self.assertIn("gex", deriv)
            self.assertIn("vol_surface", deriv)
            self.assertIn("net_gex_dollar_per_1pct", deriv["gex"])
            self.assertIn("call_wall_strike", deriv["gex"])
            self.assertIn("put_wall_strike", deriv["gex"])
            self.assertIn("regime", deriv["gex"])

            # Predictive buy analysis should reflect GEX
            pred = analysis["predictive"]
            self.assertIn("gex_regime", pred)
            self.assertIn("call_wall_price", pred)
            self.assertIn("put_wall_price", pred)

if __name__ == "__main__":
    unittest.main()

