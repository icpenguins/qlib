#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Two-Step Reporting Pipeline & JSON Decoupling in visualize_stock_analysis.py
"""

import sys
import json
import tempfile
import unittest
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
    build_buy_timing_verdict_banner_html,
    build_gamma_squeeze_spike_card_html,
    build_multi_horizon_matrix_card_html,
    build_backtesting_protocol_card_html,
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
            },
            "earnings_gamma_squeeze": {
                "calibrate_post_earnings_volatility_surface": {
                    "expected_jump_pct": 8.5,
                    "event_variance": 0.0072,
                    "post_earnings_iv": 0.35,
                    "historical_crush_ratio": 0.42,
                    "crush_source": "winsorized_median",
                    "min_samples_threshold": 4,
                },
                "forced_dealer_hedging": {
                    "dealer_shares_to_buy": 1250000,
                    "dealer_dollar_demand": 175000000.0,
                    "dealer_hedging_velocity": "High Convexity",
                    "pct_adtv_demand": 28.5,
                },
                "liquidity_impact": {
                    "expected_spread_widening_bps": 18.2,
                    "expected_slippage_bps": 24.5,
                    "turnover_ratio": 0.045,
                    "liquidity_regime": "Expanding Spread / Thin Book",
                },
                "gsi_scores": {
                    "gsi_raw": 78.5,
                    "gsi_positive": 82.4,
                    "gsi_negative": 12.0,
                    "is_positive_squeeze_candidate": True,
                    "is_negative_squeeze_candidate": False,
                },
                "factor_orthogonalization": {
                    "residual_gsi": 76.2,
                    "fama_french_exposure": 0.15,
                    "beta_adj_factor": 1.05,
                },
                "calibrated_probabilities": {
                    "calibrated_prob_squeeze": 84.5,
                    "probability_positive_spike": 84.5,
                    "confidence_band": [78.2, 89.6],
                },
                "earnings_event_clock": {
                    "t0_timestamp": "2025-10-31 16:05:00 AMC",
                    "t1_timestamp": "2025-11-03 09:30:00 OPEN",
                    "t1_open_action": "Execute limit buy in optimal corridor",
                    "t5_exit_action": "De-gross into Upper Squeeze Wall",
                    "execution_window": "Immediate T+1 Open through T+5 Close",
                },
                "acceleration_corridors": {
                    "trigger_strike": 145.0,
                    "upper_squeeze_wall": 165.0,
                    "lower_gamma_trap": 132.0,
                    "acceleration_slope": 1.85,
                },
                "is_actionable": True,
                "provenance": "live_exchange_feed",
                "safety_status": "ACTIONABLE",
                "recommended_action": "ENTER_5DAY_SQUEEZE",
            },
            "backtesting_protocol": {
                "purged_walk_forward_cv": {
                    "train_folds": 5,
                    "test_folds": 5,
                    "embargo_days": 10,
                    "is_purged": True,
                },
                "almgren_chriss_market_impact": {
                    "temp_impact_bps": 14.5,
                    "perm_impact_bps": 10.0,
                    "half_life_decay": 0.5,
                    "total_slippage_bps": 24.5,
                },
                "borrow_fee_engine": {
                    "borrow_fee_bps": 65.0,
                    "is_hard_to_borrow": False,
                    "utilization_pct": 34.5,
                    "annualized_cost": 0.65,
                },
                "deflated_sharpe_ratio": {
                    "best_sharpe": 1.85,
                    "expected_max_sharpe_hurdle": 1.30,
                    "dsr_probability": 96.2,
                    "n_trials": 240,
                    "is_statistically_significant": True,
                },
                "verifiable_replication_event_panel": {
                    "n_events": 128,
                    "win_rate": 0.688,
                    "profit_factor": 2.45,
                    "cagr_pct": 28.5,
                    "max_drawdown_pct": -9.8,
                    "calmar_ratio": 2.91,
                },
                "strategy_rules": {
                    "entry_rule": "T1 Market Open Limit Corridor",
                    "exit_rule": "Upper Squeeze Wall or T+5 Close",
                    "stop_loss_peg": "Lower Gamma Trap",
                    "max_holding_period": 5,
                },
                "council_interrogation_outcomes": {
                    "dr_vance": {"verdict": "APPROVED", "notes": "Post-earnings volatility surface calibrated with winsorized median."},
                    "marcus_reynolds": {"verdict": "APPROVED", "notes": "Almgren-Chriss slippage and market impact validated."},
                    "dr_rostova": {"verdict": "APPROVED", "notes": "Residual GSI orthogonalized from Fama-French factors; isotonic probability valid."},
                    "julian_montgomery": {"verdict": "APPROVED", "notes": "No AMC close fill; borrow fee 65 bps general collateral."},
                    "sophia_chen": {"verdict": "APPROVED", "notes": "SUE score +1.25 exceeds 0.5 beat hurdle."},
                    "arthur_pendelton": {"verdict": "APPROVED", "notes": "Capital allocation approved with strict stop-loss peg."},
                },
            },
            "evaluation_matrix": {
                "t_plus_1_to_5": {
                    "direction": "BULLISH",
                    "conviction_score": 88.0,
                    "expected_return_pct": 8.5,
                    "sharpe_ratio": 2.15,
                    "primary_driver": "Convex Dealer Delta Hedging Squeeze",
                    "optimal_action": "Aggressive Tactical Buy at T1 Open",
                    "risk_factors": ["Overnight binary announcement gap"],
                },
                "1M": {
                    "direction": "BULLISH",
                    "conviction_score": 78.0,
                    "expected_return_pct": 6.2,
                    "sharpe_ratio": 1.75,
                    "primary_driver": "PEAD Earnings Momentum Drift",
                    "optimal_action": "Hold through 30-day post-announcement window",
                    "risk_factors": ["Macro CPI release volatility"],
                },
                "6M": {
                    "direction": "ACCUMULATE",
                    "conviction_score": 72.0,
                    "expected_return_pct": 14.5,
                    "sharpe_ratio": 1.45,
                    "primary_driver": "Volume Profile Value Area Support",
                    "optimal_action": "Accumulate on pullbacks to YTD AVWAP",
                    "risk_factors": ["BOCD regime transition hazard"],
                },
                "1Y": {
                    "direction": "BULLISH",
                    "conviction_score": 75.0,
                    "expected_return_pct": 22.0,
                    "sharpe_ratio": 1.35,
                    "primary_driver": "Fundamental Earnings Expansion & Compound Growth",
                    "optimal_action": "Core institutional long allocation",
                    "risk_factors": ["Sector rotation into defensives"],
                },
                "3Y": {
                    "direction": "ACCUMULATE",
                    "conviction_score": 70.0,
                    "expected_return_pct": 58.0,
                    "sharpe_ratio": 1.15,
                    "primary_driver": "Structural Industry Trend & Secular Margin Expansion",
                    "optimal_action": "Strategic cycle rebalancing",
                    "risk_factors": ["Macroeconomic interest rate cycles"],
                },
                "10Y": {
                    "direction": "BULLISH",
                    "conviction_score": 80.0,
                    "expected_return_pct": 195.0,
                    "sharpe_ratio": 1.05,
                    "primary_driver": "Durable Competitive Moat & Secular Reinvestment Rate",
                    "optimal_action": "Permanent compounder holding",
                    "risk_factors": ["Technological disruption"],
                },
            },
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
        self.assertEqual(payload["metadata"]["contract_version"], "1.2.0")
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
            self.assertIn("const GAMMA_SQUEEZE = REPORT_DATA.earnings_gamma_squeeze || {};", html_text)
            self.assertIn("const BACKTESTING = REPORT_DATA.backtesting_protocol || {};", html_text)
            self.assertIn("const EVALUATION_MATRIX = REPORT_DATA.evaluation_matrix || {};", html_text)

            # Verify presence of institutional HTML sections
            self.assertIn("EXECUTIVE BUY TIMING VERDICT BANNER", html_text)
            self.assertIn("Should It Be Bought?", html_text)
            self.assertIn("When Should It Be Bought?", html_text)
            self.assertIn("Next-Day to Next-Week (t+1 to t+5) Gamma Squeeze &amp; 5-Day Upward Spike Radar", html_text)
            self.assertIn("Multi-Horizon Institutional Conviction Matrix", html_text)
            self.assertIn("Institutional Backtesting Protocol &amp; Quantitative Risk Audit", html_text)

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
        self.assertEqual(build_derivatives_card_html(None, spot_price=0.0), "")

        # Fallback calibration when derivatives is None but spot_price > 0
        fallback_deriv_html = build_derivatives_card_html(None, spot_price=140.0)
        self.assertIn("Institutional Derivatives &amp; Dealer Gamma Exposure (GEX)", fallback_deriv_html)
        self.assertIn("CALIBRATED SYNTHETIC SURFACE", fallback_deriv_html)
        self.assertIn("Gamma Flip Point", fallback_deriv_html)

        # 5. Events
        events_html = build_events_card_html(self.mock_analysis["events"])
        self.assertIn("Corporate Catalyst Awareness &amp; Event Risk (PEAD Models)", events_html)
        self.assertIn("Quarterly Earnings Surprise &amp; Post-Announcement Drift History", events_html)
        self.assertEqual(build_events_card_html(None), "")

        # 6. Buy Timing Verdict Banner
        banner_html = build_buy_timing_verdict_banner_html(
            pred=self.mock_analysis["predictive"],
            gamma_squeeze=self.mock_analysis["earnings_gamma_squeeze"],
            eval_matrix=self.mock_analysis["evaluation_matrix"],
            spot_price=140.0,
        )
        self.assertIn("EXECUTIVE BUY TIMING VERDICT BANNER", banner_html)
        self.assertIn("Should It Be Bought?", banner_html)
        self.assertIn("When Should It Be Bought?", banner_html)
        self.assertIn("Optimal Entry Corridor", banner_html)
        self.assertIn("Invalidation Stop-Loss", banner_html)
        self.assertIn("5-Day Spike Potential", banner_html)
        self.assertEqual(build_buy_timing_verdict_banner_html(None, None), "")

        # Test Synthetic Provenance Safety Invariant in Verdict Banner
        synth_gamma = dict(self.mock_analysis["earnings_gamma_squeeze"])
        synth_gamma["provenance"] = "synthetic_research_fallback"
        synth_gamma["safety_status"] = "ACTION_SUPPRESSED"
        synth_banner = build_buy_timing_verdict_banner_html(
            pred=self.mock_analysis["predictive"],
            gamma_squeeze=synth_gamma,
            spot_price=140.0,
        )
        self.assertIn("SAFETY INVARIANT: SYNTHETIC RESEARCH DATA", synth_banner)
        self.assertIn("RESEARCH ONLY", synth_banner)

        # 7. Gamma Squeeze & 5-Day Upward Spike Radar Card
        spike_html = build_gamma_squeeze_spike_card_html(
            self.mock_analysis["earnings_gamma_squeeze"],
            spot_price=140.0,
        )
        self.assertIn("Next-Day to Next-Week (t+1 to t+5) Gamma Squeeze &amp; 5-Day Upward Spike Radar", spike_html)
        self.assertIn("ACTIVE 5-DAY UPWARD SPIKE DETECTED", spike_html)
        self.assertIn("Post-Earnings Vol Surface", spike_html)
        self.assertIn("Forced Dealer Hedging", spike_html)
        self.assertIn("Microstructure &amp; Liquidity", spike_html)
        self.assertIn("5-Day Execution Clock", spike_html)
        self.assertEqual(build_gamma_squeeze_spike_card_html(None), "")

        # 8. Multi-Horizon Conviction Matrix
        matrix_html = build_multi_horizon_matrix_card_html(self.mock_analysis["evaluation_matrix"])
        self.assertIn("Multi-Horizon Institutional Conviction Matrix", matrix_html)
        self.assertIn("Next-Day to Next-Week (5 Trading Days)", matrix_html)
        self.assertIn("5-DAY RADAR", matrix_html)
        self.assertIn("1 Month (21 Trading Days)", matrix_html)
        self.assertIn("6 Months (126 Trading Days)", matrix_html)
        self.assertIn("1 Year (252 Trading Days)", matrix_html)
        self.assertIn("3 Years (756 Trading Days)", matrix_html)
        self.assertIn("10 Years (2520 Trading Days)", matrix_html)
        self.assertEqual(build_multi_horizon_matrix_card_html(None), "")

        # 9. Institutional Backtesting Protocol & Quantitative Risk Audit Card
        backtest_html = build_backtesting_protocol_card_html(self.mock_analysis["backtesting_protocol"])
        self.assertIn("Institutional Backtesting Protocol &amp; Quantitative Risk Audit", backtest_html)
        self.assertIn("Deflated Sharpe Prob", backtest_html)
        self.assertIn("Purged Walk-Forward CV", backtest_html)
        self.assertIn("Execution Impact Engine", backtest_html)
        self.assertIn("Borrow Fee Engine", backtest_html)
        self.assertIn("@team-finance Council Interrogation &amp; Audit Sign-Offs", backtest_html)
        self.assertIn("Dr. Victoria Vance", backtest_html)
        self.assertIn("Marcus Reynolds", backtest_html)
        self.assertIn("Dr. Elena Rostova", backtest_html)
        self.assertIn("Julian Montgomery", backtest_html)
        self.assertIn("Sophia Chen", backtest_html)
        self.assertIn("Arthur Pendelton III", backtest_html)
        self.assertEqual(build_backtesting_protocol_card_html(None), "")

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

    def test_capital_preservation_execution_safety_invariants(self):
        """
        Verify that in Capital Preservation / Risk-Off regimes, all buy execution instructions,
        active buy windows, and entry corridors are strictly inhibited.
        Guarantees that a trader cannot receive a 'DO NOT BUY' verdict alongside limit-buy instructions.
        """
        # 1. Test Executive Verdict Banner suppression
        risk_off_pred = dict(self.mock_analysis["predictive"])
        risk_off_pred["recommendation"] = "RISK-OFF / CAPITAL PRESERVATION"
        risk_off_pred["is_capital_preservation"] = True
        risk_off_pred["is_entry_allowed"] = False
        risk_off_pred["execution_posture"] = "ENTRIES_INHIBITED"

        banner_html = build_buy_timing_verdict_banner_html(
            pred=risk_off_pred,
            gamma_squeeze=self.mock_analysis["earnings_gamma_squeeze"],
            eval_matrix=self.mock_analysis["evaluation_matrix"],
            spot_price=140.0,
        )
        self.assertIn("DO NOT BUY / CAPITAL PRESERVATION MODE", banner_html)
        self.assertIn("NO &mdash; STAND ASIDE", banner_html)
        self.assertIn("ENTRIES INHIBITED", banner_html)
        self.assertIn("No Active Buy Window &bull; Stand Aside", banner_html)
        self.assertIn("Capital Protection Floor", banner_html)
        self.assertIn("N/A &mdash; STAND ASIDE", banner_html)
        self.assertNotIn("Execute limit buy", banner_html)
        self.assertNotIn("Immediate Market Open limit entry", banner_html)

        # 2. Test 5-Day Execution Clock suppression in Gamma Squeeze Card
        spike_html = build_gamma_squeeze_spike_card_html(
            self.mock_analysis["earnings_gamma_squeeze"],
            spot_price=140.0,
            recommendation="RISK-OFF / CAPITAL PRESERVATION",
            pred=risk_off_pred,
        )
        self.assertIn("INACTIVE / STAND ASIDE (CAPITAL PRESERVATION)", spike_html)
        self.assertIn("STAND ASIDE", spike_html)
        self.assertIn("ENTRIES INHIBITED &mdash; STAND ASIDE (Risk-Off Regime)", spike_html)
        self.assertIn("SUSPENDED &mdash; CAPITAL PRESERVATION", spike_html)
        self.assertIn("No Active Position Authorized", spike_html)
        self.assertNotIn("Execute limit buy", spike_html)

        # 3. Test Full Dashboard HTML rendering under Capital Preservation
        with tempfile.TemporaryDirectory() as tmpdir:
            risk_off_analysis = dict(self.mock_analysis)
            risk_off_analysis["predictive"] = risk_off_pred
            out_html_path = Path(tmpdir) / "RISK_OFF_report.html"
            generate_html_dashboard(risk_off_analysis, out_html_path)
            self.assertTrue(out_html_path.exists())
            html_text = out_html_path.read_text(encoding="utf-8")

            # Must contain capital preservation badges and suppressed corridors
            self.assertIn("ENTRIES INHIBITED", html_text)
            self.assertIn("SUSPENDED (Risk-Off Regime)", html_text)
            self.assertIn("NO &mdash; STAND ASIDE", html_text)
            # Must NOT contain actionable limit buy instructions anywhere
            self.assertNotIn("Execute limit buy at 09:30 AM open", html_text)
            self.assertNotIn("Immediate Market Open limit entry", html_text)

    def test_gamma_squeeze_card_synthetic_and_corridor_invariants(self):
        # 1. Synthetic provenance test: verify theoretical metrics displayed without zeroing
        synth_gamma = dict(self.mock_analysis["earnings_gamma_squeeze"])
        synth_gamma["provenance"] = "synthetic_research_fallback"
        synth_gamma["is_actionable"] = False
        synth_gamma["safety_status"] = "ACTION_SUPPRESSED"
        synth_gamma["gsi_scores"] = {
            "gsi_positive_raw": 83.91,
            "gsi_positive": 83.91,
            "is_positive_squeeze_candidate": True,
        }
        synth_gamma["calibrated_probabilities"] = {
            "p_positive_squeeze": 0.845,
            "calibrated_prob_squeeze": 84.5,
        }
        # Inverted corridor mock: trigger strike $524.69 above wall $517.39
        synth_gamma["acceleration_corridors"] = {
            "trigger_strike": 524.69,
            "upper_squeeze_wall": 517.39,
            "lower_gamma_trap": 480.0,
        }
        spot = 499.70
        html = build_gamma_squeeze_spike_card_html(synth_gamma, spot_price=spot)

        # Theoretical simulation banner
        self.assertIn("THEORETICAL SPIKE SETUP (ACTION SUPPRESSED: SYNTHETIC DATA)", html)
        self.assertIn("SIMULATION", html)
        # Metrics not zeroed: 84.5% and 83.9
        self.assertIn("84.5%", html)
        self.assertIn("83.9 / 100", html)
        # Corridor geometry invariant enforced: Trigger Strike must NOT be 524.69 above 517.39
        # Clamped trigger = round(499.70 + 0.35 * (517.39 - 499.70), 2) = $505.89
        self.assertIn("$505.89", html)
        self.assertIn("$517.39", html)
        self.assertNotIn("$524.69", html)


if __name__ == "__main__":
    unittest.main()
