"""Unit tests for evaluate_earnings_gamma_squeeze orchestrator."""

import unittest
import pandas as pd
from qlib.contrib.derivatives.earnings_gamma_squeeze_engine import evaluate_earnings_gamma_squeeze
from qlib.contrib.derivatives.data_provenance_guard import DataProvenance


class TestEarningsGammaSqueezeEngine(unittest.TestCase):
    def setUp(self):
        self.spot = 150.0
        self.adtv = 2_000_000.0
        # Synthetic chain
        rows = []
        for strike in [140.0, 145.0, 150.0, 155.0, 160.0]:
            rows.append({
                "strike": strike,
                "option_type": "call",
                "openInterest": 2500,
                "impliedVolatility": 0.45,
                "dte": 7,
                "delta_call": 0.5,
                "delta_put": -0.5,
            })
            rows.append({
                "strike": strike,
                "option_type": "put",
                "openInterest": 1500,
                "impliedVolatility": 0.45,
                "dte": 7,
                "delta_call": 0.5,
                "delta_put": -0.5,
            })
        self.df_chain = pd.DataFrame(rows)

    def test_live_clearance_emits_probabilities(self):
        res = evaluate_earnings_gamma_squeeze(
            spot=self.spot,
            df_chain=self.df_chain,
            adtv_20=self.adtv,
            sue_score=2.0,
            short_interest_pct=0.18,
            provenance=DataProvenance.LIVE_OPRA_VERIFIED,
            is_pit_timestamp=True,
        )
        self.assertTrue(res["is_actionable"])
        self.assertEqual(res["safety_status"], "PRODUCTION_CLEAR")
        self.assertIsNotNone(res["calibrated_probabilities"]["p_positive_squeeze"])
        self.assertIsNotNone(res["calibrated_probabilities"]["conformal_bounds_positive"])
        self.assertIn("upper_squeeze_wall", res["acceleration_corridors"])
        self.assertIn("lower_trapdoor", res["acceleration_corridors"])

        # Validate calibrate_post_earnings_volatility_surface presence
        self.assertIn("calibrate_post_earnings_volatility_surface", res)
        vol_surf = res["calibrate_post_earnings_volatility_surface"]
        self.assertIn("expected_jump_pct", vol_surf)
        self.assertIn("post_earnings_iv", vol_surf)
        self.assertIn("volatility_crush_pct", vol_surf)

        # Validate factor_orthogonalization presence
        self.assertIn("factor_orthogonalization", res)
        ortho = res["factor_orthogonalization"]
        self.assertTrue(ortho["is_orthogonalized"])
        self.assertIn("gsi_orthogonal", ortho)
        self.assertIn("idiosyncratic_alpha_ratio", ortho)

        # Validate earnings_event_clock presence
        self.assertIn("earnings_event_clock", res)
        clock = res["earnings_event_clock"]
        self.assertEqual(clock["reporting_time"], "AMC")
        self.assertIn("disallowed_fill_rule", clock)

        # Validate backtesting_protocol presence
        self.assertIn("backtesting_protocol", res)
        bp = res["backtesting_protocol"]
        self.assertIn("purged_walk_forward_cv", bp)
        self.assertIn("almgren_chriss_market_impact", bp)
        self.assertIn("borrow_fee_engine", bp)
        self.assertIn("deflated_sharpe_ratio", bp)
        self.assertIn("verifiable_replication_event_panel", bp)
        self.assertIn("strategy_rules", bp)
        self.assertIn("council_interrogation_outcomes", bp)

        # Validate evaluation_matrix presence
        self.assertIn("evaluation_matrix", res)
        self.assertIn("t_plus_1_to_t_plus_5", res["evaluation_matrix"])

    def test_synthetic_suppression_locks_action(self):
        # INVARIANT: Synthetic surface must suppress actionable signals
        res = evaluate_earnings_gamma_squeeze(
            spot=self.spot,
            df_chain=self.df_chain,
            adtv_20=self.adtv,
            sue_score=3.5,
            short_interest_pct=0.30,
            provenance=DataProvenance.SYNTHETIC_RESEARCH_FALLBACK,
            is_pit_timestamp=True,
        )
        self.assertFalse(res["is_actionable"])
        self.assertEqual(res["safety_status"], "ACTION_SUPPRESSED")
        self.assertEqual(res["recommended_action"], "RESEARCH_ONLY_NO_ACTION")
        # Quantitative Transparency: Synthetic research data computes theoretical probabilities
        self.assertIsNotNone(res["calibrated_probabilities"]["p_positive_squeeze"])
        self.assertGreater(res["calibrated_probabilities"]["p_positive_squeeze"], 0.0)
        self.assertIn("calibrated_prob_squeeze", res["calibrated_probabilities"])
        self.assertIn("gsi_positive", res["gsi_scores"])
        self.assertEqual(res["gsi_scores"]["gsi_positive"], res["gsi_scores"]["gsi_positive_raw"])
        # Invariant: Backtesting protocol is still available for academic reference
        self.assertIn("backtesting_protocol", res)
        self.assertIn("deflated_sharpe_ratio", res["backtesting_protocol"])

    def test_corridor_geometric_ordering_invariant(self):
        # Strict Invariant: Spot < Trigger Strike < Upper Squeeze Wall
        # Downside Invariant: Lower Trapdoor < Downside Trigger < Spot
        for test_spot in [100.0, 499.70, 750.25]:
            res = evaluate_earnings_gamma_squeeze(
                spot=test_spot,
                df_chain=pd.DataFrame(),  # Tests synthetic chain fallback
                adtv_20=self.adtv,
                sue_score=1.5,
                short_interest_pct=0.08,
                provenance=DataProvenance.SYNTHETIC_RESEARCH_FALLBACK,
                is_pit_timestamp=True,
            )
            corridors = res["acceleration_corridors"]
            trigger = corridors["trigger_strike"]
            wall = corridors["upper_squeeze_wall"]
            trap = corridors["lower_trapdoor"]
            trigger_down = corridors["downside_trigger"]

            self.assertLess(test_spot, trigger, f"Failed: Spot {test_spot} < Trigger {trigger}")
            self.assertLess(trigger, wall, f"Failed: Trigger {trigger} < Wall {wall}")
            self.assertLess(trap, trigger_down, f"Failed: Trap {trap} < TriggerDown {trigger_down}")
            self.assertLess(trigger_down, test_spot, f"Failed: TriggerDown {trigger_down} < Spot {test_spot}")

            # Also verify forced dealer hedging has non-zero shares and dollar demand
            fd = res["forced_dealer_hedging"]
            self.assertIn("dealer_shares_to_buy", fd)
            self.assertIn("dealer_dollar_demand", fd)
            self.assertGreater(fd["dealer_shares_to_buy"], 0)
            self.assertGreater(fd["dealer_dollar_demand"], 0.0)


if __name__ == "__main__":
    unittest.main()

