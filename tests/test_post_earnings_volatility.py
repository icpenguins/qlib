"""Unit tests for calibrate_post_earnings_volatility & calibrate_post_earnings_volatility_surface."""

import unittest
from qlib.contrib.derivatives.post_earnings_volatility import (
    calibrate_post_earnings_volatility,
    calibrate_post_earnings_volatility_surface,
)


class TestPostEarningsVolatility(unittest.TestCase):
    def test_straddle_jump_and_post_iv(self):
        spot = 100.0
        straddle = 10.0  # $10 straddle on $100 stock
        pre_iv = 0.60
        realized_21d = 0.25
        jump_pct, post_iv = calibrate_post_earnings_volatility(
            spot=spot,
            atm_straddle_price=straddle,
            pre_earnings_iv=pre_iv,
            realized_21d_vol=realized_21d,
            dte_days=7,
        )
        # Expected jump ~ 0.798 * 10 / 100 = 7.98%
        self.assertAlmostEqual(jump_pct, 7.98, places=1)
        # Post-earnings IV must be bounded below by realized 21-day volatility
        self.assertGreaterEqual(post_iv, realized_21d)

    def test_calibrate_post_earnings_volatility_surface(self):
        spot = 100.0
        straddle = 8.0
        pre_iv = 0.50
        realized_21d = 0.22
        res = calibrate_post_earnings_volatility_surface(
            spot=spot,
            atm_straddle_price=straddle,
            pre_earnings_iv=pre_iv,
            realized_21d_vol=realized_21d,
            dte_days=7,
        )
        self.assertIn("expected_jump_pct", res)
        self.assertIn("post_earnings_iv", res)
        self.assertIn("event_variance", res)
        self.assertIn("implied_move_dollars", res)
        self.assertIn("volatility_crush_pct", res)
        self.assertIn("volatility_crush_ratio", res)
        self.assertAlmostEqual(res["expected_jump_pct"], 6.38, places=1)
        self.assertGreaterEqual(res["post_earnings_iv"], realized_21d)

    def test_zero_spot_handled(self):
        jump_pct, post_iv = calibrate_post_earnings_volatility(0.0, 10.0, 0.60, 0.25)
        self.assertEqual(jump_pct, 0.0)
        self.assertGreaterEqual(post_iv, 0.25)

        res_zero = calibrate_post_earnings_volatility_surface(0.0, 10.0, 0.60, 0.25)
        self.assertEqual(res_zero["expected_jump_pct"], 0.0)
        self.assertGreaterEqual(res_zero["post_earnings_iv"], 0.25)


if __name__ == "__main__":
    unittest.main()

