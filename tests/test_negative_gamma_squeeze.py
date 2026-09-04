"""Unit tests for compute_negative_gamma_squeeze_index."""

import unittest
from qlib.contrib.derivatives.negative_gamma_squeeze import compute_negative_gamma_squeeze_index


class TestNegativeGammaSqueeze(unittest.TestCase):
    def test_liquidation_cascade_alert(self):
        # Bearish LIR high, big earnings miss, spot below gamma flip, liquidity void active
        res = compute_negative_gamma_squeeze_index(
            lir_bear=3.0,
            sue_score=-3.5,
            spot=92.0,
            gamma_flip_price=100.0,
            in_liquidity_void=True,
        )
        self.assertGreaterEqual(res["gsi_minus_score"], 75.0)
        self.assertTrue(res["is_cascade_alert"])
        self.assertEqual(res["action"], "LIQUIDATION_CASCADE_ALERT")

    def test_normal_pullback_above_flip(self):
        res = compute_negative_gamma_squeeze_index(
            lir_bear=0.2,
            sue_score=-0.5,
            spot=105.0,
            gamma_flip_price=100.0,
            in_liquidity_void=False,
        )
        self.assertLess(res["gsi_minus_score"], 75.0)
        self.assertFalse(res["is_cascade_alert"])
        self.assertEqual(res["action"], "NORMAL_PULLBACK")


if __name__ == "__main__":
    unittest.main()

