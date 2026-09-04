"""Unit tests for compute_positive_gamma_squeeze_index."""

import unittest
from qlib.contrib.derivatives.positive_gamma_squeeze import compute_positive_gamma_squeeze_index


class TestPositiveGammaSqueeze(unittest.TestCase):
    def test_strong_positive_squeeze_alert(self):
        res = compute_positive_gamma_squeeze_index(
            lir_bull=2.5,
            sue_score=3.0,
            call_oi_otm=15_000,
            put_oi_atm=3_000,
            short_interest_pct=0.25,
        )
        self.assertGreaterEqual(res["gsi_plus_score"], 75.0)
        self.assertTrue(res["is_squeeze_alert"])
        self.assertEqual(res["action"], "AGGRESSIVE_BULL_GAMMA_SQUEEZE")

    def test_normal_drift_on_low_signals(self):
        res = compute_positive_gamma_squeeze_index(
            lir_bull=0.1,
            sue_score=0.2,
            call_oi_otm=1_000,
            put_oi_atm=1_000,
            short_interest_pct=0.02,
        )
        self.assertLess(res["gsi_plus_score"], 75.0)
        self.assertFalse(res["is_squeeze_alert"])
        self.assertEqual(res["action"], "NORMAL_DRIFT")

    def test_bounded_output(self):
        # Extreme inputs should remain strictly in [0.0, 100.0]
        res_high = compute_positive_gamma_squeeze_index(100.0, 50.0, 1e6, 1.0, 0.90)
        self.assertLessEqual(res_high["gsi_plus_score"], 100.0)
        res_low = compute_positive_gamma_squeeze_index(0.0, -50.0, 0.0, 1e6, 0.0)
        self.assertGreaterEqual(res_low["gsi_plus_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

