"""Unit tests for calculate_historical_iv_crush."""

import unittest
from qlib.contrib.derivatives.historical_iv_crush import calculate_historical_iv_crush


class TestHistoricalIVCrush(unittest.TestCase):
    def test_empirical_winsorized_median(self):
        # 6 observed quarters of (pre_iv, post_iv)
        # Crushes: 40%, 45%, 50%, 55%, 60%, and an outlier 90% (e.g. shock)
        pairs = [
            (0.80, 0.48),  # 40%
            (0.80, 0.44),  # 45%
            (0.80, 0.40),  # 50%
            (0.80, 0.36),  # 55%
            (0.80, 0.32),  # 60%
            (0.80, 0.08),  # 90% outlier
        ]
        res = calculate_historical_iv_crush(observed_iv_pairs=pairs, min_observed_pairs=4)
        self.assertTrue(res["is_empirical"])
        self.assertEqual(res["crush_source"], "empirical_winsorized_median")
        self.assertEqual(res["observed_count"], 6)
        # Winsorized median should be centered around 50%-55%, NOT dominated by the 90% outlier
        self.assertLess(res["iv_crush_ratio"], 0.65)
        self.assertGreater(res["iv_crush_ratio"], 0.45)

    def test_insufficient_observed_pairs_fallback_to_term_structure(self):
        # Only 2 quarters (less than minimum 4)
        pairs = [(0.70, 0.45), (0.75, 0.40)]
        # month1_iv = 0.60, month2_iv = 0.36 -> slope = 1 - 0.36/0.60 = 40%
        res = calculate_historical_iv_crush(
            observed_iv_pairs=pairs,
            month1_iv=0.60,
            month2_iv=0.36,
            min_observed_pairs=4,
        )
        self.assertFalse(res["is_empirical"])
        self.assertEqual(res["crush_source"], "term_structure_proxy")
        self.assertAlmostEqual(res["iv_crush_ratio"], 0.40, places=2)

    def test_conservative_default_when_no_data(self):
        res = calculate_historical_iv_crush(observed_iv_pairs=None, month1_iv=None, month2_iv=None)
        self.assertFalse(res["is_empirical"])
        self.assertEqual(res["crush_source"], "conservative_default")
        self.assertEqual(res["iv_crush_ratio"], 0.40)


if __name__ == "__main__":
    unittest.main()

