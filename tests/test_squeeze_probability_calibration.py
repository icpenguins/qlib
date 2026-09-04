"""Unit tests for squeeze_probability_calibration."""

import unittest
import numpy as np
from qlib.contrib.derivatives.squeeze_probability_calibration import (
    generate_dual_squeeze_label,
    fit_platt_calibrator,
    calibrate_squeeze_probability,
)


class TestSqueezeProbabilityCalibration(unittest.TestCase):
    def test_dual_ground_truth_label_sign_agreement(self):
        # Case A: Jump magnitude satisfied AND sign agreement -> y = 1
        y_true = generate_dual_squeeze_label(
            ar_open=0.06,              # +6% gap at open
            dealer_shares_demand=500_000.0,  # Dealers forced to buy
            daily_vol=0.02,
            threshold_mult=1.5,        # 1.5 * 0.02 = 3% hurdle
        )
        self.assertEqual(y_true, 1)

        # Case B: Jump magnitude satisfied BUT sign DISAGREES -> y = 0
        # E.g. stock gapped up 6%, but dealer demand was negative (dealers forced to SELL into gap)
        y_false_sign = generate_dual_squeeze_label(
            ar_open=0.06,
            dealer_shares_demand=-500_000.0,
            daily_vol=0.02,
            threshold_mult=1.5,
        )
        self.assertEqual(y_false_sign, 0)

        # Case C: Sign agrees but magnitude insufficient -> y = 0
        y_insufficient = generate_dual_squeeze_label(
            ar_open=0.01,
            dealer_shares_demand=500_000.0,
            daily_vol=0.02,
            threshold_mult=1.5,
        )
        self.assertEqual(y_insufficient, 0)

    def test_calibrate_squeeze_probability_monotonicity(self):
        # Higher GSI scores must produce higher calibrated probabilities
        p_low = calibrate_squeeze_probability(20.0)
        p_mid = calibrate_squeeze_probability(50.0)
        p_high = calibrate_squeeze_probability(85.0)

        self.assertLess(p_low, p_mid)
        self.assertLess(p_mid, p_high)
        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)

    def test_fit_platt_calibrator(self):
        # Generate synthetic training set: higher scores have more 1s
        scores = np.array([10.0, 25.0, 35.0, 45.0, 60.0, 75.0, 85.0, 95.0] * 3)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
        a, b = fit_platt_calibrator(scores, labels)
        self.assertLess(a, 0.0)  # Negative slope ensures P increases with score
        self.assertGreater(b, 0.0)


if __name__ == "__main__":
    unittest.main()

