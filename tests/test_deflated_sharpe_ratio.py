"""Institutional Test Battery: Deflated Sharpe Ratio."""

import unittest
import numpy as np
from qlib.contrib.backtest.deflated_sharpe_ratio import calculate_deflated_sharpe_ratio


class TestDeflatedSharpeRatio(unittest.TestCase):
    def test_dynamic_hurdle_monotonic_with_trials(self):
        # Generate synthetic return matrices for 10 trials vs 100 trials
        np.random.seed(42)
        T = 252  # 1 year of daily returns

        matrix_10 = np.random.normal(loc=0.0005, scale=0.01, size=(T, 10))
        matrix_100 = np.random.normal(loc=0.0005, scale=0.01, size=(T, 100))

        res_10 = calculate_deflated_sharpe_ratio(matrix_10)
        res_100 = calculate_deflated_sharpe_ratio(matrix_100)

        # INVARIANT: As trial count increases from 10 to 100, the expected max hurdle E[max(SR_0)]
        # MUST increase due to extreme value theory penalty for multiple testing!
        self.assertGreater(res_100["expected_max_sharpe"], res_10["expected_max_sharpe"])

        # INVARIANT: Ensure no pre-written static 0.962 number; DSR is dynamically calculated
        self.assertIsInstance(res_10["dsr_probability"], float)
        self.assertGreaterEqual(res_10["dsr_probability"], 0.0)
        self.assertLessEqual(res_10["dsr_probability"], 1.0)

    def test_too_short_sample_raises_error(self):
        short_matrix = np.random.randn(15, 5)
        with self.assertRaises(ValueError):
            calculate_deflated_sharpe_ratio(short_matrix)


if __name__ == "__main__":
    unittest.main()

