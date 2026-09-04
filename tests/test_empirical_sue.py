"""Unit tests for calculate_empirical_sue."""

import unittest
from qlib.contrib.events.empirical_sue import calculate_empirical_sue


class TestEmpiricalSUE(unittest.TestCase):
    def test_positive_earnings_beat(self):
        # Actual EPS $2.20, Consensus $2.00, Historical error std dev ~ $0.10
        hist_errors = [0.08, 0.12, -0.05, 0.15, 0.09, 0.11]
        sue = calculate_empirical_sue(
            actual_eps=2.20,
            consensus_eps=2.00,
            historical_forecast_errors=hist_errors,
        )
        self.assertGreater(sue, 1.5)

    def test_negative_earnings_miss(self):
        hist_errors = [0.05, -0.02, 0.04, -0.01]
        sue = calculate_empirical_sue(
            actual_eps=1.80,
            consensus_eps=2.00,
            historical_forecast_errors=hist_errors,
        )
        self.assertLess(sue, -2.0)

    def test_zero_variance_floor(self):
        # All historical errors identical -> std = 0.0, floor prevents division by zero
        hist_errors = [0.10, 0.10, 0.10, 0.10]
        sue = calculate_empirical_sue(
            actual_eps=2.10,
            consensus_eps=2.00,
            historical_forecast_errors=hist_errors,
            min_std_floor=0.02,
        )
        self.assertAlmostEqual(sue, 5.0, places=1)


if __name__ == "__main__":
    unittest.main()

