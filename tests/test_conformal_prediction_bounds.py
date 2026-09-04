"""Unit tests for calculate_conformal_bounds."""

import unittest
from qlib.contrib.derivatives.conformal_prediction_bounds import calculate_conformal_bounds


class TestConformalPredictionBounds(unittest.TestCase):
    def test_bounds_contain_point_estimate(self):
        p = 0.75
        p_low, p_high = calculate_conformal_bounds(p, confidence_level=0.90, residual_quantile=0.08)
        self.assertLessEqual(p_low, p)
        self.assertGreaterEqual(p_high, p)

    def test_boundary_clipping(self):
        # Near 1.0
        _, p_high = calculate_conformal_bounds(0.98, confidence_level=0.90, residual_quantile=0.08)
        self.assertLessEqual(p_high, 1.0)
        # Near 0.0
        p_low, _ = calculate_conformal_bounds(0.02, confidence_level=0.90, residual_quantile=0.08)
        self.assertGreaterEqual(p_low, 0.0)


if __name__ == "__main__":
    unittest.main()

