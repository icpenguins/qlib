"""Unit tests for calculate_liquidity_impact_ratio."""

import unittest
from qlib.contrib.derivatives.liquidity_impact_ratio import calculate_liquidity_impact_ratio


class TestLiquidityImpactRatio(unittest.TestCase):
    def test_standard_computation(self):
        # 100,000 shares demand, ADTV = 1,000,000, depth = 10% (100,000 shares) -> LIR = 1.0
        lir = calculate_liquidity_impact_ratio(
            shares_demand=100_000.0,
            adtv_20=1_000_000.0,
            depth_factor=0.10,
        )
        self.assertAlmostEqual(lir, 1.0, places=4)

    def test_monotonicity(self):
        # Higher shares demand must produce higher LIR
        lir1 = calculate_liquidity_impact_ratio(50_000.0, 1_000_000.0, 0.10)
        lir2 = calculate_liquidity_impact_ratio(150_000.0, 1_000_000.0, 0.10)
        self.assertGreater(lir2, lir1)

    def test_zero_adtv_guard(self):
        lir_inf = calculate_liquidity_impact_ratio(50_000.0, 0.0, 0.10)
        self.assertEqual(lir_inf, float("inf"))
        lir_zero = calculate_liquidity_impact_ratio(0.0, 0.0, 0.10)
        self.assertEqual(lir_zero, 0.0)


if __name__ == "__main__":
    unittest.main()

