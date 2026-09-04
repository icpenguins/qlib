"""Institutional Test Battery: Almgren-Chriss Market Impact Model."""

import unittest
from qlib.contrib.microstructure.almgren_chriss_impact import AlmgrenChrissImpactModel


class TestAlmgrenChrissMarketImpact(unittest.TestCase):
    def setUp(self):
        self.model = AlmgrenChrissImpactModel(gamma_perm=0.10, eta_temp=0.15, alpha=0.50, fixed_bps=0.0005)
        self.adtv = 1_000_000.0  # 1M shares ADTV
        self.daily_vol = 0.02    # 2% daily vol

    def test_non_linear_impact_growth_above_10pct_adtv(self):
        # 1% ADTV (10,000 shares)
        res_1pct = self.model.calculate_impact(10_000.0, self.adtv, self.daily_vol)
        # 5% ADTV (50,000 shares)
        res_5pct = self.model.calculate_impact(50_000.0, self.adtv, self.daily_vol)
        # 15% ADTV (150,000 shares) - exceeds 10% liquidity threshold
        res_15pct = self.model.calculate_impact(150_000.0, self.adtv, self.daily_vol)

        # Invariant: Total impact cost must increase monotonically
        self.assertLess(res_1pct["total_cost_bps"], res_5pct["total_cost_bps"])
        self.assertLess(res_5pct["total_cost_bps"], res_15pct["total_cost_bps"])

        # Invariant: At 15% ADTV, impact cost should show severe non-linear degradation (> 15 bps)
        self.assertGreater(res_15pct["total_cost_bps"], 15.0)

    def test_zero_volume_baseline(self):
        res_zero = self.model.calculate_impact(0.0, self.adtv, self.daily_vol)
        # Should only reflect fixed exchange/clearing fee (5 bps)
        self.assertEqual(res_zero["total_cost_bps"], 5.0)
        self.assertEqual(res_zero["participation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

