"""Unit tests for calculate_forced_dealer_hedging_demand."""

import unittest
import pandas as pd
import numpy as np
from qlib.contrib.derivatives.forced_dealer_hedging import calculate_forced_dealer_hedging_demand


class TestForcedDealerHedging(unittest.TestCase):
    def setUp(self):
        self.spot = 100.0
        self.adtv = 1_000_000.0
        # Synthetic small chain
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        data = []
        for k in strikes:
            data.append({
                "strike": k,
                "option_type": "call",
                "openInterest": 500,
                "impliedVolatility": 0.40,
                "dte": 14,
                "delta_call": 0.5,
                "delta_put": -0.5,
            })
            data.append({
                "strike": k,
                "option_type": "put",
                "openInterest": 500,
                "impliedVolatility": 0.40,
                "dte": 14,
                "delta_call": 0.5,
                "delta_put": -0.5,
            })
        self.df_chain = pd.DataFrame(data)

    def test_scenarios_returned(self):
        scenarios = [-0.10, -0.05, 0.05, 0.10]
        res = calculate_forced_dealer_hedging_demand(
            spot=self.spot,
            df_chain=self.df_chain,
            adtv_20=self.adtv,
            jump_scenarios=scenarios,
        )
        self.assertEqual(len(res), 4)
        for s in scenarios:
            self.assertIn(s, res)
            self.assertIn("shares_demand", res[s])
            self.assertIn("lir", res[s])
            self.assertIn("dollar_demand", res[s])

    def test_empty_chain_returns_zeros(self):
        res = calculate_forced_dealer_hedging_demand(
            spot=self.spot,
            df_chain=pd.DataFrame(),
            adtv_20=self.adtv,
            jump_scenarios=[0.05],
        )
        self.assertEqual(res[0.05]["shares_demand"], 0.0)
        self.assertEqual(res[0.05]["lir"], 0.0)
        self.assertTrue(res[0.05]["invariant_ok"])

    def test_shares_demand_never_exceeds_physical_oi_ceiling(self):
        """
        Regression test for the FIX adversarial audit finding: aggregate dealer
        share demand must never exceed 100 shares/contract x total chain open
        interest, since each leg's delta change is bounded to at most 1.0 (see
        BlackScholesGreeks.calc_delta). A prior report showed a demand more than
        4x this physical ceiling, caused by a data-source inconsistency rather
        than a bug in this formula -- this test guards the formula's own
        invariant regardless of where the chain came from.
        """
        scenarios = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
        res = calculate_forced_dealer_hedging_demand(
            spot=self.spot,
            df_chain=self.df_chain,
            adtv_20=self.adtv,
            jump_scenarios=scenarios,
        )
        total_oi = self.df_chain["openInterest"].sum()
        max_physical = 100.0 * total_oi
        for dS in scenarios:
            self.assertLessEqual(abs(res[dS]["shares_demand"]), max_physical + 1e-6)
            self.assertTrue(res[dS]["invariant_ok"])
            self.assertAlmostEqual(res[dS]["max_physical_shares_demand"], max_physical, places=2)

    def test_invariant_flag_present_and_consistent_with_ceiling_formula(self):
        """
        Pins the ceiling formula's shape (100 shares/contract x total OI) so a
        future change to it is caught, and confirms `invariant_ok`/
        `max_physical_shares_demand` are always present on every scenario result.
        """
        df_chain = pd.DataFrame([
            {
                "strike": 100.0, "option_type": "call", "openInterest": 1.0,
                "impliedVolatility": 0.40, "dte": 14, "delta_call": 0.0, "delta_put": 0.0,
            },
        ])
        res = calculate_forced_dealer_hedging_demand(
            spot=100.0, df_chain=df_chain, adtv_20=self.adtv, jump_scenarios=[0.10],
        )
        self.assertIn("invariant_ok", res[0.10])
        self.assertAlmostEqual(res[0.10]["max_physical_shares_demand"], 100.0, places=2)


if __name__ == "__main__":
    unittest.main()

