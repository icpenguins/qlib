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


if __name__ == "__main__":
    unittest.main()

