# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import unittest
import math
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.derivatives import (
    BlackScholesGreeks,
    DealerGammaEngine,
    VolatilitySurfaceFeatures,
    OptionsDataLoader,
    SyntheticOptionSurfaceGenerator,
)


class TestDerivativesGEX(unittest.TestCase):
    """
    Unit tests for standalone derivatives, Black-Scholes Greeks,
    Dealer Gamma Exposure (GEX), and volatility surface metrics.
    """

    def setUp(self):
        self.spot = 500.0
        self.vol = 0.25
        self.r = 0.045
        self.q = 0.0
        self.dte = 30
        self.t_years = self.dte / 365.0

    def test_black_scholes_greeks(self):
        """Test Black-Scholes Gamma and Delta analytical calculations."""
        # 1. ATM Option
        gamma_atm = BlackScholesGreeks.calc_gamma(
            spot=self.spot,
            strike=self.spot,
            t_years=self.t_years,
            sigma=self.vol,
            r=self.r,
            q=self.q,
        )
        self.assertGreater(gamma_atm, 0.0)

        # 2. OTM Option Gamma should be lower than ATM Gamma
        gamma_otm = BlackScholesGreeks.calc_gamma(
            spot=self.spot,
            strike=self.spot * 1.20,
            t_years=self.t_years,
            sigma=self.vol,
            r=self.r,
            q=self.q,
        )
        self.assertLess(gamma_otm, gamma_atm)

        # 3. Delta bounds
        delta_call = BlackScholesGreeks.calc_delta(
            spot=self.spot,
            strike=self.spot,
            t_years=self.t_years,
            sigma=self.vol,
            is_call=True,
            r=self.r,
            q=self.q,
        )
        delta_put = BlackScholesGreeks.calc_delta(
            spot=self.spot,
            strike=self.spot,
            t_years=self.t_years,
            sigma=self.vol,
            is_call=False,
            r=self.r,
            q=self.q,
        )
        self.assertTrue(0.40 <= delta_call <= 0.65)
        self.assertTrue(-0.65 <= delta_put <= -0.35)

    def test_synthetic_option_surface_generator(self):
        """Test deterministic synthetic option chain generation."""
        df_chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
            spot_price=self.spot,
            annual_vol=self.vol,
            dte_days=30,
            num_strikes=21,
        )
        self.assertIsInstance(df_chain, pd.DataFrame)
        self.assertEqual(len(df_chain), 42)  # 21 strikes * 2 (call + put)
        self.assertIn("strike", df_chain.columns)
        self.assertIn("option_type", df_chain.columns)
        self.assertIn("openInterest", df_chain.columns)
        self.assertIn("impliedVolatility", df_chain.columns)
        self.assertTrue((df_chain["openInterest"] >= 0).all())
        self.assertTrue((df_chain["impliedVolatility"] > 0).all())

    def test_dealer_gamma_engine(self):
        """Test Dealer Gamma Exposure, Gamma Flip Point, Walls, and Max Pain."""
        df_chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
            spot_price=self.spot,
            annual_vol=self.vol,
            dte_days=30,
        )
        engine = DealerGammaEngine(risk_free_rate=self.r, dividend_yield=self.q)
        gex_res = engine.compute_gex(df_chain, spot_price=self.spot)

        self.assertIn("net_gex_millions", gex_res)
        self.assertIn("call_gex_millions", gex_res)
        self.assertIn("put_gex_millions", gex_res)
        self.assertIn("gamma_flip_price", gex_res)
        self.assertIn("call_wall", gex_res)
        self.assertIn("put_wall", gex_res)
        self.assertIn("max_pain", gex_res)
        self.assertIn("regime", gex_res)
        self.assertIn("strike_profile", gex_res)

        # Check walls are positive numbers around spot and satisfy non-degeneracy
        self.assertGreater(gex_res["call_wall"], 0.0)
        self.assertGreater(gex_res["put_wall"], 0.0)
        self.assertGreater(gex_res["max_pain"], 0.0)
        self.assertGreater(gex_res["gamma_flip_price"], 0.0)
        self.assertGreater(len(gex_res["strike_profile"]), 0)

        # Structural non-degeneracy invariant: Put Wall < Call Wall
        self.assertGreater(gex_res["call_wall"], gex_res["put_wall"])
        self.assertGreaterEqual(gex_res["call_wall"], self.spot)
        self.assertLessEqual(gex_res["put_wall"], self.spot)

        # Call GEX is positive, Put GEX is negative
        self.assertGreater(gex_res["call_gex_millions"], 0.0)
        self.assertLess(gex_res["put_gex_millions"], 0.0)

    def test_mega_cap_gex_scale_and_strike_dispersion(self):
        """
        Verify institutional scale and strike dispersion for mega-cap tickers (e.g. MSFT):
        1. Open interest is scaled to hundreds of thousands / millions of contracts.
        2. Strikes use standard discrete increments ($5.00 for $250+).
        3. Strike collapse is prevented: Put Wall < Spot < Call Wall, Put Wall <= Max Pain <= Call Wall.
        4. Net GEX reflects institutional multi-million/hundred-million dollar magnitudes.
        """
        spot_msft = 505.0
        adtv_msft = 22_000_000.0

        df_chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
            spot_price=spot_msft,
            annual_vol=0.22,
            dte_days=30,
            num_strikes=25,
            symbol="MSFT",
            adtv=adtv_msft,
        )

        # 1. Strike increments must be $5.00 multiples for $500+ stock
        strikes = df_chain["strike"].unique()
        for k in strikes:
            self.assertEqual(k % 5.0, 0.0, f"Strike ${k} is not a multiple of 5.0")

        # 2. Mega-cap open interest scale
        call_oi = df_chain[df_chain["option_type"] == "call"]["openInterest"].sum()
        put_oi = df_chain[df_chain["option_type"] == "put"]["openInterest"].sum()
        total_oi = call_oi + put_oi

        self.assertGreater(call_oi, 500_000, f"Call OI {call_oi} too small for mega-cap")
        self.assertGreater(put_oi, 500_000, f"Put OI {put_oi} too small for mega-cap")
        self.assertGreater(total_oi, 1_000_000, f"Total OI {total_oi} too small for mega-cap")

        # 3. GEX calculation and non-degeneracy
        engine = DealerGammaEngine(risk_free_rate=0.045)
        gex_res = engine.compute_gex(df_chain, spot_price=spot_msft)

        call_wall = gex_res["call_wall"]
        put_wall = gex_res["put_wall"]
        max_pain = gex_res["max_pain"]

        self.assertGreater(call_wall, spot_msft, f"Call Wall ${call_wall} should sit above spot ${spot_msft}")
        self.assertLess(put_wall, spot_msft, f"Put Wall ${put_wall} should sit below spot ${spot_msft}")
        self.assertNotEqual(call_wall, put_wall, "Call Wall and Put Wall must not collapse to identical strike")
        self.assertTrue(put_wall <= max_pain <= call_wall, f"Max Pain ${max_pain} should sit between Put Wall ${put_wall} and Call Wall ${call_wall}")

        # 4. Institutional Net GEX magnitude (> $20M per 1% move)
        self.assertGreater(abs(gex_res["net_gex_millions"]), 20.0, "Net GEX is toy book scale")

    def test_volatility_surface_features(self):
        """Test 25-Delta Risk Reversal skew and Variance Risk Premium (VRP)."""
        df_chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
            spot_price=self.spot,
            annual_vol=self.vol,
            dte_days=30,
        )
        vol_features = VolatilitySurfaceFeatures.compute_features(
            df_chain,
            spot_price=self.spot,
            realized_vol_21d=0.20,
        )

        self.assertIn("atm_iv_pct", vol_features)
        self.assertIn("risk_reversal_25d_pct", vol_features)
        self.assertIn("vrp_pct", vol_features)
        self.assertIn("skew_regime", vol_features)

        # Realized vol is 20%, ATM IV is ~25%, so VRP should be positive (~5%)
        self.assertGreater(vol_features["vrp_pct"], 0.0)
        # Standard equity skew has negative risk reversal (put IV > call IV)
        self.assertLess(vol_features["risk_reversal_25d_pct"], 0.0)

    def test_options_data_loader_fallback(self):
        """Test that OptionsDataLoader falls back cleanly to synthetic generation when file is absent."""
        loader = OptionsDataLoader(data_dir=None)
        df_chain, is_synthetic = loader.get_options_chain(
            symbol="TEST",
            spot_price=150.0,
            annual_vol=0.30,
            auto_download=False,
        )
        self.assertTrue(is_synthetic)
        self.assertFalse(df_chain.empty)
        self.assertIn("strike", df_chain.columns)


if __name__ == "__main__":
    unittest.main()
