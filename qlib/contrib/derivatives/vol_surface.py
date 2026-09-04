# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Volatility Surface & Skew Features
==================================
Computes institutional derivatives metrics:
1. 25-Delta Risk Reversal (RR25): Skew measure reflecting demand for downside crash protection vs. upside calls.
2. Variance Risk Premium (VRP): Spread between 30-day Implied Volatility and 21-day Realized Volatility.
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from .gex import BlackScholesGreeks


class VolatilitySurfaceFeatures:
    """
    Extracts structural volatility surface and skew features from an option chain.
    """

    @staticmethod
    def compute_features(
        df_options: pd.DataFrame,
        spot_price: float,
        realized_vol_21d: Optional[float] = None,
    ) -> Dict[str, Union[float, str]]:
        """
        Compute 25-Delta Risk Reversal, ATM 30d Implied Volatility, and Variance Risk Premium (VRP).

        Parameters
        ----------
        df_options : pd.DataFrame
            Option chain with 'strike', 'option_type', 'impliedVolatility', 'dte'.
        spot_price : float
            Current underlying spot price.
        realized_vol_21d : Optional[float]
            21-day annualized realized volatility (e.g. 0.30 for 30%).

        Returns
        -------
        Dict[str, Union[float, str]]
            Summary dictionary with 'atm_iv_pct', 'risk_reversal_25d_pct', 'vrp_pct', and 'skew_regime'.
        """
        if df_options.empty:
            return {
                "atm_iv_pct": 25.0,
                "risk_reversal_25d_pct": -2.0,
                "vrp_pct": 0.0,
                "skew_regime": "Neutral Skew",
            }

        spot = float(spot_price)
        df = df_options.copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.25)
        df["dte"] = pd.to_numeric(df.get("dte", 30), errors="coerce").fillna(30)
        df = df.dropna(subset=["strike"])

        is_call = df["option_type"].str.lower().str.startswith("c")

        # 1. ATM 30-Day Implied Volatility
        atm_dist = (df["strike"] - spot).abs()
        atm_idx = atm_dist.idxmin()
        atm_iv = float(df.loc[atm_idx, "impliedVolatility"]) * 100.0

        # 2. 25-Delta Risk Reversal
        # Calculate Delta for all contracts
        t_years = np.maximum(1.0 / 365.0, df["dte"].values / 365.0)
        deltas = BlackScholesGreeks.calc_delta(
            spot=spot,
            strike=df["strike"].values,
            t_years=t_years,
            sigma=df["impliedVolatility"].values,
            is_call=is_call.values,
        )
        df["delta"] = deltas

        # 25-Delta Call: Delta closest to +0.25
        calls = df[is_call]
        if not calls.empty:
            call_25_idx = (calls["delta"] - 0.25).abs().idxmin()
            call_25_iv = float(calls.loc[call_25_idx, "impliedVolatility"]) * 100.0
        else:
            call_25_iv = atm_iv * 0.95

        # 25-Delta Put: Delta closest to -0.25
        puts = df[~is_call]
        if not puts.empty:
            put_25_idx = (puts["delta"] - (-0.25)).abs().idxmin()
            put_25_iv = float(puts.loc[put_25_idx, "impliedVolatility"]) * 100.0
        else:
            put_25_iv = atm_iv * 1.05

        # Risk Reversal: 25d Call IV - 25d Put IV (typically negative in equities)
        risk_reversal_25d = round(call_25_iv - put_25_iv, 2)

        # 3. Variance Risk Premium (VRP) = ATM IV - Realized Vol
        vrp = 0.0
        if realized_vol_21d is not None and realized_vol_21d > 0:
            realized_vol_pct = realized_vol_21d * 100.0 if realized_vol_21d < 2.0 else realized_vol_21d
            vrp = round(atm_iv - realized_vol_pct, 2)

        # Skew Regime
        if risk_reversal_25d < -4.5:
            skew_regime = "Heavy Put Skew (Downside Hedging Demand)"
        elif risk_reversal_25d > 1.0:
            skew_regime = "Inverted Skew (Aggressive Call Speculation)"
        else:
            skew_regime = "Normal Equity Skew"

        return {
            "atm_iv_pct": round(atm_iv, 1),
            "call_25d_iv_pct": round(call_25_iv, 1),
            "put_25d_iv_pct": round(put_25_iv, 1),
            "risk_reversal_25d_pct": risk_reversal_25d,
            "vrp_pct": vrp,
            "skew_regime": skew_regime,
        }

    @classmethod
    def compute_surface_metrics(
        cls,
        options_df: pd.DataFrame,
        spot: float,
        realized_vol_21d: Optional[float] = None,
        r: float = 0.045,
    ) -> Dict[str, Union[float, str]]:
        """
        Convenience alias returning formatted surface metrics.
        """
        res = cls.compute_features(options_df, spot_price=spot, realized_vol_21d=realized_vol_21d)
        res["iv_30d_atm"] = res["atm_iv_pct"] / 100.0
        res["rr25_skew"] = res["risk_reversal_25d_pct"]
        return res

