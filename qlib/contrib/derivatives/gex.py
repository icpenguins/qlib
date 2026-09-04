# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Dealer Gamma Exposure (GEX) & Greeks Engine
===========================================
Pure NumPy vectorized implementation of Black-Scholes option Greeks,
dealer Net Gamma Exposure (GEX), the Gamma Flip Point (volatility trigger),
Call & Put Gamma Walls, and Max Pain strike analysis.

Mathematical Foundation:
------------------------
For an underlying price S, strike K, time-to-maturity tau (in years),
implied volatility sigma, risk-free rate r, and dividend yield q:

    d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * tau] / (sigma * sqrt(tau))
    Gamma = [exp(-q * tau) / (S * sigma * sqrt(tau))] * phi(d1)

    where phi(x) = (1 / sqrt(2*pi)) * exp(-0.5 * x^2)

Dealer Gamma Assumptions:
-------------------------
In institutional options microstructure, market makers (dealers) are generally net
short calls (bought by directional investors/hedgers) and net short puts (bought by downside hedgers).
- Long Calls bought by customers -> Dealers short calls -> Dealers LONG GAMMA (+GEX)
- Long Puts bought by customers -> Dealers short puts -> Dealers SHORT GAMMA (-GEX)

Dollar Gamma per 1% move:
    GEX_call(K) = +OI_call(K) * 100 * S * Gamma(K) * S * 0.01
    GEX_put(K)  = -OI_put(K)  * 100 * S * Gamma(K) * S * 0.01
    Net_GEX(K)  = GEX_call(K) + GEX_put(K)
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BlackScholesGreeks:
    """
    Vectorized Black-Scholes analytical Greeks engine with zero external dependencies.
    """

    @staticmethod
    def calc_gamma(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        t_years: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        r: float = 0.045,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Calculate option Gamma (same for both Call and Put).
        """
        S = np.maximum(np.asarray(spot, dtype=np.float64), 1e-6)
        K = np.maximum(np.asarray(strike, dtype=np.float64), 1e-6)
        tau = np.maximum(np.asarray(t_years, dtype=np.float64), 1e-5)
        vol = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-4)

        denom = vol * np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * (vol ** 2)) * tau) / denom
        phi_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (d1 ** 2))

        gamma = (np.exp(-q * tau) / (S * denom)) * phi_d1
        return gamma

    @staticmethod
    def calc_delta(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        t_years: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        is_call: Union[bool, np.ndarray],
        r: float = 0.045,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Calculate option Delta.
        """
        S = np.maximum(np.asarray(spot, dtype=np.float64), 1e-6)
        K = np.maximum(np.asarray(strike, dtype=np.float64), 1e-6)
        tau = np.maximum(np.asarray(t_years, dtype=np.float64), 1e-5)
        vol = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-4)

        denom = vol * np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * (vol ** 2)) * tau) / denom
        
        # Exact normal CDF using vectorized math.erf
        c_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
        delta_call = np.exp(-q * tau) * c_cdf
        delta_put = np.exp(-q * tau) * (c_cdf - 1.0)

        is_call_arr = np.asarray(is_call, dtype=bool)
        return np.where(is_call_arr, delta_call, delta_put)


class DealerGammaEngine:
    """
    Computes comprehensive Dealer Gamma Exposure (GEX) metrics across strike prices:
    - Net Dollar GEX ($ Millions per 1% move)
    - Gamma Flip Point (Zero-Gamma boundary S* where market flips regime)
    - Major Call Gamma Wall (resistance / magnet pin)
    - Major Put Gamma Wall (support floor / breakdown accelerator)
    - Max Pain Strike Price
    - GEX Regime Classification
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        dividend_yield: float = 0.0,
    ):
        self.r = float(risk_free_rate)
        self.q = float(dividend_yield)

    def compute_gex(
        self,
        df_options: pd.DataFrame,
        spot_price: float,
    ) -> Dict[str, Any]:
        """
        Compute full Dealer Gamma Exposure profile from an option chain DataFrame.

        Parameters
        ----------
        df_options : pd.DataFrame
            DataFrame with columns:
            - 'strike' (float)
            - 'option_type' (str: 'call' or 'put')
            - 'openInterest' (int/float)
            - 'impliedVolatility' (float)
            - 'dte' (int/float: days to expiration)
        spot_price : float
            Current underlying stock spot price.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'net_gex_millions': Total Net Dollar GEX ($M per 1% move)
            - 'call_gex_millions': Total Call GEX ($M)
            - 'put_gex_millions': Total Put GEX ($M)
            - 'gamma_flip_price': Underlying price where Net GEX = 0
            - 'dist_to_flip_pct': Percentage distance from spot to Gamma Flip
            - 'call_wall': Major Call Gamma Wall strike ($)
            - 'put_wall': Major Put Gamma Wall strike ($)
            - 'absolute_wall': Major Absolute Gamma Wall strike ($)
            - 'max_pain': Max Pain strike ($)
            - 'regime': '+GEX Mean-Reverting Stabilizer' vs '-GEX Volatility Accelerant'
            - 'regime_state': 1 (+GEX) or -1 (-GEX)
            - 'put_call_oi_ratio': Put OI / Call OI
            - 'strike_profile': List of dicts for top strikes (K, call_gex, put_gex, net_gex)
        """
        if df_options.empty:
            raise ValueError("Option chain DataFrame is empty.")

        spot = float(spot_price)
        df = df_options.copy()
        
        # Standardize columns
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0.25), errors="coerce").fillna(0.25)
        df["dte"] = pd.to_numeric(df.get("dte", 30), errors="coerce").fillna(30)
        df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)

        # Time in years
        t_years = np.maximum(1.0 / 365.0, df["dte"].values / 365.0)
        sigmas = np.maximum(0.05, df["impliedVolatility"].values)
        strikes = df["strike"].values

        # Vectorized Gamma
        gammas = BlackScholesGreeks.calc_gamma(
            spot=spot,
            strike=strikes,
            t_years=t_years,
            sigma=sigmas,
            r=self.r,
            q=self.q,
        )
        df["gamma"] = gammas

        # Dollar Gamma per 1% underlying move:
        # GEX($) = OI * 100 * Spot * Gamma * (0.01 * Spot)
        dollar_gamma_factor = 100.0 * spot * (0.01 * spot)
        is_call = df["option_type"].str.lower().str.startswith("c").values

        call_gex_raw = np.where(is_call, df["openInterest"].values * gammas * dollar_gamma_factor, 0.0)
        put_gex_raw = np.where(~is_call, -df["openInterest"].values * gammas * dollar_gamma_factor, 0.0)

        df["call_gex_dollar"] = call_gex_raw
        df["put_gex_dollar"] = put_gex_raw
        df["net_gex_dollar"] = call_gex_raw + put_gex_raw

        # Aggregate across strikes
        strike_groups = df.groupby("strike").agg({
            "call_gex_dollar": "sum",
            "put_gex_dollar": "sum",
            "net_gex_dollar": "sum",
            "openInterest": "sum",
        }).reset_index()

        call_oi_total = df[is_call]["openInterest"].sum()
        put_oi_total = df[~is_call]["openInterest"].sum()
        pc_ratio = round(float(put_oi_total) / max(1.0, float(call_oi_total)), 2)

        total_call_gex_m = float(df["call_gex_dollar"].sum() / 1e6)
        total_put_gex_m = float(df["put_gex_dollar"].sum() / 1e6)
        total_net_gex_m = float(df["net_gex_dollar"].sum() / 1e6)

        # 1. Major Gamma Walls
        # Call Wall: Strike with maximum Call GEX
        call_wall_row = strike_groups.loc[strike_groups["call_gex_dollar"].idxmax()]
        call_wall = float(call_wall_row["strike"])

        # Put Wall: Strike with maximum negative Put GEX (most negative)
        put_wall_row = strike_groups.loc[strike_groups["put_gex_dollar"].idxmin()]
        put_wall = float(put_wall_row["strike"])

        # Absolute Wall: Strike with greatest absolute Net GEX
        abs_wall_row = strike_groups.loc[strike_groups["net_gex_dollar"].abs().idxmax()]
        absolute_wall = float(abs_wall_row["strike"])

        # 2. Gamma Flip Point (Zero-Gamma Threshold S*)
        gamma_flip = self._solve_gamma_flip(df, spot)
        dist_to_flip_pct = round(((spot - gamma_flip) / spot) * 100.0, 2)

        # 3. Max Pain Strike Price
        max_pain = self._calculate_max_pain(df)

        # 4. GEX Regime Classification
        # If spot > Gamma Flip and Net GEX > 0: Positive Gamma (Mean Reverting)
        # If spot < Gamma Flip or Net GEX < 0: Negative Gamma (Volatility Accelerant)
        if abs(dist_to_flip_pct) <= 1.2:
            regime = "Gamma Flip Boundary (Inflection Alert)"
            regime_state = 0
            regime_color = "amber"
            badge_class = "bg-amber-500/10 text-amber-400 border-amber-500/30"
            description = (
                f"Spot (${spot:.2f}) is hovering within {abs(dist_to_flip_pct):.1f}% of the Gamma Flip level (${gamma_flip:.2f}). "
                "Expect heightened sensitivity and an imminent transition between volatility compression and expansion."
            )
        elif spot >= gamma_flip and total_net_gex_m >= 0:
            regime = "+GEX Regime (Mean-Reverting Stabilizer)"
            regime_state = 1
            regime_color = "emerald"
            badge_class = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
            description = (
                f"Dealers are net long gamma (+${total_net_gex_m:.1f}M/1%). Market makers counter-cyclically buy dips "
                f"and sell rallies, dampening realized volatility and pinning price between the Put Wall (${put_wall:.2f}) and Call Wall (${call_wall:.2f})."
            )
        else:
            regime = "-GEX Regime (Volatility Accelerant / Squeeze)"
            regime_state = -1
            regime_color = "rose"
            badge_class = "bg-rose-500/10 text-rose-400 border-rose-500/30"
            description = (
                f"Dealers are net short gamma (${total_net_gex_m:.1f}M/1%). Market makers pro-cyclically sell declines "
                f"and buy rallies to delta-hedge, triggering directional volatility cascades and flash breakouts."
            )

        # 5. Strike Profile Window (+/- 15% around spot)
        profile_df = strike_groups[
            (strike_groups["strike"] >= spot * 0.85) & 
            (strike_groups["strike"] <= spot * 1.15)
        ].copy()

        strike_profile = []
        for _, row in profile_df.iterrows():
            strike_profile.append({
                "strike": round(float(row["strike"]), 2),
                "call_gex_m": round(float(row["call_gex_dollar"]) / 1e6, 2),
                "put_gex_m": round(float(row["put_gex_dollar"]) / 1e6, 2),
                "net_gex_m": round(float(row["net_gex_dollar"]) / 1e6, 2),
                "open_interest": int(row["openInterest"]),
                "is_call_wall": bool(round(float(row["strike"]), 2) == round(call_wall, 2)),
                "is_put_wall": bool(round(float(row["strike"]), 2) == round(put_wall, 2)),
                "is_max_pain": bool(round(float(row["strike"]), 2) == round(max_pain, 2)),
            })

        return {
            "spot_price": spot,
            "net_gex_millions": round(total_net_gex_m, 2),
            "net_gex_dollar_per_1pct": round(total_net_gex_m * 1e6, 2),
            "call_gex_millions": round(total_call_gex_m, 2),
            "put_gex_millions": round(total_put_gex_m, 2),
            "gamma_flip_price": round(gamma_flip, 2),
            "dist_to_flip_pct": dist_to_flip_pct,
            "call_wall": round(call_wall, 2),
            "call_wall_strike": round(call_wall, 2),
            "put_wall": round(put_wall, 2),
            "put_wall_strike": round(put_wall, 2),
            "absolute_wall": round(absolute_wall, 2),
            "max_pain": round(max_pain, 2),
            "max_pain_strike": round(max_pain, 2),
            "regime": regime,
            "regime_state": regime_state,
            "regime_color": regime_color,
            "badge_class": badge_class,
            "description": description,
            "put_call_oi_ratio": pc_ratio,
            "total_call_oi": int(call_oi_total),
            "total_put_oi": int(put_oi_total),
            "strike_profile": strike_profile,
        }

    def _solve_gamma_flip(self, df: pd.DataFrame, spot: float) -> float:
        """
        Solve for the spot price S* where Total Net GEX(S*) = 0.
        Uses a dense grid search followed by linear interpolation.
        """
        grid_spots = np.linspace(spot * 0.65, spot * 1.35, 120)
        is_call = df["option_type"].str.lower().str.startswith("c").values
        strikes = df["strike"].values
        t_years = np.maximum(1.0 / 365.0, df["dte"].values / 365.0)
        sigmas = np.maximum(0.05, df["impliedVolatility"].values)
        ois = df["openInterest"].values

        net_gex_curve = []
        for s_eval in grid_spots:
            gammas = BlackScholesGreeks.calc_gamma(s_eval, strikes, t_years, sigmas, self.r, self.q)
            factor = 100.0 * s_eval * (0.01 * s_eval)
            call_g = np.where(is_call, ois * gammas * factor, 0.0)
            put_g = np.where(~is_call, -ois * gammas * factor, 0.0)
            net_gex_curve.append(np.sum(call_g + put_g))

        net_gex_curve = np.array(net_gex_curve)

        # Detect sign changes
        zero_crossings = np.where(np.diff(np.sign(net_gex_curve)))[0]
        if len(zero_crossings) > 0:
            # Pick crossing closest to current spot price
            best_idx = zero_crossings[np.argmin(np.abs(grid_spots[zero_crossings] - spot))]
            s1, s2 = grid_spots[best_idx], grid_spots[best_idx + 1]
            g1, g2 = net_gex_curve[best_idx], net_gex_curve[best_idx + 1]
            if g2 != g1:
                flip_price = s1 - g1 * ((s2 - s1) / (g2 - g1))
                return float(flip_price)

        # If no crossing within window, return lowest/highest strike or boundary
        if net_gex_curve[0] < 0:
            return float(grid_spots[0])
        return float(grid_spots[-1])

    def _calculate_max_pain(self, df: pd.DataFrame) -> float:
        """
        Calculate Max Pain strike price: strike where total dollar value of expiring options is minimized.
        """
        unique_strikes = np.sort(df["strike"].unique())
        if len(unique_strikes) == 0:
            return 0.0

        is_call = df["option_type"].str.lower().str.startswith("c").values
        strikes = df["strike"].values
        ois = df["openInterest"].values

        min_pain = float("inf")
        best_strike = float(unique_strikes[len(unique_strikes) // 2])

        for K_test in unique_strikes:
            # Call payout: max(0, K_test - K) * OI * 100
            call_payouts = np.where(is_call, np.maximum(0.0, K_test - strikes) * ois * 100.0, 0.0)
            # Put payout: max(0, K - K_test) * OI * 100
            put_payouts = np.where(~is_call, np.maximum(0.0, strikes - K_test) * ois * 100.0, 0.0)
            total_pain = np.sum(call_payouts + put_payouts)

            if total_pain < min_pain:
                min_pain = total_pain
                best_strike = float(K_test)

        return best_strike


def compute_dealer_gex_summary(
    options_df: pd.DataFrame,
    spot: float,
    r: float = 0.045,
    q: float = 0.0,
    symbol: str = "",
) -> Dict[str, Any]:
    """
    Convenience wrapper to compute Dealer Gamma Exposure summary.
    """
    engine = DealerGammaEngine(risk_free_rate=r, dividend_yield=q)
    return engine.compute_gex(options_df, spot_price=spot)

