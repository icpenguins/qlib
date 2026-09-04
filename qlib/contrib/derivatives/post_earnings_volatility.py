# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Post-Earnings Volatility Calibration Module
===========================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calibrates post-earnings jump size and residual implied volatility from ATM straddle pricing
using jump-plus-crush variance decomposition.
"""

from typing import Tuple, Dict, Any
import math


def calibrate_post_earnings_volatility_surface(
    spot: float,
    atm_straddle_price: float,
    pre_earnings_iv: float,
    realized_21d_vol: float,
    dte_days: int = 7,
) -> Dict[str, Any]:
    """
    Decomposes pre-earnings ATM straddle pricing into a comprehensive post-earnings volatility surface.

    Parameters
    ----------
    spot : float
        Current underlying spot price S_0.
    atm_straddle_price : float
        Combined price of ATM Call + ATM Put (C_ATM + P_ATM).
    pre_earnings_iv : float
        Pre-announcement implied volatility (annualized, e.g. 0.45).
    realized_21d_vol : float
        Trailing 21-day realized historical volatility (annualized, e.g. 0.25).
    dte_days : int, optional
        Days to expiration for front-week option, by default 7.

    Returns
    -------
    Dict[str, Any]
        Dictionary with complete volatility surface calibration metrics:
        - 'spot': float
        - 'atm_straddle_price': float
        - 'pre_earnings_iv': float
        - 'realized_21d_vol': float
        - 'dte_days': int
        - 'expected_jump_pct': float
        - 'event_variance': float
        - 'post_earnings_iv': float
        - 'implied_move_dollars': float
        - 'volatility_crush_pct': float
        - 'volatility_crush_ratio': float
    """
    if spot <= 0 or atm_straddle_price <= 0:
        safe_post_iv = max(0.10, realized_21d_vol)
        return {
            "spot": max(0.0, float(spot)),
            "atm_straddle_price": max(0.0, float(atm_straddle_price)),
            "pre_earnings_iv": float(pre_earnings_iv),
            "realized_21d_vol": float(realized_21d_vol),
            "dte_days": int(dte_days),
            "expected_jump_pct": 0.0,
            "event_variance": 0.0,
            "post_earnings_iv": round(safe_post_iv, 4),
            "implied_move_dollars": 0.0,
            "volatility_crush_pct": 0.0,
            "volatility_crush_ratio": 0.0,
        }

    # Expected absolute jump percentage: E[|dS|] ≈ sqrt(pi/2) * Straddle ≈ 0.79788 * Straddle / Spot
    expected_jump_fraction = (atm_straddle_price * 0.79788) / spot
    expected_jump_pct = expected_jump_fraction * 100.0
    implied_move_dollars = spot * expected_jump_fraction

    # Event variance extraction: Var_total = Var_post * tau + (E[jump])^2
    tau = max(1.0 / 365.0, float(dte_days) / 365.0)
    event_variance = (expected_jump_fraction ** 2) / tau

    # Post-event variance bounded below by realized 21-day volatility variance
    total_pre_variance = max(0.01, pre_earnings_iv ** 2)
    post_variance = max(realized_21d_vol ** 2, total_pre_variance - event_variance)
    post_earnings_iv = math.sqrt(post_variance)

    # Historical IV crush ratio
    if pre_earnings_iv > 1e-4:
        crush_ratio = max(0.0, (pre_earnings_iv - post_earnings_iv) / pre_earnings_iv)
    else:
        crush_ratio = 0.0

    return {
        "spot": round(float(spot), 2),
        "atm_straddle_price": round(float(atm_straddle_price), 2),
        "pre_earnings_iv": round(float(pre_earnings_iv), 4),
        "realized_21d_vol": round(float(realized_21d_vol), 4),
        "dte_days": int(dte_days),
        "expected_jump_pct": round(float(expected_jump_pct), 2),
        "event_variance": round(float(event_variance), 6),
        "post_earnings_iv": round(float(post_earnings_iv), 4),
        "implied_move_dollars": round(float(implied_move_dollars), 2),
        "volatility_crush_pct": round(float(crush_ratio * 100.0), 2),
        "volatility_crush_ratio": round(float(crush_ratio), 4),
    }


def calibrate_post_earnings_volatility(
    spot: float,
    atm_straddle_price: float,
    pre_earnings_iv: float,
    realized_21d_vol: float,
    dte_days: int = 7,
) -> Tuple[float, float]:
    """
    Backwards-compatible convenience wrapper returning (expected_jump_pct, post_earnings_iv).
    """
    surface = calibrate_post_earnings_volatility_surface(
        spot=spot,
        atm_straddle_price=atm_straddle_price,
        pre_earnings_iv=pre_earnings_iv,
        realized_21d_vol=realized_21d_vol,
        dte_days=dte_days,
    )
    return (surface["expected_jump_pct"], surface["post_earnings_iv"])
