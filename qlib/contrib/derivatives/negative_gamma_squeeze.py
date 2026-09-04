# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Negative Gamma Squeeze / Liquidation Cascade Index (GSI-) Module
================================================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates the raw Negative Gamma Squeeze / Liquidation Cascade Index score (GSI-)
predicting next-day to next-week forced dealer dumping pressure following earnings misses.
"""

from typing import Dict, Any
import math


def compute_negative_gamma_squeeze_index(
    lir_bear: float,
    sue_score: float,
    spot: float,
    gamma_flip_price: float,
    in_liquidity_void: bool = False,
) -> Dict[str, Any]:
    """
    Computes raw negative gamma squeeze continuous score in [0.0, 100.0].

    Parameters
    ----------
    lir_bear : float
        Liquidity Impact Ratio under negative spot jump scenario (-5% to -10%).
    sue_score : float
        Empirical Standardized Unexpected Earnings score (Z-score).
    spot : float
        Current spot price.
    gamma_flip_price : float
        Estimated Gamma Flip / Volatility Trigger level S*.
    in_liquidity_void : bool, optional
        True if spot is in a low-volume profile node below support, by default False.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'gsi_minus_score': float in [0.0, 100.0]
        - 'is_cascade_alert': bool (score >= 75.0)
        - 'action': "LIQUIDATION_CASCADE_ALERT" or "NORMAL_PULLBACK"
        - 'components': dict of input features
    """
    bounded_lir = max(0.0, min(10.0, float(lir_bear)))
    z_miss = math.tanh(-float(sue_score) / 2.0)
    flip_active = 1.0 if spot < gamma_flip_price else 0.0
    void_active = 1.0 if in_liquidity_void else 0.0

    logit = (
        1.6 * bounded_lir
        + 1.3 * z_miss
        + 1.5 * flip_active
        + 1.2 * void_active
    )

    gsi_minus = 100.0 / (1.0 + math.exp(-logit))
    bounded_score = round(float(max(0.0, min(100.0, gsi_minus))), 2)

    is_alert = bounded_score >= 75.0
    return {
        "gsi_minus_score": bounded_score,
        "is_cascade_alert": is_alert,
        "action": "LIQUIDATION_CASCADE_ALERT" if is_alert else "NORMAL_PULLBACK",
        "time_horizon": "1-Day to 3-Day",
        "components": {
            "lir_bear": round(bounded_lir, 4),
            "z_miss": round(float(z_miss), 4),
            "below_flip_point": bool(flip_active > 0.5),
            "liquidity_void_active": in_liquidity_void,
        },
    }

