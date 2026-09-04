# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Liquidity Impact Ratio (LIR) Module
===================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates the ratio of forced dealer hedging share demand to available market liquidity depth.
"""

from typing import Union


def calculate_liquidity_impact_ratio(
    shares_demand: float,
    adtv_20: float,
    depth_factor: float = 0.10,
) -> float:
    """
    Computes the Liquidity Impact Ratio (LIR):
        LIR = |shares_demand| / (ADTV_20 * depth_factor)

    Parameters
    ----------
    shares_demand : float
        Net shares dealers are forced to buy or sell to re-hedge delta neutrality.
    adtv_20 : float
        20-day Average Daily Trading Volume.
    depth_factor : float, optional
        Fraction of daily volume accessible in regular liquidity without breaking books,
        by default 0.10 (10%).

    Returns
    -------
    float
        Normalized liquidity impact ratio. If ADTV <= 0, returns float('inf') if shares_demand != 0 else 0.0.
    """
    if adtv_20 <= 0:
        return float("inf") if abs(shares_demand) > 0 else 0.0

    effective_depth = adtv_20 * depth_factor
    if effective_depth <= 0:
        return float("inf") if abs(shares_demand) > 0 else 0.0

    return float(abs(shares_demand) / effective_depth)

