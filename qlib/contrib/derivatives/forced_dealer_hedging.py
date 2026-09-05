# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Forced Dealer Hedging Demand Module
===================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates the forced dealer hedging demand across spot-vol jump scenarios
incorporating empirical IV crush and DTE decay across strike options.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from .gex import BlackScholesGreeks


def calculate_forced_dealer_hedging_demand(
    spot: float,
    df_chain: pd.DataFrame,
    adtv_20: float,
    jump_scenarios: List[float] = None,
    iv_crush_ratio: float = 0.40,
    depth_factor: float = 0.10,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluates net dealer delta hedging share demand D(Delta S) across discrete spot price jumps.

    Parameters
    ----------
    spot : float
        Underlying pre-event spot price S_0.
    df_chain : pd.DataFrame
        Options chain containing 'strike', 'option_type', 'openInterest',
        'impliedVolatility', 'dte', 'delta_call', 'delta_put'.
    adtv_20 : float
        20-day average daily trading volume of underlying equity.
    jump_scenarios : List[float], optional
        Percentage spot jumps to simulate, by default [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15].
    iv_crush_ratio : float, optional
        Proportional drop in implied volatility post-announcement, by default 0.40.
    depth_factor : float, optional
        Fraction of ADTV available in top-of-book / near liquidity, by default 0.10.

    Returns
    -------
    Dict[float, Dict[str, float]]
        Map of jump percentage -> {'shares_demand': float, 'lir': float, 'dollar_demand': float}
    """
    if jump_scenarios is None:
        jump_scenarios = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]

    if df_chain.empty or spot <= 0:
        return {
            dS: {"shares_demand": 0.0, "lir": 0.0, "dollar_demand": 0.0}
            for dS in jump_scenarios
        }

    strikes = df_chain["strike"].values
    ois_call = np.where(df_chain["option_type"] == "call", df_chain["openInterest"].fillna(0).values, 0.0)
    ois_put = np.where(df_chain["option_type"] == "put", df_chain["openInterest"].fillna(0).values, 0.0)
    ivs = np.maximum(0.05, df_chain["impliedVolatility"].fillna(0.30).values)
    dtes = np.maximum(1, df_chain["dte"].fillna(30).values)

    # Base deltas
    tau_0 = np.maximum(1.0 / 365.0, dtes / 365.0)
    if "delta_call" in df_chain.columns and "delta_put" in df_chain.columns:
        delta_0_c = df_chain["delta_call"].fillna(0.5).values
        delta_0_p = df_chain["delta_put"].fillna(-0.5).values
    else:
        delta_0_c = BlackScholesGreeks.calc_delta(spot, strikes, tau_0, ivs, is_call=True)
        delta_0_p = BlackScholesGreeks.calc_delta(spot, strikes, tau_0, ivs, is_call=False)

    results = {}
    for dS in jump_scenarios:
        S_new = spot * (1.0 + dS)
        # Apply firm-specific / estimated IV crush
        sigma_new = np.maximum(0.05, ivs * (1.0 - iv_crush_ratio))
        tau_new = np.maximum(1.0 / 365.0, (dtes - 1.0) / 365.0)

        delta_new_c = BlackScholesGreeks.calc_delta(S_new, strikes, tau_new, sigma_new, is_call=True)
        delta_new_p = BlackScholesGreeks.calc_delta(S_new, strikes, tau_new, sigma_new, is_call=False)

        # Forced dealer re-hedging demand: dealers are net short customer calls and short customer puts
        # As stock rises, customer call delta increases -> dealers must BUY stock to stay delta neutral
        # Customer put delta becomes less negative -> dealers must BUY back existing short stock hedges
        shares_call = np.sum(100.0 * ois_call * (delta_new_c - delta_0_c))
        shares_put = np.sum(100.0 * ois_put * (delta_new_p - delta_0_p))
        net_shares_demand = float(shares_call + shares_put)

        effective_liquidity = max(1.0, adtv_20 * depth_factor)
        lir = abs(net_shares_demand) / effective_liquidity
        dollar_demand = net_shares_demand * S_new

        results[dS] = {
            "shares_demand": round(net_shares_demand, 2),
            "lir": round(float(lir), 4),
            "dollar_demand": round(float(dollar_demand), 2),
        }

    return results

