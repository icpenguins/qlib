# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Positive Gamma Squeeze Index (GSI+) Module
==========================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates the raw Positive Gamma Squeeze Index score (GSI+) predicting
next-day to next-week forced dealer buying pressure following earnings beats.
"""

from typing import Dict, Any
import math


def compute_positive_gamma_squeeze_index(
    lir_bull: float,
    sue_score: float,
    call_oi_otm: float,
    put_oi_atm: float,
    short_interest_pct: float,
) -> Dict[str, Any]:
    """
    Computes raw positive gamma squeeze continuous score in [0.0, 100.0].

    Parameters
    ----------
    lir_bull : float
        Liquidity Impact Ratio under positive spot jump scenario (+5% to +10%).
    sue_score : float
        Empirical Standardized Unexpected Earnings score (Z-score).
    call_oi_otm : float
        Open interest in out-of-the-money calls.
    put_oi_atm : float
        Open interest in at-the-money / near-the-money puts.
    short_interest_pct : float
        Short interest as a percentage of float (e.g. 0.15 for 15%).

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'gsi_plus_score': float in [0.0, 100.0]
        - 'is_squeeze_alert': bool (score >= 75.0)
        - 'action': "AGGRESSIVE_BULL_GAMMA_SQUEEZE" or "NORMAL_DRIFT"
        - 'components': dict of input features
    """
    # Bound inputs
    bounded_lir = max(0.0, min(10.0, float(lir_bull)))
    z_sue = math.tanh(float(sue_score) / 2.0)
    asymmetry = min(5.0, float(call_oi_otm) / max(1.0, float(put_oi_atm)))
    si_factor = max(0.0, min(0.50, float(short_interest_pct)))

    logit = (
        1.5 * bounded_lir
        + 1.2 * z_sue
        + 0.6 * asymmetry
        + 3.0 * (si_factor - 0.05)
    )

    # Sigmoid projection to continuous score in [0.0, 100.0]
    gsi_plus = 100.0 / (1.0 + math.exp(-logit))
    bounded_score = round(float(max(0.0, min(100.0, gsi_plus))), 2)

    is_alert = bounded_score >= 75.0
    return {
        "gsi_plus_score": bounded_score,
        "is_squeeze_alert": is_alert,
        "action": "AGGRESSIVE_BULL_GAMMA_SQUEEZE" if is_alert else "NORMAL_DRIFT",
        "time_horizon": "1-Day to 5-Day",
        "components": {
            "lir_bull": round(bounded_lir, 4),
            "z_sue": round(float(z_sue), 4),
            "oi_asymmetry": round(float(asymmetry), 4),
            "short_interest_factor": round(float(si_factor), 4),
        },
    }

