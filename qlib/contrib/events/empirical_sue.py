# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Empirical SUE Normalization Module
==================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates company-specific Standardized Unexpected Earnings (SUE) normalized by
the firm's trailing 12-quarter analyst forecast error standard deviation, avoiding
arbitrary scaling heuristics.
"""

from typing import List, Optional
import numpy as np


def calculate_empirical_sue(
    actual_eps: float,
    consensus_eps: float,
    historical_forecast_errors: Optional[List[float]] = None,
    min_std_floor: float = 0.02,
) -> float:
    """
    Computes empirical Standardized Unexpected Earnings:
        SUE_i = (EPS_actual,i - EPS_consensus,i) / sigma_forecast_error,i

    Parameters
    ----------
    actual_eps : float
        Reported EPS for current period.
    consensus_eps : float
        Mean consensus forecast EPS for current period.
    historical_forecast_errors : Optional[List[float]], optional
        List of historical forecast surprises (actual - consensus) over trailing 4-12 quarters.
    min_std_floor : float, optional
        Minimum denominator floor to prevent division by zero for stable EPS firms, by default 0.02.

    Returns
    -------
    float
        Standardized Z-score of unexpected earnings.
    """
    surprise = float(actual_eps - consensus_eps)

    if historical_forecast_errors and len(historical_forecast_errors) >= 3:
        arr = np.asarray(historical_forecast_errors, dtype=float)
        # Sample standard deviation (ddof=1)
        sigma_error = float(np.std(arr, ddof=1))
        effective_std = max(min_std_floor, sigma_error)
    else:
        # Fallback when historical series is insufficient
        effective_std = max(min_std_floor, abs(consensus_eps) * 0.10)

    sue_z = surprise / effective_std
    return round(float(np.clip(sue_z, -10.0, 10.0)), 4)

